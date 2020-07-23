from tqdm import tqdm
import json
import argparse
from warnings import warn
import torch

from data import *
from utils import check_pretrain_model, get_data_ids, GetMetrics, get_dataloaders
from outlayers import DxMap
from run_12ECG_classifier import load_12ECG_model
from evaluate_12ECG_score import load_weights


"""
File follows the general structure of driver.py / train.py
It is used to test our complete model in the end in the same way as driver.py.
"""


if __name__ == '__main__':
    # Experiment parameters
    config_parser = argparse.ArgumentParser(add_help=False)
    # Learning parameters
    config_parser.add_argument('--sample_freq', type=int, default=400,
                               help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    config_parser.add_argument('--batch_size', type=int, default=128,
                               help='batch size, higher for testing (default: 128).')
    config_parser.add_argument('--test_classes', choices=['dset', 'scored'], default='scored_classes',
                               help='what classes are to be used during testing.')
    args, rem_args = config_parser.parse_known_args()
    # System setting
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument('--input_folder', type=str, default='Training_WFDB',
                            help='input folder.')
    sys_parser.add_argument('--dx', type=str, default='./dx',
                            help='Path to folder containing class information.')
    sys_parser.add_argument('--cuda', action='store_true',
                            help='use cuda for computations. (default: False)')
    sys_parser.add_argument('--folder', default=os.getcwd() + '/mdl',
                            help='output folder. If we pass /PATH/TO/FOLDER/ ending with `/`,'
                                 'it creates a folder `output_YYYY-MM-DD_HH_MM_SS_MMMMMM` inside it'
                                 'and save the content inside it. If it does not ends with `/`, the content is saved'
                                 'in the folder provided.')
    settings, unk = sys_parser.parse_known_args(rem_args)
    #  Final parser is needed for generating help documentation
    parser = argparse.ArgumentParser(parents=[sys_parser, config_parser])
    _, unk = parser.parse_known_args(unk)
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    # Set device
    device = torch.device('cuda:0' if settings.cuda else 'cpu')
    # get model folder
    folder = settings.folder
    # Get config
    list_model_folder = os.listdir(folder)
    temp_folder = os.path.join(folder, list_model_folder[0])
    config = os.path.join(temp_folder, 'config.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)
    # write some parameters of config_dict to args
    args.seq_length = config_dict['seq_length']
    args.seed = config_dict['seed']
    args.n_total = config_dict['n_total']
    args.valid_split = config_dict['valid_split']
    # Check if there is pretrained model in the given folder
    config_dict_pretrain_stage, ckpt_pretrain_stage, pretrain_ids = check_pretrain_model(temp_folder)
    pretrain_train_ids, pretrain_valid_ids = pretrain_ids

    tqdm.write("Define dataset...")
    dset = ECGDataset(settings.input_folder, freq=args.sample_freq)
    tqdm.write("Done!")

    tqdm.write("Load test=valid data split...")
    # if pretrained ids are available (not empty)
    if pretrain_valid_ids:
        valid_ids = pretrain_valid_ids
    else:
        _, valid_ids = get_data_ids(dset, args)

        valid_dset = dset.get_subdataset(valid_ids)
    tqdm.write("Done!")

    tqdm.write("Define metrics...")
    # Get all classes in the dataset
    dset_classes = dset.get_classes()
    dx, test_classes = get_dx(dset_classes, args.test_classes, args.test_classes, config_dict['outlayer'], settings.dx)
    weights = load_weights(os.path.join(settings.dx, 'weights.csv'), test_classes)
    NORMAL = '426783006'
    get_metrics = GetMetrics(weights, test_classes.index(NORMAL))
    tqdm.write("Done!")

    tqdm.write("Get dataloaders...")
    valid_loader = ECGBatchloader(valid_dset, dx, batch_size=args.batch_size, length=args.seq_length)
    tqdm.write("Done!")

    # Load model.
    tqdm.write('Loading 12ECG model...')
    model_collection = load_12ECG_model()
    models, dx, out_layer, threshold, model_config, _ = model_collection
    tqdm.write("Done!")

    # progress bar
    test_desc = "Testing"
    test_bar = tqdm(initial=0, leave=True, total=len(valid_loader), desc=test_desc, position=0)
    # collection lists
    n_entries = 0
    all_outputs = []
    all_targets = []
    all_ids = []
    # Compute model predictions
    tqdm.write('Compute model predictions...')
    for i, batch in enumerate(valid_loader):
        with torch.no_grad():
            traces, target, ids, sub_ids = batch
            traces = traces.to(device=device)
            target = target.to(device=device)
            # run models
            logits = []
            # loop over all ensemble models
            for model in models:
                model.eval()
                # update sub_ids for final prediction stage model
                if model[-1]._get_name() == 'RNNPredictionStage':
                    model[-1].update_sub_ids(sub_ids)
                # forward pass
                out = model(traces)
                # collect logits
                logits.append(out)
            # average logits
            mean_logits = torch.mean(torch.stack(logits), dim=0)
            output = out_layer.get_output(mean_logits)

            # append
            all_targets.append(target.detach().cpu().numpy())
            all_outputs.append(output.detach().cpu().numpy())
            all_ids.extend(ids)
            bs = target.size(0)
            n_entries += bs

            # update the progress bar
            test_bar.update(bs)
    test_bar.close()
    tqdm.write('Done')

    tqdm.write('Evaluate predictions...')
    y_score = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    y_true, y_pred, y_score = prepare_for_evaluation(dx, out_layer, y_score, all_targets, ids,
                                                     correction_factor, valid_classes,
                                                     args.pred_stage_type)
    metrics = get_metrics(y_true, y_pred, y_score)
    # print metrics
    message = "Metrics: \tf_beta: {:.3f} \tg_beta: {:.3f} \tgeom_mean: {:.3f} \t challenge: {:.3f}".format(
        metrics['f_beta'], metrics['g_beta'], metrics['geom_mean'], metrics['challenge_metric'])
    tqdm.write(message)
    tqdm.write('Done!')

    # save data
    store_file_name = 'results.json'
    with open(os.path.join(folder, store_file_name), 'w') as f:
        json.dump(metrics, f, indent='\t')
