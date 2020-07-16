from tqdm import tqdm
import os
import json
import argparse
from warnings import warn
import torch

from data import *
from utils import check_pretrain_model, get_data_ids, GetMetrics, get_dataloaders
from output_layer import get_dx, collapse, get_collapse_fun
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier
from evaluate_12ECG_score import (load_weights)
from driver import load_challenge_data


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
    sys_parser.add_argument('--folder', default=os.getcwd() + '/mdl_nopretrain',
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
    args.batch_size = config_dict['batch_size']
    args.seq_length = config_dict['seq_length']
    args.seed = config_dict['seed']
    args.n_total = config_dict['n_total']
    args.valid_split = config_dict['valid_split']
    args.test_split = config_dict['test_split']
    # Check if there is pretrained model in the given folder
    config_dict_pretrain_stage, ckpt_pretrain_stage, pretrain_ids = check_pretrain_model(temp_folder)
    pretrain_train_ids, pretrain_valid_ids, pretrain_test_ids = pretrain_ids

    tqdm.write("Define dataset...")
    dset = ECGDataset(settings.input_folder, freq=args.sample_freq)
    tqdm.write("Done!")

    tqdm.write("Load data...")
    # if pretrained ids are available (not empty)
    if pretrain_train_ids and pretrain_valid_ids and pretrain_test_ids:
        test_ids = pretrain_test_ids
    else:
        _, _, test_ids = get_data_ids(dset, args)
    tqdm.write("Done!")

    tqdm.write("Define metrics...")
    # Get all classes in the dataset
    dset_classes = dset.get_classes()
    dx, test_classes = get_dx(dset_classes, args.test_classes, args.test_classes, config_dict['outlayer'], settings.dx)
    weights = load_weights(os.path.join(settings.dx, 'weights.csv'), test_classes)
    NORMAL = '426783006'
    get_metrics = GetMetrics(weights, test_classes.index(NORMAL))
    tqdm.write("Done!")

    # Load model.
    tqdm.write('Loading 12ECG model...')
    model = load_12ECG_model()
    tqdm.write("Done!")

    # get paths of all files to loop over them (no batch-wise but same way as in driver.py)
    path = os.path.join(os.getcwd(), settings.input_folder)
    file_paths = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.mat' in file and file[:-4] in test_ids:
                file_paths.append(os.path.join(r, file))

    # progress bar
    test_desc = "Testing"
    test_bar = tqdm(initial=0, leave=True, total=len(file_paths), desc=test_desc, position=0)
    # collection lists
    y_true_list = []
    y_pred_list = []
    y_score_list = []
    # Compute model predictions
    tqdm.write('Compute model predictions...')
    for i, file_path in enumerate(file_paths):
        with torch.no_grad():
            # get folder and file name
            file_name = os.path.split(file_path)[-1]
            file_folder = os.path.split(os.path.split(file_path)[0])[-1]

            # loop over all test files
            data, header_data = load_challenge_data(file_path)
            y_pred, y_score = run_12ECG_classifier(data, header_data, test_classes, model)

            # get target / true value
            idx = dset.input_file.index(os.path.join(file_folder, file_name))
            header = dset._getsample(13, only_header=True)
            target = dx.get_target_from_labels(header['labels']).reshape(1, -1)
            y_true1 = dx.target_to_binaryclass(target)
            y_true2 = dx.reorganize(y_true1, test_classes).reshape(1, -1)

            # collect the predictions, scores and true labels
            y_pred_list.append(y_pred)
            y_score_list.append(y_score)
            y_true_list.append(y_true2)

            # update the progress bar
            test_bar.update(1)
    test_bar.close()
    tqdm.write('Done')
    # lists to arrays
    y_p = np.concatenate(y_pred_list, axis=0)
    y_s = np.concatenate(y_score_list, axis=0)
    y_t = np.concatenate(y_true_list, axis=0)
    # Get metrics
    metrics = get_metrics(y_t, y_p, y_s)
    metrics = get_metrics(y_t, y_p, y_s)
    metrics = get_metrics(y_t, y_p, y_s)
    metrics = get_metrics(y_t, y_p, y_s)
    # print metrics
    message = "Metrics: \tf_beta: {:.3f} \tg_beta: {:.3f} \tgeom_mean: {:.3f} \t challenge: {:.3f}".format(
        metrics['f_beta'], metrics['g_beta'], metrics['geom_mean'], metrics['challenge_metric'])
    tqdm.write(message)

    # save data
    store_file_name = 'results.json'
    with open(os.path.join(folder, store_file_name), 'w') as f:
        json.dump(metrics, f, indent='\t')
