import argparse
import torch
import random
import pandas as pd
from warnings import warn
from tqdm import tqdm
from outlayers import DxMap, outlayer_from_str

from data import *
from utils import set_output_folder, check_pretrain_model, get_data_ids, \
    write_data_ids, get_model, GetMetrics, get_output_layer, \
    get_correction_factor, check_model, save_config, initialize_history, save_history, \
    print_message, get_targets, update_history
from outlayers import collapse, get_collapse_fun


# %% Train model
def train(ep, model, optimizer, train_loader, out_layer, device, shuffle):
    model.train()
    total_loss = 0
    n_entries = 0
    # training progress bar
    if shuffle:
        train_loader.shuffle()
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(train_loader),
                     desc=train_desc.format(ep, 0), position=0)
    for i, batch in enumerate(train_loader):
        traces, target, ids, sub_ids = batch
        traces = traces.to(device=device)
        target = target.to(device=device)
        # update sub_ids for final prediction stage model
        if model[-1]._get_name() == 'RNNPredictionStage':
            model[-1].update_sub_ids(sub_ids)
        # Reinitialize grad
        model.zero_grad()
        # Forward pass
        output = model(traces)
        # softmax and sigmoid layer implicitly contained in the loss
        loss = out_layer.loss(output, target)
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        total_loss += loss.detach().cpu().numpy()
        bs = target.size(0)
        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(bs)
    train_bar.close()
    # reset prediction stage variables
    if model[-1]._get_name() == 'RNNPredictionStage':
        model[-1].reset()

    return total_loss / n_entries


def compute_logits(ep, model, valid_loader, device):
    model.eval()
    n_entries = 0
    all_logits = []
    all_ids = []
    eval_desc = "Epoch {0:2d}: valid                 " if ep >= 0 else "valid"
    eval_bar = tqdm(initial=0, leave=True, total=len(valid_loader),
                    desc=eval_desc.format(ep, 0), position=0)
    for i, batch in enumerate(valid_loader):
        with torch.no_grad():
            traces, ids, sub_ids = batch
            traces = traces.to(device=device)
            # update sub_ids for final prediction stage model
            if model[-1]._get_name() == 'RNNPredictionStage':
                model[-1].update_sub_ids(sub_ids)
            # Forward pass
            logits = model(traces)
            # append
            all_logits.append(logits.detach().cpu())
            all_ids.extend(ids)
            bs = traces.size(0)
            n_entries += bs
            # Print result
            eval_bar.update(bs)
    eval_bar.close()
    # reset prediction stage variables
    if model[-1]._get_name() == 'RNNPredictionStage':
        model[-1].reset()
    return torch.cat(all_logits), all_ids


def output_from_logits(out_layer, all_logits, batch_size, device):
    n_entries = all_logits.shape[0]
    all_outputs = []
    n_batches = int(np.ceil(n_entries / batch_size))
    end = 0
    for i in range(n_batches):
        start = end
        end = min(start + batch_size, n_entries)
        logits = all_logits[start:end, :]
        logits.to(device)
        # Outputs
        outputs = out_layer(logits)
        all_outputs.append(outputs.detach().cpu())
    return torch.cat(all_outputs)


def evaluate(ep, model, out_layer, dx, correction_factor, targets_ids, valid_loader,
             classes, batch_size, pred_stage_type, device):
    # compute_logits
    all_logits, ids = compute_logits(ep, model, valid_loader, device)
    y_score = output_from_logits(out_layer, all_logits,  batch_size, device)
    y_score = y_score.numpy()
    # Collapse entries with the same id:
    _, y_score = collapse(y_score, ids, fn=get_collapse_fun(pred_stage_type), unique_ids=targets_ids)
    # Correct for class imbalance
    y_score = y_score * correction_factor
    # Get zero one prediction
    y_pred_aux = out_layer.get_prediction(y_score)
    y_pred = dx.prepare_target(y_pred_aux, classes)
    y_score = dx.prepare_probabilities(y_score, classes)
    return y_pred, y_score


# TODO:1) Include command line option to save logits.
#  2) Add something to deal with the classes the same way they do in evaluate_12ECG_score.py
#  (i.e. deal with repeated classes)
#  3) Add validation loss to metrics. Now validation loss is not computed at all due to a recent refactoring
#  4) Merge check_model and check_pretrain_model into one single function?
if __name__ == '__main__':
    # Experiment parameters
    config_parser = argparse.ArgumentParser(add_help=False)
    # Learning parameters
    config_parser.add_argument('--seed', type=int, default=2,
                               help='random seed for number generator (default: 2)')
    config_parser.add_argument('--dont_shuffle', action='store_true',
                               help='dont shuffle training samples each epoch.')
    config_parser.add_argument('--epochs', type=int, default=200,
                               help='maximum number of epochs (default: 70)')
    config_parser.add_argument('--sample_freq', type=int, default=400,
                               help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    config_parser.add_argument('--seq_length', type=int, default=4096,
                               help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                    'to fit into the given size. (default: 4096)')
    config_parser.add_argument('--batch_size', type=int, default=32,
                               help='batch size (default: 32).')
    config_parser.add_argument('--lr', type=float, default=0.001,
                               help='learning rate (default: 0.001)')
    config_parser.add_argument('--milestones', nargs='+', type=int,
                               default=[75, 125, 175],
                               help='milestones for lr scheduler (default: [100, 200])')
    config_parser.add_argument("--lr_factor", type=float, default=0.1,
                               help='reducing factor for the lr in a plateeu (default: 0.1)')
    # Pretrain Model parameters
    config_parser.add_argument('--pretrain_output_size', type=int, default=64,
                               help='The output of the pretrained model goes through a linear layer, which outputs'
                                    'a tensor with the given number of features (default: 64).')
    config_parser.add_argument('--finetuning', type=bool, default=False,
                               help='when there is a pre-trained model, by default it '
                                    'freezes the weights of the pre-trained model, but with this option'
                                    'these weight will be fine-tuned during training. Default is False')
    config_parser.add_argument('--eval_transformer', type=bool, default=False,
                               help='Evaluates the transformer. If true a small MLP classifier is chosen instead of '
                                    'the ResNet + prediction stage (default: False).')
    # Model parameters
    config_parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                               help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    config_parser.add_argument('--net_seq_length', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                               help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    config_parser.add_argument('--dropout_rate', type=float, default=0.5,
                               help='dropout rate (default: 0.5).')
    config_parser.add_argument('--kernel_size', type=int, default=17,
                               help='kernel size in convolutional layers (default: 17).')
    # Final Predictor parameters
    config_parser.add_argument('--pred_stage_type', choices=['lstm', 'gru', 'rnn', 'mean', 'max'], default='mean',
                               help='type of prediction stage model: lstm, gru, rnn, mean (default), max.')
    config_parser.add_argument('--pred_stage_n_layer', type=int, default=1,
                               help='number of rnn layers in prediction stage (default: 2).')
    config_parser.add_argument('--pred_stage_hidd', type=int, default=400,
                               help='size of hidden layer in prediction stage rnn (default: 30).')
    config_parser.add_argument('--valid_classes', choices=['dset', 'scored'], default='scored_classes',
                               help='what classes are to be used during evaluation.')
    # System settings
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument('--train_classes', choices=['dset', 'scored'], default='scored_classes',
                               help='what classes are to be used during training.')
    sys_parser.add_argument('--only_test', action='store_true',
                               help='just evaluate the model whithout any training step.')
    sys_parser.add_argument('--out_layer', type=str, default='softmax',
                               help='what is the type used for the output layer. Options are '
                                    '(sigmoid, softmax) or the name of .txt files in dx with the correct format.')
    sys_parser.add_argument('--valid_split', type=float, default=0.30,
                            help='fraction of the data used for validation (default: 0.3).')
    sys_parser.add_argument('--n_total', type=int, default=-1,
                            help='number of samples to be used during training. By default use '
                                 'all the samples available. Useful for quick tests.')
    sys_parser.add_argument('--expected_class_distribution', choices=['uniform', 'train'], default='uniform',
                            help="The expected distribution of classes. If 'uniform' consider uniform distribution."
                                 "If 'train' consider the distribution observed in the training dataset."
                                 "The classifier will be corrected to account for this prior information"
                                 "on the distribution of the classes.")
    sys_parser.add_argument('--input_folder', type=str, default='Training_WFDB',
                            help='input folder.')
    sys_parser.add_argument('--dx', type=str, default='./dx',
                            help='Path to folder containing class information.')
    sys_parser.add_argument('--no_cuda', action='store_true',
                            help='do not use cuda for computations, even if available. (default: False)')
    sys_parser.add_argument('--save_last', action='store_true',
                            help='if true save the last model, otherwise, save the best model'
                                 '(according to challenge metric).')
    sys_parser.add_argument('--folder', default=os.getcwd() + '/',
                            help='output folder. If we pass /PATH/TO/FOLDER/ ending with `/`,'
                                 'it creates a folder `output_YYYY-MM-DD_HH_MM_SS_MMMMMM` inside it'
                                 'and save the content inside it. If it does not ends with `/`, the content is saved'
                                 'in the folder provided.')

    settings, rem_args = sys_parser.parse_known_args()

    # Set device
    use_cuda = not settings.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    if use_cuda:
        tqdm.write("Using gpu!")
    # Generate output folder if needed and save config file
    folder = set_output_folder(settings.folder)
    # Check if there is pretrained model in the given folder
    config_dict_pretrain_stage, ckpt_pretrain_stage, pretrain_ids, _ = check_pretrain_model(folder)
    pretrain_train_ids, pretrain_valid_ids = pretrain_ids
    # Check if there is a model in the given folder
    config_dict, ckpt, dx, out_layer, correction_factor, ids, history = check_model(folder)
    train_ids, valid_ids = ids
    # Set defaults according to the configuration file inside the folder
    if config_dict is not None:
        config_parser.set_defaults(**config_dict)
    # get configurations
    args, unk = config_parser.parse_known_args(rem_args)
    #  Final parser is needed for generating help documentation
    parser = argparse.ArgumentParser(parents=[sys_parser, config_parser])
    _, unk = parser.parse_known_args(unk)
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    # Save in folder
    if config_dict is None:
        save_config(folder, args)
    # Set seed
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    # Define only_test variable
    only_test = settings.only_test or settings.valid_split >= 1.0

    tqdm.write("Define dataset...")
    dset = ECGDataset.from_folder(settings.input_folder, freq=args.sample_freq)
    if len(dset) == 0:
        raise ValueError('Dataset is empty.')
    tqdm.write("Done!")

    tqdm.write("Define train and validation splits...")
    # if pretrained ids are available (not empty)
    if train_ids and valid_ids:
        pass
    elif pretrain_train_ids and pretrain_valid_ids:
        train_ids = pretrain_train_ids
        valid_ids = pretrain_valid_ids
    else:
        train_ids, valid_ids = get_data_ids(dset, settings.valid_split, settings.n_total, rng)
    # Save train, validation ids
    write_data_ids(folder, train_ids, valid_ids)
    # Get dataset
    train_dset = dset.get_subdataset(train_ids)
    valid_dset = dset.get_subdataset(valid_ids)
    tqdm.write("Done!")

    tqdm.write("Define output layer...")
    # Get all classes in the dataset
    dset_classes = dset.get_classes()
    # Get all classes to be scored
    df = pd.read_csv(os.path.join(settings.dx, 'dx_mapping_scored.csv'))
    scored_classes = [str(c) for c in list(df['SNOMED CT Code'])]
    # Get classes to be taken under consideration
    valid_classes = dset_classes if args.valid_classes == 'dset' else scored_classes
    # Get outlayer and map
    if (dx is None) and (out_layer is None):
        if settings.out_layer in ['softmax', 'sigmoid']:
            # Get output layer classes
            train_classes = dset_classes if settings.train_classes == 'dset' else scored_classes
            out_layer = outlayer_from_str(settings.out_layer)
            dx = DxMap.infer_from_out_layer(train_classes, out_layer)
        else:
            path_to_outmap = os.path.join(settings.dx, settings.out_layer + '.txt')
            out_layer, dx = get_output_layer(path_to_outmap)
        with open(os.path.join(folder, 'out_layer.txt'), 'w') as f:
            f.write('{:}\n{:}'.format(out_layer, dx))
    else:
        tqdm.write("\tUsing pre-specified outlayer!")
    tqdm.write('\t{:}'.format(out_layer))
    tqdm.write("Done!")

    if len(valid_dset) > 0:
        tqdm.write("Get targets for validation...")
        targets = get_targets(valid_dset, dx)
        tqdm.write("Done!")

    tqdm.write("Define  correction factor (for class imbalance) ...")
    if correction_factor is None:
        correction_factor = get_correction_factor(dset, dx, settings.expected_class_distribution)
        np.savetxt(os.path.join(folder, 'correction_factor.txt'), correction_factor, fmt='%.10f')
    else:
        tqdm.write("\tUsing pre-specified correction factor!")
    tqdm.write("\tCorrection factor = {:}".format(str(list(correction_factor))))
    tqdm.write("Done!")

    tqdm.write("Define metrics...")
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    get_metrics = GetMetrics(os.path.join(settings.dx, 'weights.csv'), valid_classes, normal_class, equivalent_classes)
    tqdm.write("Done!")

    tqdm.write("Get dataloaders...")
    train_loader = ECGBatchloader(train_dset, dx, batch_size=args.batch_size,
                                  length=args.seq_length, seed=args.seed)
    valid_loader = ECGBatchloader(valid_dset, batch_size=args.batch_size, length=args.seq_length)
    tqdm.write("\t train:  {:d} ({:2.2f}%) ECG records divided into {:d} samples of fixed length"
               .format(len(train_dset), 100 * len(train_dset) / len(dset), len(train_loader))),
    tqdm.write("\t valid:  {:d} ({:2.2f}%) ECG records divided into {:d} samples of fixed length"
               .format(len(valid_dset), 100 * len(valid_dset) / len(dset), len(valid_loader)))
    tqdm.write("Done!")

    tqdm.write("Define model...")
    model = get_model(vars(args), len(dx), config_dict_pretrain_stage, ckpt_pretrain_stage)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    model.to(device=device)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_factor)
    if ckpt is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    tqdm.write("Done!")

    tqdm.write("Start training...")
    start_epoch = 0 if ckpt is None else ckpt['epoch']+1
    best_challenge_metric = -np.Inf
    if history is None:
        history = initialize_history()
    else:
        try:
            best_challenge_metric = max(history[history['epoch'] == ckpt['epoch']]['challenge_metric'])
        except:
            pass
        tqdm.write("\tContinuing from epoch {:}...".format(start_epoch))

    def compute_metrics(ep=-1):
        # Evaluate
        y_pred, y_score = evaluate(ep, model, out_layer, dx, correction_factor, valid_ids, valid_loader,
                                   valid_classes, args.batch_size, args.pred_stage_type, device)
        # Get metrics
        y_true = dx.prepare_target(targets, valid_classes)
        # Compute metrics
        metrics = get_metrics(y_true, y_pred, y_score)
        return metrics

    # run over all epochs
    epochs = args.epochs
    if only_test:
        metrics = compute_metrics()
        # Print
        print_message(metrics)
        epochs = start_epoch  # To avoid entering the loop
    for ep in range(start_epoch, epochs):
        # train
        train_loss = train(ep, model, optimizer, train_loader, out_layer, device, not args.dont_shuffle)
        metrics = compute_metrics(ep) if len(valid_loader) > 0 else None
        # Get learning rate
        learning_rate = 0
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Print message
        print_message(metrics, ep, learning_rate, train_loss)
        # Save history
        history = update_history(history, learning_rate, train_loss, metrics, ep)
        save_history(folder, history)
        # Call optimizer step
        scheduler.step()
        # Check if it is best metric
        is_best_metric = best_challenge_metric < metrics['challenge_metric'] if metrics is not None else False
        # Save best model
        if (settings.save_last and (ep == args.epochs - 1)) or is_best_metric:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()},
                       os.path.join(folder, 'model.pth'))
            tqdm.write("Save model!")
        # Update best metric
        if is_best_metric:
            best_challenge_metric = metrics['challenge_metric']
