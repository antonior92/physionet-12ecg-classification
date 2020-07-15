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
    args.batch_size = config_dict['batch_size']
    args.seq_length = config_dict['seq_length']
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

    """tqdm.write("Get dataloaders...")
    test_loader = get_dataloaders(dset, test_ids, args, dx)
    tqdm.write("Done!")"""

    # Load model.
    tqdm.write('Loading 12ECG model...')
    model = load_12ECG_model()
    tqdm.write("Done!")

    # get paths of all files to loop over them (no batch-wise but same way as in driver.py)
    path = os.path.join(os.getcwd(),settings.input_folder)
    file_paths = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.mat' in file and file[:-4] in test_ids:
                file_paths.append(os.path.join(r, file))

    # Compute model predictions
    tqdm.write('Compute model predictions...')
    for i, file_name in enumerate(file_paths):
        with torch.no_grad():

            # loop over all test files
            data, header_data = load_challenge_data(file_name)
            y_pred, y_score = run_12ECG_classifier(data, header_data, test_classes, model)

            #  _, y_score, all_targets, ids = evaluate(


            all_targets = []
            ids = []

            # Get metrics
            y_true = dx.target_to_binaryclass(all_targets)
            _, y_true = collapse(y_true, ids, fn=lambda y: y[0, :], unique_ids=unique_ids)
            y_true = dx.reorganize(y_true, test_classes)
            metrics = get_metrics(y_true, y_pred, y_score)

        """num_files = len(input_files)
        
        for i, f in enumerate(input_files):
            print('    {}/{}...'.format(i + 1, num_files))
            tmp_input_file = os.path.join(input_directory, f)
            data, header_data = load_challenge_data(tmp_input_file)
            current_label, current_score = run_12ECG_classifier(data, header_data, classes, model)
            # Save results."""
    tqdm.write('Done')

    # evaluate
    tqdm.write('Evaluate predictions...')
    # TODO
    tqdm.write('Done')
