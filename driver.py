import os
import json
import torch
import argparse
import datetime
from warnings import warn
from ecg_dataset import *
from tqdm import tqdm
from models.resnet import ResNet1d
from metrics import get_threshold, get_metrics
from train import evaluate


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='./Training_WFDB',
                            help='input folder.')
    parser.add_argument('--cuda', action='store_true',
                            help='use cuda for computations. (default: False)')
    parser.add_argument('--folder', default=os.path.join(os.getcwd(), 'mdl/'),
                        help='folder to load mdl config and weights')
    args, unk = parser.parse_known_args()

    # Load check point
    ckpt = torch.load(os.path.join(args.folder, 'model.pth'), map_location=lambda storage, loc: storage)

    # Get config
    config = os.path.join(args.folder, 'config.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)

    # Set device
    device = torch.device('cuda:0' if args.cuda else 'cpu')

    # Get dataset
    n_classes = len(CLASSES)
    dset = ECGDataset(args.input_folder, freq=config_dict['sample_freq'])[:10]

    # Define model
    N_LEADS = 12
    model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                     blocks_dim=list(zip(config_dict['net_filter_size'],
                                     config_dict['net_seq_lengh'])),
                     n_classes=n_classes,
                     kernel_size=config_dict['kernel_size'],
                     dropout_rate=config_dict['dropout_rate'])
    model.load_state_dict(ckpt["model"])
    model.to(device=device)

    # Define output layer
    output_layer = torch.nn.Sigmoid()

    # Evaluate model
    y_score, ids = evaluate(0, model, dset, output_layer,
                            n_classes, device, config_dict['seq_length'])



