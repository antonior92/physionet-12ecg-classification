#!/usr/bin/env python

import os
import json
import torch
from models.resnet import ResNet1d
from ecg_dataset import get_sample, split_long_signals, CLASSES


def run_12ECG_classifier(data, header_data, classes, mdl):

    # Get model specifications
    model, output_layer, threshold, config_dict, device = mdl

    # Get sample
    sample = get_sample(data, header_data, config_dict['sample_freq'])

    # Get traces
    model.eval()

    # Run model
    with torch.no_grad():
        traces = torch.stack([torch.tensor(s['data'], dtype=torch.float32, device=device) for s in
                              split_long_signals(sample, length=config_dict['seq_length'])], dim=0)
        logits = model(traces).mean(dim=0)
        y_score = output_layer(logits).detach().cpu().numpy()
        y_pred = (y_score > threshold).astype(int)

    return y_pred, y_score


def load_12ECG_model():
    # Define model folder
    model_folder = 'mdl/'

    # Load check point
    ckpt = torch.load(os.path.join(model_folder, 'model.pth'), map_location=lambda storage, loc: storage)

    # Get config
    config = os.path.join(model_folder, 'config.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)

    # Define model
    N_LEADS = 12
    n_classes = len(CLASSES)
    model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                     blocks_dim=list(zip(config_dict['net_filter_size'],
                                     config_dict['net_seq_lengh'])),
                     n_classes=n_classes,
                     kernel_size=config_dict['kernel_size'],
                     dropout_rate=config_dict['dropout_rate'])
    model.load_state_dict(ckpt["model"])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Output layer
    output_layer = torch.nn.Sigmoid()

    # Threshold
    threshold = ckpt['threshold']

    return model, output_layer, threshold, config_dict, device
