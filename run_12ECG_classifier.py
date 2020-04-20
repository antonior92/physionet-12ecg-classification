#!/usr/bin/env python
import os
import json
import torch
from models.resnet import ResNet1d
from output_layer import OutputLayer
from ecg_dataset import (get_sample, split_long_signals, CLASSES, mututally_exclusive,
                         add_normal_column)


def run_12ECG_classifier(data, header_data, classes, mdl):
    # Get model specifications
    model, out_layer, threshold, config_dict, device = mdl
    # Get sample
    sample = get_sample(data, header_data, config_dict['sample_freq'])
    # Get trace
    model.eval()
    # Run model
    with torch.no_grad():
        # Get traces
        traces = torch.stack([torch.tensor(s['data'], dtype=torch.float32, device=device) for s in
                              split_long_signals(sample, length=config_dict['seq_length'])], dim=0)
        # Apply model
        logits = model(traces)
        y_score = out_layer.get_output(logits).mean(dim=0).detach().cpu().numpy()

        # Get threshold
        y_pred = (y_score > threshold).astype(int)

        # Add column corresponding to normal
        y_score = add_normal_column(y_score, prob=True)
        y_pred = add_normal_column(y_pred, prob=False)

        # Reorder according to vector classes
        current_order = CLASSES + ['Normal']
        dict_current_order = dict(zip(current_order, range(len(current_order))))
        new_idx = [dict_current_order[c] for c in classes]
        y_score = y_score[new_idx]
        y_pred = y_pred[new_idx]

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

    # Threshold
    threshold = ckpt['threshold']

    # Output layer
    out_layer = OutputLayer(config_dict['batch_size'], mututally_exclusive, device)

    return model, out_layer, threshold, config_dict, device
