#!/usr/bin/env python
import os
import json
import torch
from train import get_model
from output_layer import OutputLayer, DxClasses
from data import (get_sample)
from data.ecg_dataloader import SplitLongSignals


def run_12ECG_classifier(data, header_data, classes, mdl):
    # Get model specifications
    model, dx, out_layer, threshold, config_dict, device = mdl
    # Get sample
    sample = get_sample(header_data, data, config_dict['sample_freq'])
    # Get trace
    model.eval()
    # Run model
    with torch.no_grad():
        # Get traces
        traces = torch.stack([torch.tensor(s['data'], dtype=torch.float32, device=device) for s in
                              SplitLongSignals(sample, length=config_dict['seq_length'])], dim=0)
        # Apply model
        logits = model(traces)
        y_score = out_layer.get_output(logits).mean(dim=0).detach().cpu().numpy()

        # Get threshold
        y_pred = (y_score > threshold).astype(int)

        # Reorder according to vector classes
        y_score = dx.reorganize(y_score, classes, prob=True)
        y_pred = dx.reorganize(y_pred, classes, prob=False)
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

    # get classes
    dx = DxClasses.read_csv(os.path.join(model_folder, 'classes.txt'))

    # Get pretrained stage config (if available)
    try:
        config_pretrain_stage = os.path.join(model_folder, 'pretrain_config.json')
        with open(config_pretrain_stage, 'r') as f:
            config_dict_pretrain_stage = json.load(f)
    except:
        config_dict_pretrain_stage = None

    # Define model
    model = get_model(config_dict, len(dx), config_dict_pretrain_stage)
    model.load_state_dict(ckpt["model"])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Threshold
    threshold = ckpt['threshold']

    # Output layer
    out_layer = OutputLayer(max(config_dict['batch_size'], 1000), dx, device)

    return model, dx, out_layer, threshold, config_dict, device
