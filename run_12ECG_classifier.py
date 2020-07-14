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
    models, dx, out_layer, threshold, config_dict, device = mdl
    # Get sample
    sample = get_sample(header_data, data, config_dict['sample_freq'])
    # Run model
    with torch.no_grad():
        # Get traces
        traces = torch.stack([torch.tensor(s['data'], dtype=torch.float32, device=device) for s in
                              SplitLongSignals(sample, length=config_dict['seq_length'])], dim=0)

        logits = []
        # Apply all models model
        for model in models:
            model.eval()
            output = model(traces)
            logits.append(output)
        # average logits
        mean_logits = torch.mean(torch.stack(logits), dim=0)
        y_score = out_layer.get_output(mean_logits).mean(dim=0).detach().cpu().numpy()

        # Get threshold
        y_pred = (y_score > threshold).astype(int)

        # Reorder according to vector classes
        y_score = dx.reorganize(y_score, classes, prob=True)
        y_pred = dx.reorganize(y_pred, classes, prob=False)
    return y_pred, y_score


def load_12ECG_model():
    # Define model folder
    models_folder = 'mdl/'

    # running device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # list of models
    models = []

    # loop over all folders with models
    list_model_folder = os.listdir(models_folder)
    for model_folder in list_model_folder:

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
        model.to(device)

        # Threshold
        threshold = ckpt['threshold']

        # Output layer
        out_layer = OutputLayer(max(config_dict['batch_size'], 1000), dx, device)

        # append model to list of models
        models.append(model)

    return models, dx, out_layer, threshold, config_dict, device
