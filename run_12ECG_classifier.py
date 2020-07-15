#!/usr/bin/env python
import os
import json
import torch

from utils import get_model, check_pretrain_model
from output_layer import OutputLayer, DxClasses, collapse, get_collapse_fun, get_dx
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
        output = out_layer.get_output(mean_logits).mean(dim=0).detach().cpu().numpy()

        # Collapse entries with the same id:
        unique_ids, y_score = collapse(output, ids, fn=get_collapse_fun(config_dict['pred_stage_type']))

        # Get zero one prediction
        y_pred_aux = dx.apply_threshold(y_score, threshold)
        y_pred = dx.target_to_binaryclass(y_pred_aux)
        y_pred = dx.reorganize(y_pred, classes, prob=False)
        y_score = dx.reorganize(y_score, classes, prob=True)
    return y_pred, y_score


def load_12ECG_model():
    # Define model folder
    models_folder = os.path.join(os.getcwd(), 'mdl')

    # running device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # list of models
    models = []

    # get all model folders
    list_model_subfolder = os.listdir(models_folder)
    list_model_subfolder.sort()    # make deterministic

    # loop over all model folders
    for model_subfolder in list_model_subfolder:
        # get path
        path = os.path.join(models_folder, model_subfolder)

        # Get pretrained stage config (if available)
        config_dict_pretrain_stage, _, _ = check_pretrain_model(path, do_print=False)

        # Load check point
        ckpt = torch.load(os.path.join(path, 'model.pth') , map_location=lambda storage, loc: storage)

        # Get config
        config = os.path.join(path, 'config.json')
        with open(config, 'r') as f:
            config_dict = json.load(f)

        # get classes
        classes = 'scored'
        settings_dx = './dx'
        dx, test_classes = get_dx([], classes, classes, config_dict['outlayer'], settings_dx)

        # Define output layer
        out_layer = OutputLayer(config_dict['batch_size'], dx, device)

        # Define threshold
        threshold = ckpt['threshold']

        # Define model
        model = get_model(config_dict, len(dx), config_dict_pretrain_stage)
        model.load_state_dict(ckpt["model"])
        model.to(device=device)

        # append model to list of models
        models.append(model)

    return models, dx, out_layer, threshold, config_dict, device
