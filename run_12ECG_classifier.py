#!/usr/bin/env python
import torch
import pandas as pd

from train import evaluate
from utils import check_pretrain_model, check_model, get_model
from data import get_sample
from data.ecg_dataloader import SplitLongSignals


def run_12ECG_classifier(data, header_data, mdl):
    # Get model specifications
    model, dx, out_layer, correction_factor, classes, config_dict, device = mdl
    # Get sample
    sample = get_sample(header_data, data, config_dict['sample_freq'])
    # Get traces
    traces = [torch.tensor(s['data'], dtype=torch.float32, device=device)[None, :, :] for s in
              SplitLongSignals(sample, length=config_dict['seq_length'])]
    # Assign ids and subids
    l = len(traces)
    ids, subids = [[sample['id']]] * l, [[si] for si in range(l)]
    # Get loader (which, unlike in train.py, is just a list)
    valid_loader = list(zip(traces, ids, subids))
    # Compute prediction and score
    y_pred, y_score = evaluate(-1, model, out_layer, dx, correction_factor, ids, valid_loader,
                               classes, 1, config_dict['pred_stage_type'], device)
    return y_pred, y_score, classes


def load_12ECG_model(folder):
    # Check if there is pretrained model in the given folder
    config_dict_pretrain_stage, ckpt_pretrain_stage, _, _ = check_pretrain_model(folder)
    # Check if there is a model in the given folder
    config_dict, ckpt, dx, out_layer, correction_factor, _, _ = check_model(folder)
    # running device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get all classes to be scored
    df = pd.read_csv('dx/dx_mapping_scored.csv')
    classes = [str(c) for c in list(df['SNOMED CT Code'])]
    # get model
    model = get_model(config_dict, len(dx), config_dict_pretrain_stage, ckpt_pretrain_stage)
    model.load_state_dict(ckpt["model"])
    model.to(device=device)
    return model, dx, out_layer, correction_factor, classes, config_dict, device
