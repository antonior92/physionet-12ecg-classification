#!/usr/bin/env python
import torch
import pandas as pd

from train import evaluate, compute_logits
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
    ID = 42  # 42 is arbitrary... Any other int would seve the purpose here
    ids, subids = [[ID]] * l, [[si] for si in range(l)]
    # Get loader (which, unlike in train.py, is just a list)
    valid_loader = list(zip(traces, ids, subids))
    # Compute logits
    all_logits, ids, sub_ids = compute_logits(-1, model, valid_loader, device)
    # Compute prediction and score
    y_pred, y_score = evaluate(ids, all_logits, out_layer, dx, correction_factor, [ID],
                               classes, 1, config_dict['combination_strategy'],
                               config_dict['predict_before_collapse'], device,
                               config_dict['scale_by_n_predictions'])
    return list(y_pred.flatten()), list(y_score.flatten()), classes


def load_12ECG_model(folder):
    # Check if there is pretrained model in the given folder
    config_dict_pretrain_stage, ckpt_pretrain_stage, _, _ = check_pretrain_model(folder)
    # Check if there is a model in the given folder
    config_dict, ckpt, dx, out_layer, correction_factor, _, _, _ = check_model(folder)
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
