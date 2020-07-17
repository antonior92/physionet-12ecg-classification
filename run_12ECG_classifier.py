#!/usr/bin/env python
import os
import json
import torch
import numpy as np

from utils import get_model, check_pretrain_model
from output_layer import OutputLayer, collapse, get_collapse_fun, get_dx
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
        num_traces = traces.size(0)

        all_logits = []
        # Apply all models model
        for model in models:
            model.eval()
            sub_id = [0]
            all_outputs = []
            # run trace sequentially (similar to train.py where they are run in sequential batches)
            for i, trace in enumerate(traces):
                trace = trace.unsqueeze(dim=0)
                # update sub_ids for final prediction stage model
                if model[-1]._get_name() == 'RNNPredictionStage':
                    model[-1].update_sub_ids(sub_id)
                # forward pass over model
                output = model(trace)
                # append
                all_outputs.append(output.detach().cpu())
                # increase sub_id
                sub_id = [x+1 for x in sub_id]
            all_logits.append(all_outputs)

        # here we average the model outputs before putting each single sequence in the output layer.
        all_out = []
        for i in range(num_traces):
            # average logits
            mean_logits = torch.mean(torch.stack([elem[i] for elem in all_logits]), dim=0)
            out = out_layer.get_output(mean_logits)
            # append
            all_out.append(out.detach().cpu().numpy())
        y_score = np.concatenate(all_out)

        # all have the same id, choose randomly name '42'
        ids = ['42' for i in range(num_traces)]

        # Collapse entries with the same id
        _, y_score = collapse(y_score, ids, fn=get_collapse_fun(config_dict['pred_stage_type']))

        # get y_score of shape (1,n_classes)
        y_score1 = y_score.reshape(1, -1)

        # Get zero one prediction
        y_pred_aux = dx.apply_threshold(y_score1, threshold)
        y_pred1 = dx.target_to_binaryclass(y_pred_aux).reshape(1, -1)
        y_pred2 = dx.reorganize(y_pred1, classes, prob=False)
        y_score2 = dx.reorganize(y_score1, classes, prob=True)
    return y_pred2, y_score2


def load_12ECG_model():
    # Define model folder
    models_folder = os.path.join(os.getcwd(), 'mdl_nopretrain_rnn')

    # running device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # list of models
    models = []

    # get all model folders
    list_model_subfolder = [filename for filename in os.listdir(models_folder) if
                            os.path.isdir(os.path.join(models_folder, filename))]
    list_model_subfolder.sort()  # make deterministic

    # loop over all model folders
    for model_subfolder in list_model_subfolder:
        # get path
        path = os.path.join(models_folder, model_subfolder)

        # Get pretrained stage config (if available)
        config_dict_pretrain_stage, _, _ = check_pretrain_model(path, do_print=False)

        # Load check point
        ckpt = torch.load(os.path.join(path, 'model.pth'), map_location=lambda storage, loc: storage)

        # Get config
        config = os.path.join(path, 'config.json')
        with open(config, 'r') as f:
            config_dict = json.load(f)

        # get classes
        classes = 'scored'
        settings_dx = './dx'
        dx, test_classes = get_dx([], classes, classes, config_dict['outlayer'], settings_dx)

        # Define output layer
        # we need bigger bs since some exams have ~170*4096 points and then we get problems in output layer
        bs = 256  # config_dict['batch_size']
        out_layer = OutputLayer(bs, dx, device)

        # Define threshold
        threshold = ckpt['threshold']

        # Define model
        model = get_model(config_dict, len(dx), config_dict_pretrain_stage)
        model.load_state_dict(ckpt["model"])
        model.to(device=device)

        # append model to list of models
        models.append(model)

    return models, dx, out_layer, threshold, config_dict, device
