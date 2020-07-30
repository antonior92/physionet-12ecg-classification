import os
import datetime
import json
from tqdm import tqdm
from warnings import warn
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.resnet import ResNet1d
from outlayers import DxMap, outlayer_from_str
from models.mlp import MlpClassifier
from models.prediction_model import RNNPredictionStage, LinearPredictionStage
from evaluate_12ECG_score import (compute_beta_measures, compute_auc, compute_accuracy, compute_f_measure,
                                  compute_challenge_metric)
from outlayers import collapse, get_collapse_fun


def prepare_for_evaluation(dx, out_layer, y_score, all_targets, ids, correction_factor, valid_classes, pred_stage_type):
    # Collapse entries with the same id:
    unique_ids, y_score = collapse(y_score, ids, fn=get_collapse_fun(pred_stage_type))
    # Correct for class imbalance
    y_score = y_score * correction_factor
    # Get zero one prediction
    y_pred_aux = out_layer.get_prediction(y_score)
    y_pred = dx.prepare_target(y_pred_aux, valid_classes)
    y_score = dx.prepare_probabilities(y_score, valid_classes)
    # Get metrics
    _, y_true = collapse(all_targets, ids, fn=lambda y: y[0, :], unique_ids=unique_ids)
    y_true = dx.prepare_target(y_true, valid_classes)
    return y_true, y_pred, y_score


class GetMetrics(object):

    def __init__(self, weights, normal_index=None):
        """Compute metrics"""
        self.weights = weights
        self.normal_index = normal_index

    def __call__(self, y_true, y_pred, y_score):
        """Return dictionary with relevant metrics"""
        auroc, auprc = compute_auc(y_true, y_score)
        accuracy = compute_accuracy(y_true, y_pred)
        f_measure = compute_f_measure(y_true, y_pred)
        f_beta, g_beta = compute_beta_measures(y_true, y_pred, beta=2)
        challenge_metric = compute_challenge_metric(self.weights, y_true, y_pred, self.normal_index)
        geometric_mean = np.sqrt(f_beta * g_beta)
        return {'acc': accuracy, 'f_measure': f_measure, 'f_beta': f_beta, 'g_beta': g_beta,
                'geom_mean': geometric_mean, 'auroc': auroc, 'auprc': auprc, 'challenge_metric': challenge_metric}


def get_model(config, n_classes, pretrain_stage_config=None, pretrain_stage_ckpt=None):
    N_LEADS = 12
    n_input_channels = N_LEADS if pretrain_stage_config is None else config['pretrain_output_size']
    # Remove blocks from the convolutional neural network if they are not in accordance with seq_len
    removed_blocks = 0
    for l in config['net_seq_length']:
        if l > config['seq_length']:
            del config['net_seq_length'][0]
            del config['net_filter_size'][0]
            removed_blocks += 1
    if removed_blocks > 0:
        warn("The output of the pretrain stage is not consistent with the conv net "
             "structure. We removed the first n={:d} residual blocks.".format(removed_blocks)
             + "the new configuration is " + str(list(zip(config['net_filter_size'], config['net_seq_length']))))
    # Get main model
    res_net = ResNet1d(input_dim=(n_input_channels, config['seq_length']),
                       blocks_dim=list(zip(config['net_filter_size'], config['net_seq_length'])),
                       kernel_size=config['kernel_size'], dropout_rate=config['dropout_rate'])
    # Get final prediction stage
    if config['pred_stage_type'].lower() in ['gru', 'lstm', 'rnn']:
        pred_stage = RNNPredictionStage(config, n_classes)
    else:
        n_filters_last = config['net_filter_size'][-1]
        n_samples_last = config['net_seq_length'][-1]
        pred_stage = LinearPredictionStage(model_output_dim=n_filters_last * n_samples_last, n_classes=n_classes)
    # get pretrain model if available and combine all models
    if pretrain_stage_config is None:
        # combine the models
        model = nn.Sequential(res_net, pred_stage)
    else:
        # Import pretrain only if needed
        from pretrain import MyRNN, MyTransformer, MyTransformerXL
        # load pretrained model
        if pretrain_stage_config['pretrain_model'].lower() in {'rnn', 'lstm', 'gru'}:
            pretrained = MyRNN(pretrain_stage_config)
        elif pretrain_stage_config['pretrain_model'].lower() == 'transformer':
            pretrained = MyTransformer(pretrain_stage_config)
        elif pretrain_stage_config['pretrain_model'].lower() == 'transformerxl':
            pretrained = MyTransformerXL(pretrain_stage_config)
        if pretrain_stage_ckpt is not None:
            pretrained.load_state_dict(pretrain_stage_ckpt['model'])
        ptrmdl = pretrained.get_pretrained(config['pretrain_output_size'], config['finetuning'])
        # combine the models
        if config['eval_transformer']:
            small_clf = MlpClassifier(config, n_classes, pretrain_stage_config)
            model = nn.Sequential(ptrmdl, small_clf)
        else:
            model = nn.Sequential(ptrmdl, res_net, pred_stage)
    return model


def set_output_folder(folder):
    if folder[-1] == '/':
        folder = os.path.join(folder, 'output_' +
                              str(datetime.datetime.now()).replace(":", "_").replace(" ", "_").replace(".", "_"))
    # Create output folder if needed
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    return folder


def save_config(folder, args, prefix=''):
    # Save config
    file_name = 'config.json'
    if prefix:
        file_name = prefix + '_' + file_name
    with open(os.path.join(folder, file_name), 'w') as f:
        json.dump(vars(args), f, indent='\t')


def get_data_ids(dset, valid_split, n_total, rng):
    # Get length
    n_total = len(dset) if n_total <= 0 else min(n_total, len(dset))
    n_valid = int(n_total * valid_split)
    n_train = n_total - n_valid
    assert n_train + n_valid == n_total, "data split: incorrect sizes"
    # Get ids
    all_ids = dset.get_ids()
    rng.shuffle(all_ids)
    train_ids = all_ids[:n_train]
    valid_ids = all_ids[n_train:n_train + n_valid]

    return train_ids, valid_ids


def write_data_ids(folder, train_ids, valid_ids, prefix=''):
    file_name_addon = ''
    if prefix:
        file_name_addon = prefix + '_'
    # write data
    with open(os.path.join(folder, file_name_addon+'train_ids.txt'), 'w') as f:
        f.write(','.join(train_ids))
    with open(os.path.join(folder, file_name_addon+'valid_ids.txt'), 'w') as f:
        f.write(','.join(valid_ids))


def get_output_layer(path):
    if not os.path.isfile(path):
        raise ValueError('Invalid outlayer')
    with open(path, 'r') as f:
        descriptor = f.read()
    out_layer = outlayer_from_str(descriptor.split('\n')[0])
    dx = DxMap.from_str('\n'.join(descriptor.split('\n')[1:]).strip())
    return out_layer, dx


def get_correction_factor(dset, dx, expected_class_distribution):
    dset.use_only_header(True)
    occurences = dx.prepare_target(np.vstack([dx.target_from_labels(sample['labels']) for sample in dset]))
    dset.use_only_header(False)
    n_occurences = occurences.sum(axis=0)
    fraction = n_occurences / occurences.shape[0]
    # Get occurences
    tqdm.write("\t frequencies = ocurrences / samples (for each abnormality)")
    tqdm.write("\t\t\t   = " + ', '.join(
        ["{:}:{:d}({:.3f})".format(c, n, f) for c, n, f in zip(dx.classes_at_the_output, n_occurences, fraction)]
    ))
    # Get classes of interest
    if expected_class_distribution == 'uniform':
        expected_fraction = np.array(fraction > 0, dtype=float)
    elif expected_class_distribution == 'train':
        expected_fraction = fraction
    else:
        raise ValueError('Invalid args.expected_class_distribution.')
    correction_factor = np.nan_to_num(expected_fraction / fraction)
    return correction_factor

# Load
def try_except_msg(default=None):
    def decorator(cmd):
        object_name = cmd.__name__.split('_')[-1]
        def new_cmd(*args, **kwargs):
            try:
                x = cmd(*args, **kwargs)
                if object_name:
                    tqdm.write("\tFound {:}!".format(object_name))
                return x
            except:
                if object_name:
                    tqdm.write("\tDid not found {:}!".format(object_name))
                return default
        return new_cmd
    return decorator


@try_except_msg()
def load_model(folder):
    return torch.load(os.path.join(folder, 'model.pth'), map_location=lambda storage, loc: storage)


@try_except_msg(default=(None, None))
def load_outlayer(folder):
    return get_output_layer(os.path.join(folder, 'out_layer.txt'))


@try_except_msg()
def load_configdict(folder):
    with open(os.path.join(folder, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    return config_dict


@try_except_msg()
def load_history(folder, ckpt):
    history = pd.read_csv(os.path.join(folder, 'history.csv'))
    return history[history['epoch'] < ckpt['epoch'] + 1]  # Remove epochs after the ones from the saved model


@try_except_msg(default=([], []))
def load_ids(folder):
    with open(os.path.join(folder, 'train_ids.txt'), 'r') as f:
        train_ids = f.read().split(',')
        train_ids.sort()
    with open(os.path.join(folder, 'valid_ids.txt'), 'r') as f:
        valid_ids = f.read().split(',')
        valid_ids.sort()
    return train_ids, valid_ids

@try_except_msg()
def load_correction_factor(folder):
    return np.loadtxt(os.path.join(folder, 'correction_factor.txt'))


def check_model(folder):
    tqdm.write("Looking for previous model...")
    config_dict = load_configdict(folder)
    ckpt = load_model(folder)
    out_layer, dx = load_outlayer(folder)
    correction_factor = load_correction_factor(folder)
    ids = load_ids(folder)
    history = load_history(folder, ckpt)
    tqdm.write("Done!")
    return config_dict, ckpt, dx, out_layer, correction_factor, ids, history


def check_pretrain_model(folder, do_print=True):
    try:
        ckpt_pretrain_stage = torch.load(os.path.join(folder, 'pretrain_model.pth'),
                                         map_location=lambda storage, loc: storage)
        config_pretrain_stage = os.path.join(folder, 'pretrain_config.json')
        with open(config_pretrain_stage, 'r') as f:
            config_dict_pretrain_stage = json.load(f)
        if do_print:
            tqdm.write("Found pretrained model!")
        with open(os.path.join(folder, 'pretrain_train_ids.txt'), 'r') as f:
            pretrain_train_ids = f.read().split(',')
            pretrain_train_ids.sort()
        with open(os.path.join(folder, 'pretrain_valid_ids.txt'), 'r') as f:
            pretrain_valid_ids = f.read().split(',')
            pretrain_valid_ids.sort()
    except:
        ckpt_pretrain_stage = None
        config_dict_pretrain_stage = None
        pretrain_train_ids = []
        pretrain_valid_ids = []
        if do_print:
            tqdm.write("Did not find pretrained model!")

    pretrain_ids = (pretrain_train_ids, pretrain_valid_ids)
    return config_dict_pretrain_stage, ckpt_pretrain_stage, pretrain_ids