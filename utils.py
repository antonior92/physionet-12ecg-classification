import random
import os
import datetime
import json
from tqdm import tqdm
from warnings import warn
import torch
import torch.nn as nn
import numpy as np

from data.ecg_dataloader import ECGBatchloader
from models.resnet import ResNet1d
from models.mlp import MlpClassifier
from models.prediction_model import RNNPredictionStage, LinearPredictionStage
from evaluate_12ECG_score import (compute_beta_measures, compute_auc, compute_accuracy, compute_f_measure,
                                  compute_challenge_metric)


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


def set_output_folder(args, settings, prefix=''):
    if settings.folder[-1] == '/':
        folder = os.path.join(settings.folder, 'output_' +
                              str(datetime.datetime.now()).replace(":", "_").replace(" ", "_").replace(".", "_"))
    else:
        folder = settings.folder
    # Create output folder if needed
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    # Save config
    file_name = 'config.json'
    if prefix:
        file_name = prefix + '_' + file_name
    with open(os.path.join(folder, file_name), 'w') as f:
        json.dump(vars(args), f, indent='\t')

    return folder


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


def get_data_ids(dset, args):
    rng = random.Random(args.seed)
    # Get length
    n_total = len(dset) if args.n_total <= 0 else min(args.n_total, len(dset))
    n_valid = int(n_total * args.valid_split)
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


def get_dataloaders(dset, data_ids, args, dx=None, seed=None, drop_last=False):
    data_loader = ECGBatchloader(dset, data_ids, dx, batch_size=args.batch_size,
                                 length=args.seq_length, seed=seed, drop_last=drop_last)
    n_data = len(data_ids)
    n_total = len(dset)
    tqdm.write("\t train:  {:d} ({:2.2f}\%) ECG records divided into {:d} samples of fixed length"
               .format(n_data, 100 * n_data / n_total, len(data_loader))),

    return data_loader
