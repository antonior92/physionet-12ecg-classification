import os
import datetime
import json
from tqdm import tqdm
from warnings import warn
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import copy

from models.resnet import ResNet1d
from outlayers import DxMap, outlayer_from_str
from models.mlp import MlpClassifier
from models.prediction_model import RNNPredictionStage, LinearPredictionStage
from evaluate_12ECG_score import (compute_beta_measures, compute_auc, compute_accuracy, compute_f_measure,
                                  compute_challenge_metric, prepare_classes, load_weights)


class GetMetrics(object):

    def __init__(self, path, targets, classes, normal_class=None, equivalent_classes=None):
        """Compute metrics"""
        self.path = path
        self.normal_class = normal_class
        self.equivalent_classes = equivalent_classes
        self.classes = classes
        self.targets = targets

    def __call__(self, y_pred, y_score):
        """Return dictionary with relevant metrics"""
        y_true = self.targets
        classes = copy(self.classes)
        y_true, y_pred, y_score = y_true.copy(), y_pred.copy(), y_score.copy()
        classes, y_true, y_pred, y_score = prepare_classes(classes, self.equivalent_classes,
                                                           y_true, y_pred, y_score)
        weights = load_weights(self.path, classes)
        # Only consider classes that are scored with the Challenge metric.
        indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
        classes = [x for i, x in enumerate(classes) if indices[i]]
        y_true = y_true[:, indices]
        y_pred = y_pred[:, indices]
        y_score = y_score[:, indices]
        weights = weights[np.ix_(indices, indices)]
        #  Comput metrics
        auroc, auprc = compute_auc(y_true, y_score)
        accuracy = compute_accuracy(y_true, y_pred)
        f_measure = compute_f_measure(y_true, y_pred)
        f_beta, g_beta = compute_beta_measures(y_true, y_pred, beta=2)
        challenge_metric = compute_challenge_metric(weights, y_true, y_pred, classes, self.normal_class)
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


def get_output_layer(path):
    if not os.path.isfile(path):
        raise ValueError('Invalid outlayer')
    with open(path, 'r') as f:
        descriptor = f.read()
    out_layer = outlayer_from_str(descriptor.split('\n')[0])
    dx = DxMap.from_str('\n'.join([d.split('#', 1)[0].strip() for d in descriptor.split('\n')[1:]]))
    return out_layer, dx


def get_targets(dset, classes):
    dset.use_only_header(True)
    targets = np.vstack([np.isin(classes, sample['labels']) for sample in dset])
    dset.use_only_header(False)
    return targets


def get_correction_factor(dset, dx, expected_class_distribution):
    dset.use_only_header(True)
    targets = np.vstack([dx.target_from_labels(sample['labels']) for sample in dset])
    dset.use_only_header(False)
    occurences = dx.prepare_target(targets)
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


def try_except_msg(default=None):
    def decorator(cmd):
        object_name = ''.join(cmd.__name__.split('_')[1:])
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


def fname(folder, name, prefix=''):
    return os.path.join(folder, (prefix + '_' + name) if prefix else name)


def write_data_ids(folder, train_ids, valid_ids, prefix=''):
    # write data
    with open(fname(folder, 'train_ids.txt', prefix), 'w') as f:
        f.write(','.join(train_ids))
    with open(fname(folder, 'valid_ids.txt', prefix), 'w') as f:
        f.write(','.join(valid_ids))


def save_config(folder, args, prefix=''):
    with open(fname(folder, 'config.json', prefix), 'w') as f:
        json.dump(vars(args), f, indent='\t')


def initialize_history():
    history = pd.DataFrame(columns=["epoch", "train_loss", "lr", "f_beta", "g_beta", "geom_mean"])
    return history


def update_history(history, learning_rate, train_loss, metrics, ep):
    dict_history = {"epoch": ep, "train_loss": train_loss,
                    "lr": learning_rate}
    if metrics is not None:
        dict_history.update({"f_beta": metrics['f_beta'], "g_beta": metrics['g_beta'],
                             "geom_mean": metrics['geom_mean'],
                             "challenge_metric": metrics['challenge_metric']})
    return history.append(dict_history, ignore_index=True)


def save_history(folder, history):
    history.to_csv(os.path.join(folder, 'history.csv'), index=False)


def print_message(metrics=None, ep=-1, learning_rate=None, train_loss=None):
    # Print message
    message = ''
    if ep >= 0:
        message += 'Epoch {:2d}:'.format(ep)
    else:
        message += 'Performance:'
    if learning_rate is not None:
        message += ' \tLearning Rate {:.7f}'.format(learning_rate)
    if train_loss is not None:
        message += ' \tTrain Loss {:.6f}'.format(train_loss)
    if metrics is not None:
        message += ' \tFbeta: {:.3f} \tGbeta: {:.3f} \tChallenge: {:.3f}' \
                    .format(metrics['f_beta'], metrics['g_beta'], metrics['challenge_metric'])
    tqdm.write(message)


@try_except_msg()
def load_model(folder, prefix=''):
    return torch.load(fname(folder, 'model.pth', prefix), map_location=lambda storage, loc: storage)


@try_except_msg()
def load_history(folder, ckpt, prefix=''):
    history = pd.read_csv(fname(folder, 'history.csv', prefix))
    return history[history['epoch'] < ckpt['epoch'] + 1]  # Remove epochs after the ones from the saved model


@try_except_msg(default=([], []))
def load_ids(folder, prefix=''):
    with open(fname(folder, 'train_ids.txt', prefix), 'r') as f:
        str = f.read()
        if len(str) == 0:
            raise ValueError
        train_ids = str.strip().split(',')

    with open(fname(folder, 'valid_ids.txt', prefix), 'r') as f:
        str = f.read()
        if len(str) == 0:
            raise ValueError
        valid_ids = str.strip().split(',')
    return train_ids, valid_ids


@try_except_msg()
def load_configdict(folder, prefix=''):
    with open(fname(folder, 'config.json', prefix), 'r') as f:
        config_dict = json.load(f)
    return config_dict


@try_except_msg(default=(None, None))
def load_outlayer(folder):
    return get_output_layer(fname(folder, 'out_layer.txt'))


@try_except_msg()
def load_correction_factor(folder):
    return np.loadtxt(fname(folder, 'correction_factor.txt'))


@try_except_msg(default=(None, None, None))
def load_logits(folder):
    logits_df = pd.read_csv(os.path.join(folder, 'logits.csv'))
    ids = list(logits_df['ids'])
    subids = list(logits_df['subids'])
    keys = list(logits_df.keys())
    keys.remove('ids')
    keys.remove('subids')
    logits = torch.tensor(logits_df[keys].values, dtype=torch.float32)
    return logits, ids, subids


def check_model(folder):
    tqdm.write("Looking for previous model...")
    config_dict = load_configdict(folder)
    ckpt = load_model(folder)
    out_layer, dx = load_outlayer(folder)
    correction_factor = load_correction_factor(folder)
    ids = load_ids(folder)
    history = load_history(folder, ckpt)
    logits = load_logits(folder)
    tqdm.write("Done!")
    return config_dict, ckpt, dx, out_layer, correction_factor, ids, history, logits


def check_pretrain_model(folder):
    tqdm.write("Looking for self-supervised pretrained stage...")
    config_dict = load_configdict(folder, prefix='pretrain')
    ckpt = load_model(folder, prefix='pretrain')
    ids = load_ids(folder, prefix='pretrain')
    history = load_history(folder, ckpt, prefix='pretrain')
    tqdm.write("Done!")
    return config_dict, ckpt, ids, history


def save_logits(all_logits, ids, sub_ids, folder):
    if (all_logits is not None) and (ids is not None) and (sub_ids is not None):
        index = pd.MultiIndex.from_tuples(list(zip(ids, sub_ids)), names=['ids', 'subids'])
        logits_df = pd.DataFrame(data=all_logits.numpy(), index=index)
        logits_df.to_csv(os.path.join(folder, 'logits.csv'))


def scale_by_number_of_predictions(y_score, scale):
    y_score = y_score.copy()
    mask = np.zeros_like(y_score, dtype=bool)
    for i, s in enumerate(scale[:-1]):
        # Compute mask
        mask_i = y_score >= np.sort(y_score, axis=-1)[:, -(i+1)][:, None]
        modify_mask = mask_i & (~mask)
        # Scale values
        y_score[modify_mask] = y_score[modify_mask] * s
        # update mask
        mask = mask | mask_i
    # Apply scale to remaing values
    y_score[~mask] = y_score[~mask] * scale[-1]
    return y_score