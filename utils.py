import os
import datetime
import json
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import copy

from models import get_model, get_prediction_stage
from models_pretrain import get_pretrain
from outlayers import DxMap, outlayer_from_str
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


def get_data_ids(dset, valid_split, n_total, rng, previous_train_ids):
    # Get all ids
    all_ids = dset.get_ids()
    set_all_ids = set(all_ids)
    previous_train_ids_in_dset = list(set_all_ids.intersection(previous_train_ids))
    other_ids_in_dset = list(set_all_ids.difference(previous_train_ids))
    n_previous_ids = len(previous_train_ids_in_dset)
    # Get length
    n_total = len(dset) if n_total <= 0 else min(n_total, len(dset))
    n_valid = int(n_total * valid_split)
    n_train = n_total - n_valid
    assert n_train + n_valid == n_total, "data split: incorrect sizes"
    # Define train and valid ids
    other_ids_in_dset.sort()  # to be deterministic
    rng.shuffle(other_ids_in_dset)
    valid_ids = other_ids_in_dset[:n_valid]
    train_ids = other_ids_in_dset[n_valid:] + previous_train_ids_in_dset
    train_ids = train_ids[:n_train]
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


def save_ckpt(folder, ep, model, optimizer, scheduler, save_ptrmdl, save_core_model):
    ptrmdl, core_model, pred_stage = split_full_model(model)
    dict_base = {'epoch': ep,
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict()}
    if save_ptrmdl and ptrmdl is not None:
        dict_base['pretrain_model'] = ptrmdl.state_dict()
    if save_core_model and core_model is not None:
        dict_base['model'] = core_model.state_dict()
    if pred_stage is not None:
        dict_base['pred_stage'] = pred_stage.state_dict()
    torch.save(dict_base, os.path.join(folder, 'model.pth'))


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


def get_input_size(ptrmdl):
    N_LEADS = 12
    return N_LEADS if ptrmdl is None else args.pretrain_output_size


# TODO: finish testing get_full_model
def get_full_model(ptrmdl, core_model, pred_stage, pretrain_output_size, freeze_ptrmdl=False, freeze_core_model=False):
    # Get pretrained model if available
    list_module_name = []
    if ptrmdl is not None:
        ptrmdl = ptrmdl.get_pretrained(pretrain_output_size, freeze_ptrmdl)
        list_module_name = [('ptrmdl', ptrmdl)]
    if freeze_core_model:
        for param in core_model.parameters():
            param.require_grad = False
    list_module_name += [('core_model', core_model)]
    list_module_name += [('pred_stage', pred_stage)]

    return nn.Sequential(OrderedDict(list_module_name))


def split_full_model(model):
    if 'ptrmdl' in model._modules.keys():
        return model.ptrmdl, model.core_model, model.pred_stage
    else:
        return None, model.core_model, model.pred_stage

@try_except_msg()
def load_ckpt(folder, prefix=''):
    return torch.load(fname(folder, 'model.pth', prefix), map_location=lambda storage, loc: storage)


@try_except_msg()
def load_history(folder, ckpt, prefix=''):
    history = pd.read_csv(fname(folder, 'history.csv', prefix))
    return history[history['epoch'] < ckpt['epoch'] + 1]  # Remove epochs after the ones from the saved model


@try_except_msg()
def load_pretrain_model(config, ckpt):
    # Load pretrained
    pretrained = get_pretrain(config)
    # Import pretrain only if needed
    if ckpt is not None:
        pretrained.load_state_dict(ckpt['model'])
    return pretrained


def update_pretrain_models(ptrmdl, ckpt):
    # import model if possible
    if (ckpt is not None) and ('pretrain_model' in ckpt.keys()) and (ptrmdl is not None):
        ptrmdl.load_state_dict(ckpt['pretrain_model'])
    return ptrmdl


@try_except_msg()
def load_model(ptrmdl, config, ckpt):
    # Get model
    model = get_model(config['mdl_type'], get_input_size(ptrmdl), config['seq_length'], **config)
    # import model if possible
    model.load_state_dict(ckpt['model'])
    return model


@try_except_msg()
def load_pred_stage(config, ckpt, dx):
    # Get prediction stage
    pred_stage = get_prediction_stage(config['pred_stage_type'], len(dx), config['net_seq_length'][-1],
                                      config['net_filter_size'][-1], **config)
    # import model if possible
    pred_stage.load_state_dict(ckpt['pred_stage'])
    return pred_stage


@try_except_msg(default=[])
def load_train_ids(folder, prefix=''):
    with open(fname(folder, 'train_ids.txt', prefix), 'r') as f:
        str = f.read()
        if len(str) == 0:
            raise ValueError
        train_ids = str.strip().split(',')
    return train_ids


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


def check_folder(folder):
    config_dict_pretrain_stage, ckpt_pretrain_stage, ptrmdl, pretrain_ids, _ = check_pretrain_folder(folder)
    tqdm.write("Looking for previous model...")
    # Load config
    config_dict = load_configdict(folder)
    ckpt = load_ckpt(folder)
    out_layer, dx = load_outlayer(folder)
    ptrmdl = update_pretrain_models(ptrmdl, ckpt)
    core_model = load_model(ptrmdl, config_dict, ckpt)
    pred_stage = load_pred_stage(config_dict, ckpt, dx)
    correction_factor = load_correction_factor(folder)
    train_ids = load_train_ids(folder)
    history = load_history(folder, ckpt)
    logits = load_logits(folder)
    tqdm.write("Done!")
    mdl = (ptrmdl, core_model, pred_stage)
    return config_dict, ckpt, dx,  out_layer, mdl, correction_factor, set(train_ids).union(pretrain_ids), \
           history, logits


def check_pretrain_folder(folder):
    tqdm.write("Looking for self-supervised pretrained stage...")
    config_dict = load_configdict(folder, prefix='pretrain')
    ckpt = load_ckpt(folder, prefix='pretrain')
    model = load_pretrain_model(config_dict, ckpt)
    train_ids = load_train_ids(folder, prefix='pretrain')
    history = load_history(folder, ckpt, prefix='pretrain')
    tqdm.write("Done!")
    return config_dict, ckpt, model, train_ids, history


def save_logits(all_logits, ids, sub_ids, folder):
    if (all_logits is not None) and (ids is not None) and (sub_ids is not None):
        index = pd.MultiIndex.from_tuples(list(zip(ids, sub_ids)), names=['ids', 'subids'])
        logits_df = pd.DataFrame(data=all_logits.numpy(), index=index)
        logits_df.to_csv(os.path.join(folder, 'logits.csv'))