import os
import json
import torch.nn as nn
import argparse
import random
import pandas as pd
import sys
# sys.path.insert(1, os.getcwd())
from data import *
from warnings import warn
from tqdm import tqdm
from metrics import get_metrics
from output_layer import OutputLayer, collapse

os.chdir('../')
sys.path.append(os.getcwd())
from train import train, evaluate
# os.chdir(sys.path[1])

from data.ecg_dataloader import ECGBatchloader
from models.resnet import ResNet1d
from models.prediction_model import RNNPredictionStage, LinearPredictionStage
from output_layer import OutputLayer, collapse, DxClasses
from evaluate_12ECG_score import (compute_beta_measures, compute_auc, compute_accuracy, compute_f_measure,
                                  compute_challenge_metric, load_weights)


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


# get the model
def get_model(config, n_classes, pretrain_stage_config, pretrain_stage_ckpt):
    # moved get simple classifier from here

    # get pretraining model
    if pretrain_stage_config['pretrain_model'].lower() in {'rnn', 'lstm', 'gru'}:
        pretrained = MyRNN(pretrain_stage_config)
    elif pretrain_stage_config['pretrain_model'].lower() == 'transformer':
        pretrained = MyTransformer(pretrain_stage_config)
    elif pretrain_stage_config['pretrain_model'].lower() == 'transformerxl':
        pretrained = MyTransformerXL(pretrain_stage_config)
    if pretrain_stage_ckpt is not None:
        pretrained.load_state_dict(pretrain_stage_ckpt['model'])
    ptrmdl = pretrained.get_pretrained(config['pretrain_output_size'], config['finetuning'])

    # get simple classifier
    clf = EvalClassifier(config, n_classes, pretrain_stage_config)

    # combine model
    model = nn.Sequential(ptrmdl, clf)

    # return full model
    return model


# define small classifier
class EvalClassifier(nn.Module):
    """ Simple classifier """

    def __init__(self, args, n_classes, pretrain_stage_config):
        super(EvalClassifier, self).__init__()
        self.input_size = int(args['pretrain_output_size'] * args['seq_length'] / pretrain_stage_config['steps_concat'])
        self.hidden_dim1 = 1024
        self.hidden_dim2 = 512
        self.hidden_dim3 = 256

        self.fc1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_dim1)
        self.fc2 = nn.Linear(in_features=self.hidden_dim1, out_features=self.hidden_dim2)
        self.fc3 = nn.Linear(in_features=self.hidden_dim2, out_features=self.hidden_dim3)
        self.fc4 = nn.Linear(in_features=self.hidden_dim3, out_features=n_classes)

    def forward(self, src):
        batch_size = src.size(0)

        src1 = src.reshape(batch_size, -1)
        src2 = nn.functional.relu(self.fc1(src1))
        src3 = nn.functional.relu(self.fc2(src2))
        src4 = nn.functional.relu(self.fc3(src3))
        out = self.fc4(src4)

        return out


if __name__ == '__main__':
    # Experiment parameters
    config_parser = argparse.ArgumentParser(add_help=False)
    # Learning parameters
    config_parser.add_argument('--seed', type=int, default=2,
                               help='random seed for number generator (default: 2)')
    config_parser.add_argument('--epochs', type=int, default=200,
                               help='maximum number of epochs (default: 70)')
    config_parser.add_argument('--sample_freq', type=int, default=400,
                               help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    config_parser.add_argument('--seq_length', type=int, default=4096,
                               help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                    'to fit into the given size. (default: 4096)')
    config_parser.add_argument('--batch_size', type=int, default=32,
                               help='batch size (default: 32).')
    config_parser.add_argument('--valid_split', type=float, default=0.30,
                               help='fraction of the data used for validation (default: 0.1).')
    config_parser.add_argument('--lr', type=float, default=0.001,
                               help='learning rate (default: 0.001)')
    config_parser.add_argument('--milestones', nargs='+', type=int,
                               default=[75, 125, 175],
                               help='milestones for lr scheduler (default: [100, 200])')
    config_parser.add_argument("--lr_factor", type=float, default=0.1,
                               help='reducing factor for the lr in a plateeu (default: 0.1)')
    # Pretrain Model parameters
    config_parser.add_argument('--pretrain_output_size', type=int, default=64,
                               help='The output of the pretrained model goes through a linear layer, which outputs'
                                    'a tensor with the given number of features (default: 64).')
    config_parser.add_argument('--finetuning', action='store_true', default=True,
                               help='when there is a pre-trained model, by default it '
                                    'freezes the weights of the pre-trained model, but with this option'
                                    'these weight will be fine-tunned during training.')
    args, rem_args = config_parser.parse_known_args()
    # System setting
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument('--input_folder', type=str, default='./Training_WFDB',
                            help='input folder.')
    sys_parser.add_argument('--cuda', action='store_true',
                            help='use cuda for computations. (default: False)')
    sys_parser.add_argument('--folder', default=os.getcwd() + '/evaluation/exp1_pretrain_transformer',
                            help='output folder. If we pass /PATH/TO/FOLDER/ ending with `/`,'
                                 'it creates a folder `output_YYYY-MM-DD_HH_MM_SS_MMMMMM` inside it'
                                 'and save the content inside it. If it does not ends with `/`, the content is saved'
                                 'in the folder provided.')
    settings, unk = sys_parser.parse_known_args(rem_args)
    #  Final parser is needed for generating help documentation
    parser = argparse.ArgumentParser(parents=[sys_parser, config_parser])
    _, unk = parser.parse_known_args(unk)
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    # Set device
    device = torch.device('cuda:0' if settings.cuda else 'cpu')
    # Generate output folder if needed and save config file
    if settings.folder[-1] == '/':
        warn("Need to specify folder with pretrained model!")
        sys.exit()
    else:
        folder = settings.folder
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    with open(os.path.join(folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')
    # Set seed
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    # Check if there is pretrained model in the given folder
    try:
        ckpt_pretrain_stage = torch.load(os.path.join(folder, 'pretrain_model.pth'),
                                         map_location=lambda storage, loc: storage)
        config_pretrain_stage = os.path.join(folder, 'pretrain_config.json')
        with open(config_pretrain_stage, 'r') as f:
            config_dict_pretrain_stage = json.load(f)
        tqdm.write("Found pretrained model!")
        with open(os.path.join(folder, 'pretrain_train_ids.txt'), 'r') as f:
            pretrain_ids = f.read().split(',')
        # Import pretrain only if needed
        from pretrain import MyRNN, MyTransformer, MyTransformerXL
    except:
        # break file execution if no pretrain file is found in the given folder!
        warn("No pretrain model found!")
        sys.exit()

    tqdm.write("Define dataset...")
    dset = ECGDataset(settings.input_folder, freq=args.sample_freq)
    tqdm.write("Done!")

    tqdm.write("Define train and validation splits...")
    all_ids = dset.get_ids()
    set_all_ids = set(all_ids)
    # Get pretrained ids
    pretrain_ids = set_all_ids.intersection(pretrain_ids)  # Get only pretrain ids available
    other_ids = list(set_all_ids.difference(pretrain_ids))
    n_pretrain_ids = len(pretrain_ids)
    # Get length
    n_total = len(dset) if args.n_total <= 0 else min(args.n_total, len(dset))
    n_valid = int(n_total * args.valid_split)
    n_train = n_total - n_valid
    if n_pretrain_ids > n_train:
        tqdm.write("\t Training size extendeded to include all pretraining ids!")
        n_train = n_pretrain_ids
        n_valid = n_total - n_train
    # Get train and valid ids
    rng.shuffle(other_ids)
    train_ids = other_ids[:n_train - n_pretrain_ids] + list(pretrain_ids)
    valid_ids = other_ids[n_train - n_pretrain_ids:n_total - n_pretrain_ids]
    # Save train and test ids
    with open(os.path.join(folder, 'train_ids.txt'), 'w') as f:
        f.write(','.join(train_ids))
    with open(os.path.join(folder, 'valid_ids.txt'), 'w') as f:
        f.write(','.join(valid_ids))
    tqdm.write("Done!")

    tqdm.write("Define output layer...")
    # Get all classes in the dataset
    dset_classes = dset.get_classes()
    # Get all classes to be scored
    df = pd.read_csv(os.path.join(settings.dx, 'dx_mapping_scored.csv'))
    scored_classes = [str(c) for c in list(df['SNOMED CT Code'])]
    # Get training classes
    train_classes = dset_classes if args.train_classes == 'dset' else scored_classes
    valid_classes = dset_classes if args.valid_classes == 'dset' else scored_classes
    # Get mutually exclusive entries
    if args.outlayer == 'sigmoid-and-softmax':
        with open(os.path.join(settings.dx, 'mutually_exclusivity.txt'), 'r') as file:
            mutually_exclusive = [line.split(',') for line in file.read().split('\n')]
        with open(os.path.join(settings.dx, 'null_class.txt'), 'r') as file:
            null_class = file.read()
        dx = DxClasses(train_classes, mutually_exclusive, null_class)
    if args.outlayer == 'softmax':
        with open(os.path.join(settings.dx, 'null_class.txt'), 'r') as file:
            null_class = file.read()
        mutually_exclusive = [list(set(train_classes).difference([null_class]))]
        dx = DxClasses(train_classes, mutually_exclusive, null_class)
    else:
        dx = DxClasses(train_classes)
    out_layer = OutputLayer(args.batch_size, dx, device)
    tqdm.write("Done!")

    tqdm.write("Define metrics...")
    weights = load_weights(os.path.join(settings.dx, 'weights.csv'), valid_classes)
    NORMAL = '426783006'
    get_metrics = GetMetrics(weights, valid_classes.index(NORMAL))
    tqdm.write("Done!")

    tqdm.write("Get dataloaders...")
    train_loader = ECGBatchloader(dset, train_ids, dx, batch_size=args.batch_size, length=args.seq_length)
    valid_loader = ECGBatchloader(dset, valid_ids, dx, batch_size=args.batch_size, length=args.seq_length)
    tqdm.write("\t train:  {:d} ({:2.2f}\%) ECG records divided into {:d} samples of fixed length"
               .format(n_train, 100 * n_train / n_total, len(train_loader))),
    tqdm.write("\t valid:  {:d} ({:2.2f}\%) ECG records divided into {:d} samples of fixed length"
               .format(n_valid, 100 * n_valid / n_total, len(valid_loader)))
    tqdm.write("Done!")

    tqdm.write("Define threshold ...")
    threshold = dx.compute_threshold(dset, train_ids)
    tqdm.write("\t threshold = train_ocurrences / train_samples (for each abnormality)")
    tqdm.write("\t\t\t   = " + ', '.join(["{:}:{:.3f}".format(c, threshold[i]) for i, c in enumerate(dx.code)]))
    tqdm.write("Done!")

    tqdm.write("Define model...")
    model = get_model(vars(args), len(dx), config_dict_pretrain_stage, ckpt_pretrain_stage)
    model.to(device=device)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_factor)
    tqdm.write("Done!")

    tqdm.write("Define loss...")
    out_layer = OutputLayer(args.batch_size, mututally_exclusive, device)
    tqdm.write("Done!")

    history = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss", "lr", "f_beta", "g_beta", "geom_mean"])
    best_geom_mean = -np.Inf
    # run over all epochs
    for ep in range(args.epochs):
        # Train and evaluate
        train_loss = train(ep, model, optimizer, train_loader, out_layer, device)
        valid_loss, y_score, all_targets, ids = evaluate(ep, model, valid_loader, out_layer, device)
        # Collapse entries with the same id:
        unique_ids, y_score = collapse(y_score, ids, fn=lambda y: np.mean(y, axis=0))
        # Get zero one prediction
        y_pred_aux = dx.apply_threshold(y_score, threshold)
        y_pred = dx.target_to_binaryclass(y_pred_aux)
        y_pred = dx.reorganize(y_pred, valid_classes, prob=False)
        y_score = dx.reorganize(y_score, valid_classes, prob=True)
        # Get metrics
        y_true = dx.target_to_binaryclass(all_targets)
        _, y_true = collapse(y_true, ids, fn=lambda y: y[0, :], unique_ids=unique_ids)
        y_true = dx.reorganize(y_true, valid_classes)
        metrics = get_metrics(y_true, y_pred, y_score)
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Print message
        message = 'Epoch {:2d}: \tTrain Loss {:.6f} ' \
                  '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t' \
                  'Fbeta: {:.3f} \tGbeta: {:.3f} \tChallenge: {:.3f}' \
            .format(ep, train_loss, valid_loss, learning_rate,
                    metrics['f_beta'], metrics['g_beta'],
                    metrics['challenge_metric'])
        tqdm.write(message)
        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss, "valid_loss": valid_loss,
                                  "lr": learning_rate, "f_beta": metrics['f_beta'],
                                  "g_beta": metrics['g_beta'], "geom_mean": metrics['geom_mean'],
                                  'challenge_metric': metrics['challenge_metric']},
                                 ignore_index=True)
        history.to_csv(os.path.join(folder, 'history.csv'), index=False)

        # Save best model
        if best_geom_mean < metrics['geom_mean']:
            # Save model
            torch.save({'epoch': ep,
                        'threshold': threshold,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_geom_mean = metrics['geom_mean']
            tqdm.write("Save model!")
        # Call optimizer step
        scheduler.step()
        # Save last model
        if ep == args.epochs - 1:
            torch.save({'threshold': threshold,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'final_model.pth'))
            tqdm.write("Save model!")