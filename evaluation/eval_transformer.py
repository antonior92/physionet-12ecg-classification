import os
import json
import torch
import torch.nn as nn
import argparse
import random
import pandas as pd
import numpy as np
import sys
sys.path.insert(1,os.getcwd())
from data import *
from warnings import warn
from tqdm import tqdm
from metrics import get_metrics
from output_layer import OutputLayer, collapse

os.chdir('../')
sys.path.append(os.getcwd())
from train import train, evaluate
os.chdir(sys.path[1])


# get the model
def get_model(config, pretrain_stage_config, pretrain_stage_ckpt):
   #moved get simple classifier from here

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
    clf = EvalClassifier(config, len(CLASSES))

    # combine model
    model = nn.Sequential(ptrmdl, clf)

    # return full model
    return model


# define small classifier
class EvalClassifier(nn.Module):
    """
    Simple classifier: This is where the FF Network has to be defined
    """

    def __init__(self, args, n_classes):
        super(EvalClassifier, self).__init__()
        """TODO: Define here the network layers and parameters"""
        #self.change_shape = np.reshape((args['batch_size'],-1))
        self.fc1 = nn.Linear(in_features = int(args['pretrain_output_size']*args['seq_length']/config_dict_pretrain_stage['steps_concat']),
             out_features = 128)
        self.fc2 = nn.Linear(in_features = 128,out_features = 32)
        self.fc3 = nn.Linear(in_features = 32, out_features = n_classes)



    def forward(self, src):
        """
        Define here the network structure.
        src is the output of the transformer and has to be fed into the FF classifier to generate the output out.

        src is a tensor of size (batch_size, pretrain_output_size, seq_length/steps_concat)
        all these parameters are defined in the pretrained transformer.

        out has to be of size (batch_size, len(CLASSES))
        so you need to flatten the incoming tensor and then push it through the FF netowrk. Flattening can be done
        similar to the last stage in the ResNet (have a look there with the view method).
        """
        
        src = src.reshape(config_dict_pretrain_stage['batch_size'],-1)
        src = nn.functional.relu(self.fc1(src))
        src = nn.functional.relu(self.fc2(src))
        src = self.fc3(src)
        
        # has to be changed
        out = src

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
    config_parser.add_argument('--pretrain_model', type=str, default='Transformer',
                               help='type of pretraining net: LSTM, GRU, RNN, Transformer (default)')
    config_parser.add_argument('--pretrain_output_size', type=int, default=64,
                               help='The output of the pretrained model goes through a linear layer, which outputs'
                                    'a tensor with the given number of features (default: 64).')
    # classifier parameters
    config_parser.add_argument('--hidden_size', type=int, nargs='+', default=400,
                               help='dimension of hidden layer in FF classifier (default: 400).')
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
    all_ids = dset.get_ids()
    set_all_ids = set(all_ids)
    # Get pretrained ids
    pretrain_ids = set_all_ids.intersection(pretrain_ids)  # Get only pretrain ids available
    other_ids = list(set_all_ids.difference(pretrain_ids))
    n_pretrain_ids = len(pretrain_ids)
    # Get length
    n_total = len(dset)
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
    # Define dataset
    train_loader = get_batchloader(dset, train_ids, batch_size=args.batch_size, length=args.seq_length)
    valid_loader = get_batchloader(dset, valid_ids, batch_size=args.batch_size, length=args.seq_length)
    # Get number of batches
    n_train_batches = len(train_loader)
    n_valid_batches = len(valid_loader)
    tqdm.write("\t train:  {:d} ({:2.2f}\%) samples divided into {:d} batches"
               .format(n_train, 100 * n_train / n_total, n_train_batches)),
    tqdm.write("\t valid:  {:d} ({:2.2f}\%) samples divided into {:d} batches"
               .format(n_valid, 100 * n_valid / n_total, n_valid_batches))
    tqdm.write("Done!")

    tqdm.write("Define threshold ...")
    # Get all targets
    targets = np.stack([s['output'] for s in dset.use_only_header(True)[train_ids]])
    targets_bin = multiclass_to_binaryclass(targets)
    threshold = targets_bin.sum(axis=0) / targets_bin.shape[0]
    tqdm.write("\t threshold = train_ocurrences / train_samples (for each abnormality)")
    tqdm.write("\t\t\t   = AF:{:.2f},I-AVB:{:.2f},RBBB:{:.2f},LBBB:{:.2f},PAC:{:.2f},PVC:{:.2f},STD:{:.2f},STE:{:.2f}"
               .format(*threshold))
    tqdm.write("Done!")

    tqdm.write("Define model...")
    model = get_model(vars(args), config_dict_pretrain_stage, ckpt_pretrain_stage)
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
    for ep in range(args.epochs):
        # Train and evaluate
        train_loss = train(ep, model, optimizer, train_loader, out_layer, device)
        valid_loss, y_score, all_targets, ids = evaluate(ep, model, valid_loader, out_layer, device)
        y_true = multiclass_to_binaryclass(all_targets)
        # Collapse entries with the same id:
        unique_ids, y_score = collapse(y_score, ids, fn=lambda y: np.mean(y, axis=0))
        _, y_true = collapse(y_true, ids, fn=lambda y: y[0, :], unique_ids=unique_ids)
        # Get labels
        y_pred = y_score > threshold
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Print message
        metrics = get_metrics(y_true, y_pred)
        message = 'Epoch {:2d}: \tTrain Loss {:.6f} ' \
                  '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t' \
                  'Fbeta: {:.3f} \tGbeta: {:.3f} \tGeom Mean: {:.3f}' \
            .format(ep, train_loss, valid_loss, learning_rate,
                    metrics['f_beta'], metrics['g_beta'],
                    metrics['geom_mean'])
        tqdm.write(message)
        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss, "valid_loss": valid_loss,
                                  "lr": learning_rate, "f_beta": metrics['f_beta'],
                                  "g_beta": metrics['g_beta'], "geom_mean": metrics['geom_mean']},
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

