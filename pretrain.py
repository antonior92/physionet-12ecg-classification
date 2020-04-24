import os
import json
import torch
import argparse
import datetime
import pandas as pd
from warnings import warn
from ecg_dataset import *
from tqdm import tqdm
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


class PretrainedRNNBlock(nn.Module):
    """Get reusable part from MyRNN and return new model. Include Linear block with the given output_size."""
    def __init__(self, pretrained, output_size, freeze=False):
        super(PretrainedRNNBlock, self).__init__()
        self.rnn = pretrained._modules['rnn']
        if freeze:
            for param in self.rnn.parameters():
                param.requires_grad = False
        self.linear = nn.Linear(self.rnn.hidden_size, output_size)

    def forward(self, inp):
        o1, _ = self.rnn(inp.transpose(1, 2))
        o2 = self.linear(o1)
        return o2.transpose(1, 2)


class MyRNN(nn.Module):
    """My RNN"""
    def __init__(self, args):
        super(MyRNN, self).__init__()
        N_LEADS = 12
        self.rnn = getattr(nn, args['rnn_type'])(N_LEADS, args['hidden_size'], args['num_layers'],
                                              dropout=args['dropout'], batch_first=True)
        self.linear = nn.Linear(args['hidden_size'], N_LEADS * len(args['k_steps_ahead']))

    def forward(self, inp):
        o1, _ = self.rnn(inp.transpose(1, 2))
        o2 = self.linear(o1)
        return o2.transpose(1, 2)

    def get_pretrained(self, output_size, freeze=False):
        return PretrainedRNNBlock(self, output_size, freeze)


def get_input_and_targets(traces, args):
    max_steps_ahead = max(args.k_steps_ahead)
    inp = traces[:, :, :-max_steps_ahead]  # Try to predict k steps ahead
    n = inp.size(2)
    target = torch.cat([traces[:, :, k:k+n] for k in args.k_steps_ahead], dim=1)
    return inp, target


def selfsupervised(ep, model, optimizer, samples, loss, device, args, train):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    n_entries = 0
    n_total = len(samples)
    n_batches = int(np.ceil(n_total/args.batch_size))
    str_name = 'train' if train else 'val'
    desc = "Epoch {:2d}: {} - Loss: {:.6f}"
    bar = tqdm(initial=0, leave=True, total=n_total,
               desc=desc.format(ep, str_name, 0), position=0)
    for i in range(n_batches):
        # Send to device
        bs = min(args.batch_size, n_total - n_entries)
        traces = torch.stack([torch.tensor(s['data'], dtype=torch.float32, device=device)
                              for s in samples[n_entries:n_entries + bs]], dim=0)
        inp, target = get_input_and_targets(traces, args)
        if train:
            # Reinitialize grad
            model.zero_grad()
            # Forward pass
            output = model(inp)
            ll = loss(output, target)
            # Backward pass
            ll.backward()
            clip_grad_norm_(model.parameters(), args.clip_value)
            # Optimize
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(inp)
                ll = loss(output, target)
        # Update
        total_loss += ll.detach().cpu().numpy()
        n_entries += bs
        # Update train bar
        bar.desc = desc.format(ep, str_name, total_loss / n_entries)
        bar.update(bs)
    bar.close()
    return total_loss / n_entries


if __name__ == '__main__':
    # Experiment parameters
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--seed', type=int, default=2,
                               help='random seed for number generator (default: 2)')
    config_parser.add_argument('--epochs', type=int, default=125,
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
                               default=[40, 75, 100],
                               help='milestones for lr scheduler (default: [100, 200])')
    config_parser.add_argument("--lr_factor", type=float, default=0.1,
                               help='reducing factor for the lr in a plateeu (default: 0.1)')
    config_parser.add_argument('--dropout', type=float, default=0,
                               help='dropout rate (default: 0).')
    config_parser.add_argument('--rnn_type', type=str, default='LSTM',
                               help="Which rnn to use. Options are {'LSTM', 'GRU', 'RNN'}")
    config_parser.add_argument('--hidden_size', type=int, default=800,
                               help="Hidden size rnn. Default is 800.")
    config_parser.add_argument('--num_layers', type=int, default=1,
                               help="Number of layers. Default is 1.")
    config_parser.add_argument('--clip_value', type=float, default=1.0,
                               help='maximum value for the gradient norm (default: 1.0)')
    config_parser.add_argument('--k_steps_ahead', nargs='+', type=int, default=[10, 20, 25, 50, 75, 90, 100],
                               help='Try to predict k steps ahead')
    config_parser.add_argument('--n_total', type=int, default=-1,
                               help='number of samples to be used during training. By default use '
                                    'all the samples available. Useful for quick tests.')

    args, rem_args = config_parser.parse_known_args()
    # System setting
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument('--input_folder', type=str, default='./Training_WFDB',
                            help='input folder.')
    sys_parser.add_argument('--cuda', action='store_true',
                            help='use cuda for computations. (default: False)')
    sys_parser.add_argument('--folder', default=os.getcwd() + '/',
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
        folder = os.path.join(settings.folder, 'output_' +
                              str(datetime.datetime.now()).replace(":", "_").replace(" ", "_").replace(".", "_"))
        try:
            os.makedirs(folder)
        except FileExistsError:
            pass
    else:
        folder = settings.folder
    with open(os.path.join(folder, 'pretrain_config.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')
    # Set seed
    torch.manual_seed(args.seed)

    tqdm.write("Define dataset...")
    dset = ECGDataset(settings.input_folder, freq=args.sample_freq)
    # Get length
    n_total = len(dset) if args.n_total <= 0 else min(args.n_total, len(dset))
    n_valid = int(n_total * args.valid_split)
    n_train = n_total - n_valid
    # Define dataset
    train_dset = dset[:n_train]
    train_samples = list(itertools.chain(*[split_long_signals(s) for s in train_dset]))
    valid_dset = dset[n_train:n_total]
    valid_samples = list(itertools.chain(*[split_long_signals(s) for s in valid_dset]))
    # Save train and test ids
    np.savetxt(os.path.join(folder, 'pretrain_train_ids.txt'), [s['id'] for s in train_dset], fmt='%d')
    np.savetxt(os.path.join(folder, 'pretrain_valid_ids.txt'), [s['id'] for s in valid_dset], fmt='%d')
    # Get number of batches
    tqdm.write("Done!")

    tqdm.write("Define model...")
    model = MyRNN(vars(args))
    model.to(device=device)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_factor)
    tqdm.write("Done!")

    tqdm.write("Define loss...")
    loss = nn.MSELoss(reduction='sum')
    tqdm.write("Done!")

    history = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss", "lr",])
    best_loss = np.Inf
    for ep in range(args.epochs):
        train_loss = selfsupervised(ep, model, optimizer, train_samples, loss, device, args, train=True)
        valid_loss = selfsupervised(ep, model, optimizer, valid_samples, loss, device, args, train=False)
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Print message
        message = 'Epoch {:2d}: \tTrain Loss {:.6f} ' \
                  '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t' \
            .format(ep, train_loss, valid_loss, learning_rate)
        tqdm.write(message)

        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss, "valid_loss": valid_loss,
                                  "lr": learning_rate},
                                 ignore_index=True)
        history.to_csv(os.path.join(folder, 'pretrain_history.csv'), index=False)

        # Save best model
        if best_loss > valid_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'pretrain_model.pth'))
            # Update best validation loss
            best_loss = valid_loss
            tqdm.write("Save model!")
        # Save last model
        if ep == args.epochs - 1:
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'pretrain_final_model.pth'))
            tqdm.write("Save model!")
        # Call optimizer step
        scheduler.step()