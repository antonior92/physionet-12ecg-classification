import os
import json
import torch
import argparse
import datetime
import pandas as pd
import random
from warnings import warn
from ecg_dataset import *
from tqdm import tqdm
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils import clip_grad_norm_
import random
import math


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
        self.rnn = getattr(nn, args['pretrain_model'].upper())(N_LEADS, args['hidden_size_rnn'], args['num_layers'],
                                                               dropout=args['dropout'], batch_first=True)
        self.linear = nn.Linear(args['hidden_size_rnn'], N_LEADS * len(args['k_steps_ahead']))

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
    target = torch.cat([traces[:, :, k:k + n] for k in args.k_steps_ahead], dim=1)
    return inp, target


class PretrainedTransformerBlock(nn.Module):
    """Get reusable part from MyTransformer and return new model. Include Linear block with the given output_size."""

    def __init__(self, pretrained, output_size, freeze=False):
        super(PretrainedTransformerBlock, self).__init__()
        self.N_LEADS = 12
        self.encoder = pretrained._modules['encoder']
        self.pos_encoder = pretrained._modules['pos_encoder']
        self.transformer_encoder = pretrained._modules['transformer_encoder']

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.pos_encoder.parameters():
                param.requires_grad = False
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False

        # self.encoder.out_features is also the output feature size of the transformer
        self.decoder = nn.Linear(self.encoder.out_features, output_size)

    def forward(self, inp):
        # inp size is [batch size x N_LEADS x sequence length]
        # transform src to be of size [sequence length x batch size x N_LEADS]
        src = inp.permute(2, 0, 1)

        # process data (no mask in transformer used)
        src = self.encoder(src) * math.sqrt(self.N_LEADS)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        # permute to have the same dimensions as in the input
        return output.permute(1, 2, 0)


class PositionalEncoding(nn.Module):
    # This is the positional encoding according to paper "Attention is all you need".
    # Could be changed to learnt encoding
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=10000).
    """

    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# new pre-trained model
class MyTransformer(nn.Module):
    """My Transformer:
    inspired by https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, args):
        super(MyTransformer, self).__init__()
        self.N_LEADS = 12
        self.mask_param = [args['num_masked_subseq'], args['num_masked_samples']]

        self.encoder = nn.Linear(self.N_LEADS, args['emb_size'])
        self.pos_encoder = PositionalEncoding(args['emb_size'], args['dropout'])
        encoder_layers = TransformerEncoderLayer(args['emb_size'], args['num_heads'], args['hidden_size_trans'],
                                                 args['dropout'])
        self.transformer_encoder = TransformerEncoder(encoder_layers, args['num_trans_layers'])
        self.decoder = nn.Linear(args['emb_size'], self.N_LEADS)

        self.init_weights()

    def _generate_triangular_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_random_sequence_mask(self, sz, param):
        """
        Implementation is quite inefficient so far. Should be improved!
        Also the implementation does not care about overlapping intervals of masks.
        This may yield that different number of samples are masked in different sequences.

        According to attention definition the same mask is used for all sequences in the batch.
        Mask is a [sz x sz] matrix. If the value [i,j] is masked by a value of -inf, then the for the
        computation of output j the input i is masked, meaning that no attention is used for this input.

        sz - sequence size
        p - number of non-overlapping masked subsequences
        m - number of consecutive samples for each p masked subsequences
        """
        p = param[0]
        m = param[1]

        # allocation
        idx = torch.empty((sz, p * m), dtype=torch.int64)

        # for all rows in the indexing
        for i in range(sz):
            # sample p values without replacement
            a = random.sample(range(sz - m + 1), p)
            a.sort()
            idx_row = []
            for k in range(p):
                # generate indices for row i which should be masked
                idx_row.extend(range(a[k], a[k] + m))
            idx[i, :] = torch.tensor(idx_row)

        # mask the indices with infinity
        mask = torch.zeros(sz, sz)
        mask.scatter_(1, idx, float('-inf'))
        return mask

    def init_weights(self):
        # Comment: This is copied from the source. Not sure if necessary.
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src size is [batch size x N_LEADS x sequence length]
        # transform src to be of size [sequence length x batch size x N_LEADS]
        src = src.permute(2, 0, 1)
        # generate random mask
        mask = self._generate_random_sequence_mask(len(src), self.mask_param).to(device)
        # generate triangular mask ('predict next sample')
        # mask = self._generate_triangular_mask(len(src))
        self.mask = mask

        # process data
        src = self.encoder(src) * math.sqrt(self.N_LEADS)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.mask)
        output = self.decoder(output)
        # permute to have the same dimensions as in the input
        return output.permute(1, 2, 0)

    def get_pretrained(self, output_size, freeze=False):
        return PretrainedTransformerBlock(self, output_size, freeze)


def selfsupervised(ep, model, optimizer, samples, loss, device, args, train):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    n_entries = 0
    n_total = len(samples)
    n_batches = int(np.ceil(n_total / args.batch_size))
    str_name = 'train' if train else 'val'
    desc = "Epoch {:2d}: {} - Loss: {:.6f}"
    bar = tqdm(initial=0, leave=True, total=n_total, desc=desc.format(ep, str_name, 0), position=0)
    for i in range(n_batches):
        # Send to device
        bs = min(args.batch_size, n_total - n_entries)
        traces = torch.stack([torch.tensor(s['data'], dtype=torch.float32, device=device)
                              for s in samples[n_entries:n_entries + bs]], dim=0)
        # reduces to make it run on my laptop for debugging
        # traces = traces[:, :, :500]
        # create model input and targets
        if args.pretrain_model.lower() == 'transformer':
            inp = traces
            target = traces
        elif args.pretrain_model.lower() in {'rnn', 'lstm', 'gru'}:
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
    config_parser.add_argument('--pretrain_model', type=str, default='Transformer',
                               help='type of pretraining net: LSTM, GRU, RNN, Transformer (default)')
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
    config_parser.add_argument('--dropout', type=float, default=0.2,
                               help='dropout rate (default: 0.2).')
    config_parser.add_argument('--clip_value', type=float, default=1.0,
                               help='maximum value for the gradient norm (default: 1.0)')
    config_parser.add_argument('--n_total', type=int, default=-1,
                               help='number of samples to be used during training. By default use '
                                    'all the samples available. Useful for quick tests.')
    # parameters for recurrent netowrks
    config_parser.add_argument('--hidden_size_rnn', type=int, default=800,
                               help="Hidden size rnn. Default is 800.")
    config_parser.add_argument('--num_layers', type=int, default=1,
                               help="Number of layers. Default is 1.")
    config_parser.add_argument('--k_steps_ahead', nargs='+', type=int, default=[10, 20, 25, 50, 75, 90, 100],
                               help='Try to predict k steps ahead')
    # parameters for transformer
    config_parser.add_argument('--num_heads', type=int, default=2,
                               help="Number of attention heads. Default is 5.")
    config_parser.add_argument('--num_trans_layers', type=int, default=2,
                               help="Number of transformer blocks. Default is 2.")
    config_parser.add_argument('--emb_size', type=int, default=50,
                               help="Embedding size for transformer. Default is 50.")
    config_parser.add_argument('--hidden_size_trans', type=int, default=50,
                               help="Hidden size transformer. Default is 50.")
    config_parser.add_argument('--num_masked_subseq', type=int, default=5,
                               help="Number of attention masked subsequences. Default is 75.")
    config_parser.add_argument('--num_masked_samples', type=int, default=8,
                               help="Number of attention masked consecutive samples. Default is 8.")

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
    rng = random.Random(args.seed)

    tqdm.write("Define dataset...")
    dset = ECGDataset(settings.input_folder, freq=args.sample_freq)
    # Get length
    n_total = len(dset) if args.n_total <= 0 else min(args.n_total, len(dset))
    n_valid = int(n_total * args.valid_split)
    n_train = n_total - n_valid
    tqdm.write("\t train: {:d} ({:2.2f}\%) samples, valid: {:d}({:2.2f}\%) samples"
               .format(n_train, 100*n_train/n_total,n_valid, 100*n_valid/n_total))
    # Get ids
    all_ids = dset.get_ids()
    rng.shuffle(all_ids)
    train_ids = all_ids[:n_train]
    valid_ids = all_ids[n_train:n_total]
    # Save train and test ids
    with open(os.path.join(folder, 'pretrain_train_ids.txt'), 'w') as f:
        f.write(','.join(train_ids))
    with open(os.path.join(folder, 'pretrain_valid_ids.txt'), 'w') as f:
        f.write(','.join(valid_ids))
    # Define dataset
    train_dset = dset[train_ids]
    train_samples = list(itertools.chain(*[split_long_signals(s) for s in train_dset]))
    valid_dset = dset[valid_ids]
    valid_samples = list(itertools.chain(*[split_long_signals(s) for s in valid_dset]))

    # Get number of batches
    tqdm.write("Done!")

    tqdm.write("Define model...")
    if args.pretrain_model.lower() == 'transformer':
        model = MyTransformer(vars(args))
    elif args.pretrain_model.lower() in {'rnn', 'lstm', 'gru'}:
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

    history = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss", "lr", ])
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
