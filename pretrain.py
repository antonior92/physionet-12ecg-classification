import json
import torch
import argparse
import datetime
import pandas as pd
from warnings import warn
from data import *
from tqdm import tqdm
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import random

from data.ecg_dataloader import ECGBatchloader
from models_pretrain.transformer_pretrain import MyTransformer
from models_pretrain.rnn_pretrain import MyRNN
from models_pretrain.transformerxl_pretrain import MyTransformerXL


def selfsupervised(ep, model, optimizer, loader, loss, device, args, train):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    n_entries = 0
    str_name = 'train' if train else 'val'
    desc = "Epoch {:2d}: {} - Loss: {:.6f}"
    bar = tqdm(initial=0, leave=True, total=len(loader), desc=desc.format(ep, str_name, 0), position=0)
    # create initial memory (required for transformer xl)
    mems = []
    if args.pretrain_model.lower() == 'transformerxl':
        param = next(model.parameters())
        empty = torch.zeros(args.mem_len, args.batch_size, args.dim_model,
                            dtype=param.dtype, device=param.device, requires_grad=False)
        for _ in range(args.num_trans_layers + 1):
            mems.append(empty)
    # loop over all batches
    for i, batch in enumerate(loader):
        # Send to device
        traces, ids, sub_ids = batch
        traces = traces.to(device=device)
        # create model input and targets
        inp, target = model.get_input_and_targets(traces)
        if args.pretrain_model.lower() == 'transformerxl':
            if len(mems) is not 0:
                # reset memory depending on sub_ids (required for transformer xl)
                # create mask to only change if sub_ids is zero
                mask = (torch.tensor(sub_ids) != 0).float().to(device=param.device)
                mask = mask[None, :, None]
                mask = mask.repeat(args.mem_len, 1, args.dim_model)
                for j in range(args.num_trans_layers + 1):
                    mems[j] = mems[j] * mask
        if train:
            # Reinitialize grad
            model.zero_grad()
            # Forward pass
            output, mems = model(inp, mems)
            ll = loss(output, target)
            # Backward pass
            ll.backward()
            clip_grad_norm_(model.parameters(), args.clip_value)
            # Optimize
            optimizer.step()
        else:
            with torch.no_grad():
                output, mems = model(inp, mems)
                ll = loss(output, target)
        # Update
        total_loss += ll.detach().cpu().numpy()
        bs = traces.size(0)
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
                               help='type of pretraining net: LSTM, GRU, RNN, Transformer (default), Transformer XL')
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
                               help='milestones for lr scheduler (default: [40, 75, 100])')
    config_parser.add_argument("--lr_factor", type=float, default=0.1,
                               help='reducing factor for the lr in a plateau (default: 0.1)')
    config_parser.add_argument('--dropout', type=float, default=0.2,
                               help='dropout rate (default: 0.2).')
    config_parser.add_argument('--clip_value', type=float, default=1.0,
                               help='maximum value for the gradient norm (default: 1.0)')
    config_parser.add_argument('--n_total', type=int, default=-1,
                               help='number of samples to be used during training. By default use '
                                    'all the samples available. Useful for quick tests.')
    # parameters for recurrent networks
    config_parser.add_argument('--hidden_size_rnn', type=int, default=800,
                               help="Hidden size rnn. Default is 800.")
    config_parser.add_argument('--num_layers', type=int, default=1,
                               help="Number of layers. Default is 1.")
    config_parser.add_argument('--k_steps_ahead', nargs='+', type=int, default=[10, 20, 25, 50, 75, 90, 100],
                               help='Try to predict k steps ahead')
    # parameters for transformer network
    config_parser.add_argument('--num_heads', type=int, default=4,
                               help="Number of attention heads. Default is 4.")
    config_parser.add_argument('--num_trans_layers', type=int, default=3,
                               help="Number of transformer blocks. Default is 4.")
    config_parser.add_argument('--dim_model', type=int, default=256,
                               help="Internal dimension of transformer. Default is 512.")
    config_parser.add_argument('--dim_inner', type=int, default=768,
                               help="Size of the FF network in the transformer. Default is 2048.")
    config_parser.add_argument('--steps_concat', type=int, default=4,
                               help='number of concatenated time steps for model input. Default is 4')
    # additional parameters for transformer xl
    config_parser.add_argument('--mem_len', type=int, default=100,
                               help="Memory length of transformer xl. Default is 1000.")
    config_parser.add_argument('--dropout_attn', type=float, default=0.2,
                               help='attention mechanism dropout rate. Default is 0.2.')
    config_parser.add_argument('--init_std', type=float, default=0.02,
                               help='standard deviation of normal initialization. Default is 0.02.')
    # training types for transformer
    config_parser.add_argument('--trans_train_type', type=str, default='flipping',
                               help='Type of transformer training type: masking (default), flipping')
    config_parser.add_argument('--train_noise_std', type=float, default=0.0,
                               help='Standard deviation of Gaussian noise on transformer input for training. '
                                    'Default is 0.1')
    config_parser.add_argument('--num_masked_samples', type=int, default=8,
                               help="Number of consecutive samples masked for attention. Default is 8.")
    config_parser.add_argument('--perc_masked_samp', type=int, default=0.15,
                               help="Percentage of total masked samples. Default is 0.15.")
    args, rem_args = config_parser.parse_known_args()
    # System setting
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument('--input_folder', type=str, default='Training_WFDB',
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
    # Set output folder
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
    # TODO: double check if drop_last works properly
    drop_last = True if args.pretrain_model.lower() == 'transformerxl' else False
    train_loader = ECGBatchloader(dset, train_ids, batch_size=args.batch_size,
                                  length=args.seq_length, drop_last=drop_last)
    valid_loader = ECGBatchloader(dset, valid_ids, batch_size=args.batch_size,
                                  length=args.seq_length, drop_last=drop_last)
    tqdm.write("\t train:  {:d} ({:2.2f}\%) ECG records divided into {:d} samples of fixed length"
               .format(n_train, 100 * n_train / n_total, len(train_loader))),
    tqdm.write("\t valid:  {:d} ({:2.2f}\%) ECG records divided into {:d} samples of fixed length"
               .format(n_valid, 100 * n_valid / n_total, len(valid_loader)))
    tqdm.write("Done!")

    tqdm.write("Define model...")
    if args.pretrain_model.lower() == 'transformerxl':
        model = MyTransformerXL(vars(args))
    elif args.pretrain_model.lower() == 'transformer':
        model = MyTransformer(vars(args))
    elif args.pretrain_model.lower() in {'rnn', 'lstm', 'gru'}:
        model = MyRNN(vars(args))
    model.to(device=device)
    message = "Done! Chosen model: {}".format(args.pretrain_model)
    tqdm.write(message)

    tqdm.write("Define optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_factor)
    tqdm.write("Done!")

    tqdm.write("Define loss...")
    if args.trans_train_type.lower() == 'masking':
        loss = nn.MSELoss(reduction='sum')
    elif args.trans_train_type.lower() == 'flipping':
        loss = nn.CrossEntropyLoss()  # reduction='sum'
    tqdm.write("Done!")

    history = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss", "lr", ])
    best_loss = np.Inf
    for ep in range(args.epochs):
        train_loss = selfsupervised(ep, model, optimizer, train_loader, loss, device, args, train=True)
        valid_loss = selfsupervised(ep, model, optimizer, valid_loader, loss, device, args, train=False)
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
