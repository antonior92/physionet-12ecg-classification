import os
import json
import torch
import argparse
import datetime
from warnings import warn
from ecg_dataset import *
from tqdm import tqdm
from models.resnet import ResNet1d
from metrics import get_threshold, get_metrics


# %% Train model
def train(ep, model, optimizer, train_samples, ll, n_classes, device):
    model.train()
    total_loss = 0
    n_entries = 0
    n_train_final = len(train_samples)
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=n_train_final,
                     desc=train_desc.format(ep, 0), position=0)
    for i in range(n_train_batches):
        # Send to device
        bs = min(args.batch_size, n_train_final - n_entries + 1)
        samples = train_samples[n_entries:n_entries+bs]
        traces = torch.stack([torch.tensor(s['data'], dtype=torch.float32, device=device) for s in samples], dim=0)
        target = torch.stack([torch.tensor(s['output'], dtype=torch.float32, device=device) for s in samples], dim=0)
        # Reinitialize grad
        model.zero_grad()
        # Forward pass
        output = model(traces.transpose(2, 1))
        loss = ll(output, target)
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        total_loss += loss.detach().cpu().numpy() / n_classes
        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(bs)
    train_bar.close()
    return total_loss / n_entries


def eval(ep, model, valid_samples, ll, output_layer, n_classes, device):
    model.eval()
    total_loss = 0
    n_entries = 0
    all_outputs = np.zeros((len(valid_samples), n_classes), dtype=np.float32)
    all_targets = np.zeros((len(valid_samples), n_classes), dtype=np.float32)
    all_ids = np.zeros(len(valid_samples), dtype=int)
    eval_desc = "Epoch {0:2d}: valid - Loss: {1:.6f}" if ep >= 0 \
        else "{valid - Loss: {1:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(valid_samples),
                    desc=eval_desc.format(ep, 0), position=0)
    for i, samples in enumerate(valid_samples):
        with torch.no_grad():
            traces = torch.stack([torch.tensor(s['data'], dtype=torch.float32, device=device) for s in samples], dim=0)
            target = torch.tensor(samples[0]['output'], dtype=torch.float32, device=device)
            id = samples[0]['id']
            # Forward pass
            output = model(traces.transpose(2, 1))
            # Collapse along dim=0 (TODO: try different collapsing functions here)
            output = output_layer(output).mean(dim=0)
            # Loss
            loss = ll(output, target)
            # Add outputs
            all_outputs[i, :] = output.detach().cpu().numpy()
            all_targets[i, :] = target.detach().cpu().numpy()
            all_ids[i] = id
            # Update ids
            total_loss += loss.detach().cpu().numpy() / n_classes
            n_entries += 1
            # Print result
            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(1)
    eval_bar.close()
    return total_loss / n_entries, all_outputs, all_targets, all_ids


if __name__ == '__main__':
    # Experiment parameters
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--seed', type=int, default=2,
                               help='random seed for number generator (default: 2)')
    config_parser.add_argument('--epochs', type=int, default=70,
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
    config_parser.add_argument("--patience", type=int, default=7,
                               help='maximum number of epochs without reducing the learning rate (default: 7)')
    config_parser.add_argument("--min_lr", type=float, default=1e-7,
                               help='minimum learning rate (default: 1e-7)')
    config_parser.add_argument("--lr_factor", type=float, default=0.1,
                               help='reducing factor for the lr in a plateu (default: 0.1)')
    config_parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                               help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    config_parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                               help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    config_parser.add_argument('--dropout_rate', type=float, default=0.8,
                               help='dropout rate (default: 0.8).')
    config_parser.add_argument('--kernel_size', type=int, default=17,
                               help='kernel size in convolutional layers (default: 17).')

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
    with open(os.path.join(folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')
    # Set seed
    torch.manual_seed(args.seed)

    tqdm.write("Define dataset...")
    dset = ECGDataset(settings.input_folder, freq=args.sample_freq)
    # Get length
    n_classes = len(CLASSES)
    n_total = len(dset)
    n_valid = int(n_total * args.valid_split)
    n_train = n_total - n_valid
    # Define dataset
    train_samples = list(itertools.chain(*[split_long_signals(s) for s in dset[:10]]))
    valid_samples = [split_long_signals(s, length=args.seq_length) for s in dset[10:20]]
    # Get number of batches
    n_train_final = len(train_samples)
    n_train_batches = int(np.ceil(n_train_final/args.batch_size))
    tqdm.write("Done!")

    tqdm.write("Define model...")
    N_LEADS = 12
    model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                     blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                     n_classes=n_classes,
                     kernel_size=args.kernel_size,
                     dropout_rate=args.dropout_rate)
    model.to(device=device)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience,
                                                           min_lr=args.lr_factor*args.min_lr,
                                                           factor=args.lr_factor)
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    ll = torch.nn.BCEWithLogitsLoss(reduction='sum')
    output_layer = torch.nn.Sigmoid()
    tqdm.write("Done!")

    for ep in range(args.epochs):
        # Train and evaluate
        train_loss = train(ep, model, optimizer, train_samples, ll, n_classes, device)
        valid_loss, y_score, y_true, aids = eval(ep, model, valid_samples, ll, output_layer,
                                                            n_classes, device)
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < args.min_lr:
            break
        # Get metrics
        _, _, threshold = get_threshold(y_true, y_score)
        y_pred = y_score > threshold
        metrics = get_metrics(y_true, y_pred)
        message = 'Epoch {:2d}: \tTrain Loss {:.6f} ' \
                  '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t' \
                   'Fbeta: {:.3f} \tGbeta: {:.3f}' \
            .format(ep, train_loss, valid_loss, learning_rate,
                    metrics['f_beta'], metrics['g_beta'])
        tqdm.write(message)

