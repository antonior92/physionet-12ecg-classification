import os
import json
import torch
import argparse
import datetime
import pandas as pd
from warnings import warn
from ecg_dataset import *
from tqdm import tqdm
from models.resnet import ResNet1d
from metrics import get_threshold, get_metrics
from output_layer import OutputLayer


# %% Train model
def train(ep, model, optimizer, train_samples, out_layer, device):
    model.train()
    total_loss = 0
    n_entries = 0
    n_train_final = len(train_samples)
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=n_train_final,
                     desc=train_desc.format(ep, 0), position=0)
    for i in range(n_train_batches):
        # Send to device
        bs = min(args.batch_size, n_train_final - n_entries)
        samples = train_samples[n_entries:n_entries+bs]
        traces = torch.stack([torch.tensor(s['data'], dtype=torch.float32, device=device) for s in samples], dim=0)
        target = torch.stack([torch.tensor(s['output'], dtype=torch.long, device=device) for s in samples], dim=0)
        # Reinitialize grad
        model.zero_grad()
        # Forward pass
        output = model(traces)
        loss = out_layer.loss(output, target)
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        total_loss += loss.detach().cpu().numpy()
        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(bs)
    train_bar.close()
    return total_loss / n_entries


def evaluate(ep, model, valid_samples, out_layer, device, seq_length):
    model.eval()
    total_loss = 0
    n_entries = 0
    all_outputs = np.zeros((len(valid_samples), n_classes), dtype=np.float32)
    all_targets = np.zeros((len(valid_samples), n_target_vec), dtype=np.float32)
    all_ids = np.zeros(len(valid_samples), dtype=int)
    eval_desc = "Epoch {0:2d}: valid - Loss: {1:.6f}" if ep >= 0 \
        else "{valid - Loss: {1:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(valid_samples),
                    desc=eval_desc.format(ep, 0), position=0)
    for i, samples in enumerate(valid_samples):
        with torch.no_grad():
            # Compute model
            traces = torch.stack([torch.tensor(s['data'], dtype=torch.float32, device=device) for s in
                                  split_long_signals(samples, length=seq_length)], dim=0)
            n_subsamples = traces.size(0)
            id = samples['id']
            # Forward pass
            logits = model(traces)
            # Collapse along dim=0 (TODO: try different collapsing functions here)
            output = out_layer.get_output(logits).mean(dim=0)
            # Get loss
            target = torch.tensor(samples['output'], dtype=torch.float32, device=device)
            all_targets[i, :] = target.detach().cpu().numpy()
            # Loss
            loss = out_layer.loss(logits, torch.stack([target]*n_subsamples, dim=0))
            total_loss += loss.detach().cpu().numpy()
            # Add outputs
            all_outputs[i, :] = output.detach().cpu().numpy()
            all_ids[i] = id
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
    config_parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                               help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    config_parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                               help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    config_parser.add_argument('--dropout_rate', type=float, default=0.8,
                               help='dropout rate (default: 0.8).')
    config_parser.add_argument('--kernel_size', type=int, default=17,
                               help='kernel size in convolutional layers (default: 17).')
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
    with open(os.path.join(folder, 'config.json'), 'w') as f:
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
    valid_samples = dset[n_train:n_total]
    # Get number of batches
    n_train_final = len(train_samples)
    n_train_batches = int(np.ceil(n_train_final/args.batch_size))
    tqdm.write("Done!")

    tqdm.write("Define threshold ...")
    # Get all targets
    targets = np.stack([s['output'] for s in train_dset])
    targets_bin = multiclass_to_binaryclass(targets)
    threshold = targets_bin.sum(axis=0) / targets_bin.shape[0]
    tqdm.write("\t threshold = train_ocurrences / train_samples (for each abnormality)")
    tqdm.write("\t\t\t   = AF:{:.2f},I-AVB:{:.2f},RBBB:{:.2f},LBBB:{:.2f},PAC:{:.2f},PVC:{:.2f},STD:{:.2f},STE:{:.2f}"
               .format(*threshold))
    tqdm.write("Done!")

    tqdm.write("Define model...")
    N_LEADS = 12
    model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                     blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                     n_classes=len(CLASSES),
                     kernel_size=args.kernel_size,
                     dropout_rate=args.dropout_rate)
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
        train_loss = train(ep, model, optimizer, train_samples, out_layer, device)
        valid_loss, y_score, all_targets, ids = evaluate(ep, model, valid_samples, out_layer, device, args.seq_length)
        y_true = multiclass_to_binaryclass(all_targets)
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

