import argparse
import torch
import pandas as pd
from warnings import warn
from tqdm import tqdm

from data import *
from utils import set_output_folder, check_pretrain_model, get_data_ids, \
    write_data_ids, get_dataloaders, get_model, GetMetrics
from output_layer import OutputLayer, collapse, DxClasses, get_collapse_fun, get_dx
from evaluate_12ECG_score import (load_weights)


# %% Train model
def train(ep, model, optimizer, train_loader, out_layer, device, shuffle):
    model.train()
    total_loss = 0
    n_entries = 0
    # training progress bar
    if shuffle:
        train_loader.shuffle()
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(train_loader),
                     desc=train_desc.format(ep, 0), position=0)
    for i, batch in enumerate(train_loader):
        traces, target, ids, sub_ids = batch
        traces = traces.to(device=device)
        target = target.to(device=device)
        # update sub_ids for final prediction stage model
        if model[-1]._get_name() == 'RNNPredictionStage':
            model[-1].update_sub_ids(sub_ids)
        # Reinitialize grad
        model.zero_grad()
        # Forward pass
        output = model(traces)
        # softmax and sigmoid layer implicitly contained in the loss
        loss = out_layer.loss(output, target)
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        total_loss += loss.detach().cpu().numpy()
        bs = target.size(0)
        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(bs)
    train_bar.close()
    # reset prediction stage variables
    if model[-1]._get_name() == 'RNNPredictionStage':
        model[-1].reset()

    return total_loss / n_entries


def evaluate(ep, model, valid_loader, out_layer, device):
    model.eval()
    total_loss = 0
    n_entries = 0
    all_outputs = []
    all_targets = []
    all_ids = []
    eval_desc = "Epoch {0:2d}: valid - Loss: {1:.6f}" if ep >= 0 \
        else "{valid - Loss: {1:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(valid_loader),
                    desc=eval_desc.format(ep, 0), position=0)
    for i, batch in enumerate(valid_loader):
        if i < 25:
            with torch.no_grad():
                traces, target, ids, sub_ids = batch
                traces = traces.to(device=device)
                target = target.to(device=device)
                # update sub_ids for final prediction stage model
                if model[-1]._get_name() == 'RNNPredictionStage':
                    model[-1].update_sub_ids(sub_ids)
                # Forward pass
                logits = model(traces)
                output = out_layer.get_output(logits)
                # Loss
                loss = out_layer.loss(logits, target)
                # Get loss
                total_loss += loss.detach().cpu().numpy()
                # append
                all_targets.append(target.detach().cpu().numpy())
                all_outputs.append(output.detach().cpu().numpy())
                all_ids.extend(ids)
                bs = target.size(0)
                n_entries += bs
                # Print result
                eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
                eval_bar.update(bs)
    eval_bar.close()
    # reset prediction stage variables
    if model[-1]._get_name() == 'RNNPredictionStage':
        model[-1].reset()
    return total_loss / n_entries, np.concatenate(all_outputs), np.concatenate(all_targets), all_ids


if __name__ == '__main__':
    # Experiment parameters
    config_parser = argparse.ArgumentParser(add_help=False)
    # Learning parameters
    config_parser.add_argument('--seed', type=int, default=2,
                               help='random seed for number generator (default: 2)')
    config_parser.add_argument('--dont_shuffle', action='store_true',
                               help='dont shuffle training samples each epoch.')
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
                               help='fraction of the data used for validation (default: 0.3).')
    config_parser.add_argument('--test_split', type=float, default=0.10,
                               help='fraction of the data used for testing (default: 0.1).')
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
    config_parser.add_argument('--finetuning', type=bool, default=False,
                               help='when there is a pre-trained model, by default it '
                                    'freezes the weights of the pre-trained model, but with this option'
                                    'these weight will be fine-tuned during training. Default is False')
    config_parser.add_argument('--eval_transformer', type=bool, default=False,
                               help='Evaluates the transformer. If true a small MLP classifier is chosen instead of the '
                                    'ResNet + prediction stage (default: False).')
    # Model parameters
    config_parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                               help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    config_parser.add_argument('--net_seq_length', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                               help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    config_parser.add_argument('--dropout_rate', type=float, default=0.5,
                               help='dropout rate (default: 0.5).')
    config_parser.add_argument('--kernel_size', type=int, default=17,
                               help='kernel size in convolutional layers (default: 17).')
    config_parser.add_argument('--n_total', type=int, default=-1,
                               help='number of samples to be used during training. By default use '
                                    'all the samples available. Useful for quick tests.')
    # Final Predictor parameters
    config_parser.add_argument('--pred_stage_type', choices=['lstm', 'gru', 'rnn', 'mean', 'max'], default='mean',
                               help='type of prediction stage model: lstm, gru, rnn, mean (default), max.')
    config_parser.add_argument('--pred_stage_n_layer', type=int, default=1,
                               help='number of rnn layers in prediction stage (default: 2).')
    config_parser.add_argument('--pred_stage_hidd', type=int, default=400,
                               help='size of hidden layer in prediction stage rnn (default: 30).')
    config_parser.add_argument('--train_classes', choices=['dset', 'scored'], default='scored_classes',
                               help='what classes are to be used during training.')
    config_parser.add_argument('--valid_classes', choices=['dset', 'scored'], default='scored_classes',
                               help='what classes are to be used during evaluation.')
    config_parser.add_argument('--test_classes', choices=['scored'], default='scored_classes',
                               help='what classes are to be used during testing.')
    config_parser.add_argument('--outlayer', choices=['sigmoid', 'sigmoid-and-softmax', 'softmax'], default='softmax',
                               help='what is the type used for the output layer. Options are '
                                    '(sigmoid, sigmoid-and-softmax, softmax).')
    args, rem_args = config_parser.parse_known_args()
    # System setting
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument('--input_folder', type=str, default='Training_WFDB',
                            help='input folder.')
    sys_parser.add_argument('--dx', type=str, default='./dx',
                            help='Path to folder containing class information.')
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
    folder = set_output_folder(args, settings, prefix=[])
    # Set seed
    torch.manual_seed(args.seed)
    # Check if there is pretrained model in the given folder
    config_dict_pretrain_stage, ckpt_pretrain_stage, pretrain_ids = check_pretrain_model(folder)
    pretrain_train_ids, pretrain_valid_ids, pretrain_test_ids = pretrain_ids

    tqdm.write("Define dataset...")
    dset = ECGDataset(settings.input_folder, freq=args.sample_freq)
    tqdm.write("Done!")

    tqdm.write("Define train, validation and test splits...")
    # if pretrained ids are available (not empty)
    if pretrain_train_ids and pretrain_valid_ids and pretrain_test_ids:
        train_ids = pretrain_train_ids
        valid_ids = pretrain_valid_ids
        test_ids = pretrain_test_ids
    else:
        train_ids, valid_ids, test_ids = get_data_ids(dset, args)
    # Save train, validation and test ids
    write_data_ids(folder, train_ids, valid_ids, test_ids)
    tqdm.write("Done!")

    tqdm.write("Define output layer...")
    # Get all classes in the dataset
    dset_classes = dset.get_classes()
    dx, valid_classes = get_dx(dset_classes, args.train_classes, args.valid_classes, args.outlayer, settings.dx)
    out_layer = OutputLayer(args.batch_size, dx, device)
    tqdm.write("Done!")

    tqdm.write("Define metrics...")
    weights = load_weights(os.path.join(settings.dx, 'weights.csv'), valid_classes)
    NORMAL = '426783006'
    get_metrics = GetMetrics(weights, valid_classes.index(NORMAL))
    tqdm.write("Done!")

    tqdm.write("Get dataloaders...")
    train_loader = get_dataloaders(dset, train_ids, args, dx, seed=args.seed)
    valid_loader = get_dataloaders(dset, valid_ids, args, dx)
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

    history = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss", "lr", "f_beta", "g_beta", "geom_mean"])
    best_challenge_metric = -np.Inf
    # run over all epochs
    for ep in range(args.epochs):
        # Train and evaluate
        #train_loss = train(ep, model, optimizer, train_loader, out_layer, device, not args.dont_shuffle)
        valid_loss, y_score, all_targets, ids = evaluate(ep, model, valid_loader, out_layer, device)
        # Collapse entries with the same id:
        unique_ids, y_score = collapse(y_score, ids, fn=get_collapse_fun(args.pred_stage_type))
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
        if best_challenge_metric < metrics['challenge_metric']:
            # Save model
            torch.save({'epoch': ep,
                        'threshold': threshold,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_challenge_metric = metrics['challenge_metric']
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
