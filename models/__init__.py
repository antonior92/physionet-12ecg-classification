from .mlp import MlpClassifier
from .resnet import ResNet1d
from .prediction_model import LinearPredictionStage, RNNPredictionStage
from warnings import warn


def get_model(type, n_input_channels, seq_len, **kwargs):
    if type == 'resnet':
        # Remove blocks from the convolutional neural network if they are not in accordance with seq_len
        removed_blocks = 0
        for l in kwargs['net_seq_length']:
            if l > seq_len:
                del kwargs['net_seq_length'][0]
                del kwargs['net_filter_size'][0]
                removed_blocks += 1
        if removed_blocks > 0:
            warn("The output of the pretrain stage is not consistent with the conv net "
                 "structure. We removed the first n={:d} residual blocks.".format(removed_blocks)
                 + "the new configuration is " + str(list(zip(kwargs['net_filter_size'], kwargs['net_seq_length']))))
        # Get main model
        res_net = ResNet1d(input_dim=(n_input_channels, kwargs['seq_length']),
                           blocks_dim=list(zip(kwargs['net_filter_size'], kwargs['net_seq_length'])),
                           kernel_size=kwargs['kernel_size'], dropout_rate=kwargs['dropout_rate'])
        return res_net
    elif type == 'mlp':
        mlp = MlpClassifier(seq_len, n_input_channels)
        return mlp
    else:
        raise ValueError('Unknown type = {}'.format(type))


def get_prediction_stage(type, n_classes, n_input_channels, seq_len, **kwargs):
    if type in ['gru', 'lstm', 'rnn']:
        return RNNPredictionStage(n_classes, n_input_channels, seq_len, type,
                                  hidden_size=kwargs['pred_stage_hidd'],
                                  num_layers=kwargs['pred_stage_n_layer'])
    elif type == 'linear':
        return LinearPredictionStage(model_output_dim=n_input_channels * seq_len, n_classes=n_classes)
    else:
        raise ValueError('Unknown type = {}'.format(type))


