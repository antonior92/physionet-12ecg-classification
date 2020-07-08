import torch
import torch.nn as nn


class PretrainedRNNBlock(nn.Module):
    """Get reusable part from MyRNN and return new model. Include Linear block with the given output_size."""

    def __init__(self, pretrained, output_size, freeze=True):
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
        self.k_steps_ahead = args['k_steps_ahead']

    def forward(self, inp, dummyvar):
        o1, _ = self.rnn(inp.transpose(1, 2))
        o2 = self.linear(o1)
        return o2.transpose(1, 2), []

    def get_pretrained(self, output_size, finetuning=False):
        freeze = not finetuning
        return PretrainedRNNBlock(self, output_size, freeze)

    def get_input_and_targets(self, traces):
        max_steps_ahead = max(self.k_steps_ahead)
        inp = traces[:, :, :-max_steps_ahead]  # Try to predict k steps ahead
        n = inp.size(2)
        target = torch.cat([traces[:, :, k:k + n] for k in self.k_steps_ahead], dim=1)
        return inp, target