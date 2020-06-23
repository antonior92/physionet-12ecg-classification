import torch
import torch.nn as nn


class DummyPredictionStage(nn.Module):
    """This is a dummy prediction stage which has no model but just passes the input to the output"""

    def __init__(self):
        super(DummyPredictionStage, self).__init__()

    def forward(self, inp):
        return inp


class RNNPredictionStage(nn.Module):
    def __init__(self, args, n_classes):  # n_diagnoses, hidden_size=20, num_layers=1):
        super(RNNPredictionStage, self).__init__()
        self.n_classes = n_classes
        self.hidden_size = args['pred_stage_hidd']
        self.num_layers = args['pred_stage_n_layer']

        self.hidden_layer = None
        self.sub_ids = None

        self.rnn = getattr(nn, args['pred_stage_type'].upper())(self.n_classes, self.hidden_size, self.num_layers)
        self.linear = nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, model_out):
        # get rnn input and hidden layer
        rnn_input, hidden_layer = self.get_rnn_input(model_out, self.hidden_layer)

        # compute output of prediction stage
        out1, self.hidden_layer = self.rnn(rnn_input, hidden_layer)
        output = self.linear(out1)

        # output is of size (seq_len, batch_size, hidden_size) but only last sequence point is outputted
        return output[-1]

    def init_hidden_layer(self, batch_size, device):
        # hidden is of size (num_layer, batch_size, hidden_size)
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def get_rnn_input(self, model_out, hidden_layer):
        device = model_out.device
        batch_size = model_out.shape[0]
        # hidden layer definition
        if hidden_layer is None:
            hidden_layer_new = self.init_hidden_layer(batch_size, device)
        else:
            # create mask from sub_ids of dimension as hidden_layer
            mask = torch.tensor(self.sub_ids, device=device).view(1, -1, 1)
            mask = mask.repeat(self.num_layers, 1, self.hidden_size)
            # apply sub_ids as mask: set hidden_layer[:,i,:]=0 if sub_ids[i]==0
            hidden_layer_new = hidden_layer[:, :batch_size].masked_fill(mask == 0, float(0))
            # detach hidden layer since it should contain all previous information and BPTT is computationally too
            # expensive
            hidden_layer_new = hidden_layer_new.detach()

        # model output as input to classifier
        rnn_input = model_out.unsqueeze(0)

        # return rnn input and hidden layer
        return rnn_input, hidden_layer_new

    def update_sub_ids(self, sub_ids):
        self.sub_ids = sub_ids

    def reset(self):
        self.hidden_layer = None
        self.sub_ids = None
