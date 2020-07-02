import torch
import torch.nn as nn


class LinearPredictionStage(nn.Module):
    def __init__(self, model_output_dim, n_classes):
        super(LinearPredictionStage, self).__init__()

        self.lin = nn.Linear(model_output_dim, n_classes)

    def forward(self, x):
        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.lin(x)

        return x


class RNNPredictionStage(nn.Module):
    def __init__(self, args, n_classes):  # n_diagnoses, hidden_size=20, num_layers=1):
        super(RNNPredictionStage, self).__init__()
        # from ConvNet output in ResNet
        self.n_filters_last = args['net_filter_size'][-1]
        self.n_samples_last = args['net_seq_length'][-1]

        # self.d_model_out = self.n_filters_last * self.n_samples_last
        self.n_classes = n_classes
        self.hidden_size = args['pred_stage_hidd']
        self.num_layers = args['pred_stage_n_layer']

        self.rnn_input = None
        self.sub_ids = None

        self.rnn = getattr(nn, args['pred_stage_type'].upper())(self.n_filters_last, self.hidden_size, self.num_layers)
        self.skip_connection = nn.Linear(self.n_filters_last, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size * self.n_samples_last, self.n_classes)

    def forward(self, model_out):
        # permute to shape (n_samples_last, batch_size, n_filters_last)
        model_out = model_out.permute(2, 0, 1)

        # compute RNN output
        self.rnn_input, hidden_layer = self.get_rnn_input(model_out)
        out_rnn, _ = self.rnn(self.rnn_input, hidden_layer)

        # skip connection layer
        out_skip = self.skip_connection(model_out)

        # sum last layer of rnn with skip connection
        out_sum = out_rnn[-self.n_samples_last:, ...] + out_skip[-self.n_samples_last:, ...]

        # flatten tensor
        out_sum = out_sum.transpose(0, 1)
        out_sum = out_sum.reshape(out_sum.size(0), -1)

        # final layer
        output = self.linear(out_sum)

        # output is of size (1, batch_size, hidden_size) since only last sequence point (from RNN) is outputted
        return output

    def init_hidden_layer(self, batch_size, device):
        # hidden is of size (num_layer, batch_size, hidden_size)
        hidden_layer = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        return hidden_layer

    def init_rnn_input(self, batch_size, device):
        # rnn input is generally of size (seq_length, batch_size, input_size)
        rnn_input = torch.zeros(self.n_samples_last, batch_size, self.n_filters_last, device=device)

        return rnn_input

    def get_rnn_input(self, model_out):
        device = model_out.device
        batch_size = model_out.shape[1]

        # hidden layer definition
        hidden_layer = self.init_hidden_layer(batch_size, device)

        # RNN input definition
        if self.rnn_input is None:
            rnn_input = self.init_rnn_input(batch_size, device)
        else:
            rnn_input = self.rnn_input

        max_seq_len = (max(self.sub_ids) + 1) * self.n_samples_last
        # concatenate with old inputs (which are detached by .data)
        rnn_input = torch.cat([rnn_input[:, :batch_size].data, model_out], dim=0)[-max_seq_len:]
        # create mask as: set rnn_input_state[-sub_ids[i]:,i,:]=0 if sub_ids[i]=0
        mask = torch.zeros(max_seq_len, batch_size, device=device)
        sub_ids_tens = torch.tensor(self.sub_ids)
        block_len = int(max_seq_len / self.n_samples_last)
        for t in range(block_len):
            idx_lo = t * self.n_samples_last
            idx_hi = (t + 1) * self.n_samples_last
            mask[idx_lo:idx_hi, :] = (-(sub_ids_tens + 1) <= -block_len + t)
        mask = mask.unsqueeze(2).repeat(1, 1, self.n_filters_last)
        self.rnn_input = rnn_input.masked_fill(mask == 0, float(0))

        # return rnn input and hidden layer
        return self.rnn_input, hidden_layer

    def update_sub_ids(self, sub_ids):
        self.sub_ids = sub_ids

    def reset(self):
        self.rnn_input = None
        self.sub_ids = None
