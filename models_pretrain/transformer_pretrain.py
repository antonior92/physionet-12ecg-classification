import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models_pretrain.masks_transformer import *


class PretrainedTransformerBlock(nn.Module):
    """Get reusable part from MyTransformer and return new model. Include Linear block with the given output_size."""

    def __init__(self, pretrained, output_size, freeze=False):
        super(PretrainedTransformerBlock, self).__init__()
        self.N_LEADS = 12
        self.dim_model = pretrained._modules['decoder'].in_features

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
        self.decoder = nn.Linear(self.dim_model, output_size)
        self.steps_concat = pretrained.steps_concat

    def forward(self, src):
        batch_size, n_feature, seq_len = src.shape
        # concatenate neighboring samples in feature channel
        src1 = src.transpose(2, 1).reshape(-1, seq_len // self.steps_concat, n_feature * self.steps_concat)
        # put in the right shape for transformer
        # src2.shape = (sequence length / steps_concat), batch size, (N_LEADS * steps_concat)
        src2 = src1.transpose(0, 1)

        # process data (no mask in transformer used)
        src3 = self.encoder(src2) * math.sqrt(self.N_LEADS)
        src4 = self.pos_encoder(src3)
        out1 = self.transformer_encoder(src4)
        out2 = self.decoder(out1)

        # permute to have the same dimensions as in the input
        output = out2.permute(1, 2, 0)
        return output


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
        self.trans_train_type = args['trans_train_type']
        self.train_noise_std = args['train_noise_std']
        self.num_masked_samples = args['num_masked_samples']
        self.perc_masked_samples = args['perc_masked_samp']
        self.discrete_input = args['discrete_input']
        self.seq_len = args['seq_length']

        self.N_LEADS = 12
        self.dim_concat = int(self.N_LEADS * args['steps_concat'])
        self.dim_model = args["dim_model"]
        self.steps_concat = args['steps_concat']

        self.encoder = nn.Linear(self.dim_concat, self.dim_model)
        self.pos_encoder = PositionalEncoding(self.dim_model, args['dropout'])
        encoder_layers = TransformerEncoderLayer(self.dim_model, args['num_heads'], args['dim_inner'], args['dropout'])
        self.transformer_encoder = TransformerEncoder(encoder_layers, args['num_trans_layers'])
        if self.trans_train_type.lower() == 'masking':
            self.decoder = nn.Linear(self.dim_model, self.dim_concat)
        elif self.trans_train_type.lower() == 'flipping':
            self.decoder1 = nn.Linear(self.dim_model, self.N_LEADS)
            self.decoder2 = nn.Linear(self.seq_len//self.steps_concat, 4)

    def forward(self, src, dummyvar):
        batch_size, n_feature, seq_len = src.shape
        # concatenate neighboring samples in feature channel
        src1 = src.transpose(2, 1).reshape(-1, seq_len // self.steps_concat, n_feature * self.steps_concat)
        # put in the right shape for transformer
        # t2.shape = (sequence length / steps_concat), batch size, (N_LEADS * steps_concat)
        src2 = src1.transpose(0, 1)
        # generate mask
        if self.trans_train_type.lower() == 'masking':
            # generate random mask
            self.mask = generate_random_sequence_mask(len(src2), len(src2), self.num_masked_samples,
                                                      self.perc_masked_samples).to(next(self.parameters()).device)
        elif self.trans_train_type.lower() == 'flipping':
            # no mask needed
            self.mask = torch.zeros(len(src2), len(src2)).to(next(self.parameters()).device)

        # process data
        src3 = self.encoder(src2) * math.sqrt(self.N_LEADS)
        src4 = self.pos_encoder(src3)
        out1 = self.transformer_encoder(src4, self.mask)

        if self.trans_train_type.lower() == 'masking':
            out2 = self.decoder(out1)
            # Go back to original, without neighboring samples concatenated
            # out3.shape =  batch size, sequence length, n_feature
            out3 = out2.transpose(0, 1).reshape(-1, seq_len, n_feature)
            # Put in the right shape for transformer
            output = out3.transpose(1, 2)
        elif self.trans_train_type.lower() == 'flipping':
            out2 = self.decoder1(out1)
            out3 = out2.permute(1, 2, 0)
            out4 = self.decoder2(out3)
            output = out4.view(-1, out4.size(-1))

        return output, []

    def get_pretrained(self, output_size, freeze=False):
        return PretrainedTransformerBlock(self, output_size, freeze)

    def get_input_and_targets(self, traces):
        if self.discrete_input:
            # if discrete input then make dtype=torch.float16 since all data are 16 bit.
            traces = traces.half()

        # CASE: Masking
        if self.trans_train_type.lower() == 'masking':
            noise = torch.normal(torch.zeros(traces.shape), self.train_noise_std * torch.ones(traces.shape))
            # return input, target
            inp = traces + noise
            if self.discrete_input:
                inp.half()
            target = traces
            return inp, target

        # CASE: Flipping
        if self.trans_train_type.lower() == 'flipping':
            """how is it flipped:
            target==0: input stays the same (input)
            target==1: mirror on x-axis (-1*input)
            target==2: reverse/flip signal (input.flip(dim=-1))
            target==3: mirror and reverser (-1*input.flip(dim=-1))
            """
            # parameter
            n_fips = 4
            batch_size = traces.shape[0]

            # flipping function
            f0 = lambda x: x
            f1 = lambda x: -x
            f2 = lambda x: x.flip(-1)
            f3 = lambda x: -x.flip(-1)

            # get targets
            m = torch.distributions.categorical.Categorical(1 / n_fips * torch.ones(batch_size, self.N_LEADS, n_fips))
            target = m.sample()

            # flip input according to target
            inp = traces
            for flip_nr in range(n_fips):
                idx = (target == flip_nr).nonzero()
                i = idx[:, 0]
                j = idx[:, 1]
                if flip_nr == 0:
                    inp[i, j, :] = f0(traces[i, j, :])
                elif flip_nr == 1:
                    inp[i, j, :] = f1(traces[i, j, :])
                elif flip_nr == 2:
                    inp[i, j, :] = f2(traces[i, j, :])
                elif flip_nr == 3:
                    inp[i, j, :] = f3(traces[i, j, :])

            return inp, target.view(-1)
