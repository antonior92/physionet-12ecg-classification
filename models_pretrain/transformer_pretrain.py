import random
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PretrainedTransformerBlock(nn.Module):
    """Get reusable part from MyTransformer and return new model. Include Linear block with the given output_size."""

    def __init__(self, pretrained, output_size,  freeze=False):
        super(PretrainedTransformerBlock, self).__init__()
        self.N_LEADS = 12
        self.emb_size = pretrained._modules['decoder'].out_features
        self.pos_encoder = pretrained._modules['pos_encoder']
        self.transformer_encoder = pretrained._modules['transformer_encoder']

        if freeze:
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False
            for param in self.pos_encoder.parameters():
                param.requires_grad = False
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False

        # self.encoder.out_features is also the output feature size of the transformer
        self.decoder = nn.Linear(self.emb_size, output_size)
        self.steps_concat = pretrained.steps_concat

    def forward(self, src):
        batch_size, n_feature, seq_len = src.shape
        # concatenate neighboring samples in feature channel
        src1 = src.transpose(2, 1).reshape(-1, seq_len // self.steps_concat, n_feature * self.steps_concat)
        # put in the right shape for transformer
        # src2.shape = (sequence length / steps_concat), batch size, (N_LEADS * steps_concat)
        src2 = src1.transpose(0, 1)
        # process data (no mask in transformer used)
        # src = self.encoder(src) * math.sqrt(self.N_LEADS)
        src3 = self.pos_encoder(src2)
        out1 = self.transformer_encoder(src3)
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
        self.N_LEADS = 12
        self.mask_param = [args['num_masked_subseq'], args['num_masked_samples']]
        emb_size = int(self.N_LEADS * args['steps_concat'])
        self.pos_encoder = PositionalEncoding(emb_size, args['dropout'])
        encoder_layers = TransformerEncoderLayer(emb_size, args['num_heads'], args['hidden_size_trans'],
                                                 args['dropout'])
        self.transformer_encoder = TransformerEncoder(encoder_layers, args['num_trans_layers'])
        self.decoder = nn.Linear(emb_size, emb_size)
        self.steps_concat = args['steps_concat']

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
        return mask.to(next(self.parameters()).device)

    def forward(self, src):
        batch_size, n_feature, seq_len = src.shape
        # concatenate neighboring samples in feature channel
        src1 = src.transpose(2, 1).reshape(-1, seq_len // self.steps_concat, n_feature * self.steps_concat)
        # put in the right shape for transformer
        # t2.shape = (sequence length / steps_concat), batch size, (N_LEADS * steps_concat)
        src2 = src1.transpose(0, 1)
        # generate random mask
        mask = self._generate_random_sequence_mask(len(src2), self.mask_param)
        # generate triangular mask ('predict next sample').
        self.mask = mask
        # process data
        src3 = self.pos_encoder(src2)
        out1 = self.transformer_encoder(src3, self.mask)
        out2 = self.decoder(out1)
        # Go back to original, without neigboring samples concatenated
        # out3.shape =  batch size, sequence length, n_feature
        out3 = out2.transpose(0, 1).reshape(-1, seq_len, n_feature)
        # Put in the right shape for transformer
        output = out3.transpose(1, 2)
        return output

    def get_pretrained(self, output_size, freeze=False):
        return PretrainedTransformerBlock(self, output_size, freeze)

    def get_input_and_targets(self, traces):
        return traces, traces