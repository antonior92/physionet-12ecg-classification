from tqdm import tqdm
import math
import torch.nn as nn
import torch.distributions
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from models_pretrain.masks_transformer import *
import copy


class PretrainedTransformerBlock(nn.Module):
    """Get reusable part from MyTransformer and return new model. Include Linear block with the given output_size."""

    def __init__(self, pretrained, output_size, freeze=True):
        super(PretrainedTransformerBlock, self).__init__()
        self.freeze = freeze
        self.N_LEADS = 12
        self.output_size = output_size
        self.steps_concat = pretrained.steps_concat
        self.model_size = pretrained.decoder._modules['0'].in_features

        self.encoder = pretrained.encoder
        self.pos_encoder = pretrained.pos_encoder
        self.transformer_encoder = pretrained.transformer_encoder

        if self.freeze:
            tqdm.write("...transformer requires_grad set to False (no finetuning)...")
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.pos_encoder.parameters():
                param.requires_grad = False
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False
        else:
            tqdm.write("...transformer requires_grad set to True (finetuning)...")

        # self.encoder.out_features is also the output feature size of the transformer
        self.decoder = nn.Linear(self.model_size, self.output_size)

    def forward(self, src):
        batch_size, n_feature, seq_len = src.shape
        # concatenate neighboring samples in feature channel
        src1 = src.transpose(2, 1).reshape(-1, seq_len // self.steps_concat, n_feature * self.steps_concat)

        # process data (no mask in transformer used)
        src2 = self.encoder(src1) * math.sqrt(self.N_LEADS)
        src3 = self.pos_encoder(src2)
        src4 = src3.transpose(0, 1)
        src5 = self.transformer_encoder(src4)

        # src5.shape = (seq_length / step_concat), batch_size, (dim_model * steps_concat)
        # de-concatenate
        out1 = src5.permute(1, 0, 2).reshape(batch_size, seq_len, -1)
        out2 = out1.transpose(0, 1)
        # decode
        out3 = self.decoder(out2)
        # out3.shape = seq_length, batch_size, decoder_size
        # to right shape for training stage, i.e. batch_size, decoder_size, seq_length
        output = out3.permute(1, 2, 0)

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
        self.seq_len = args['seq_length']
        # number of total subsequences of length num_masked_samples to be masked
        self.num_subseq = int(self.perc_masked_samples * self.seq_len // self.num_masked_samples)
        self.MASK_VAL = 10

        self.N_LEADS = 12
        self.dim_concat = int(self.N_LEADS * args['steps_concat'])
        self.dim_model = args["dim_model"]
        self.emb_dim = self.dim_model * args['steps_concat']
        self.steps_concat = args['steps_concat']

        self.encoder = nn.Linear(self.dim_concat, self.emb_dim)
        self.pos_encoder = PositionalEncoding(self.emb_dim, args['dropout_attn'])
        encoder_layers = TransformerEncoderLayer(self.emb_dim, args['num_heads'],
                                                 args['dim_inner'], args['dropout_attn'])
        self.transformer_encoder = TransformerEncoder(encoder_layers, args['num_trans_layers'])
        if self.trans_train_type.lower() == 'masking':
            self.dense = nn.Linear(self.dim_model, self.dim_model)
            self.layer_norm = nn.LayerNorm(self.dim_model)
            self.decoder_layer = nn.Linear(self.dim_model, self.N_LEADS)
            self.decoder = nn.Sequential(
                self.dense,
                self.layer_norm,
                nn.ReLU(),
                self.decoder_layer
            )
        elif self.trans_train_type.lower() == 'flipping':
            self.decoder = nn.Linear(self.emb_dim, self.N_LEADS)
            self.decoder_class = nn.Linear(self.seq_len // self.steps_concat, 4)

    def forward(self, src, dummyvar):

        batch_size, n_feature, seq_len = src.shape

        # concatenate neighboring samples in feature channel
        # since attention requires n^2 memory and flops with n=seq_length
        src1 = src.transpose(2, 1).reshape(-1, seq_len // self.steps_concat, n_feature * self.steps_concat)

        # process data
        src2 = self.encoder(src1) * math.sqrt(self.N_LEADS)
        src3 = self.pos_encoder(src2)
        src4 = src3.transpose(0, 1)
        src5 = self.transformer_encoder(src4)

        # src5.shape = (seq_length / step_concat), batch_size, (dim_model * steps_concat)
        # de-concatenate
        out1 = src5.transpose(0, 1).reshape(batch_size, seq_len, -1)
        out2 = out1.transpose(0, 1)
        # decode
        out3 = self.decoder(out2)
        output = out3.permute(1, 2, 0)

        return output, []

    def get_pretrained(self, output_size, finetuning=False):
        freeze = not finetuning
        return PretrainedTransformerBlock(self, output_size, freeze)

    def get_input_and_targets(self, traces):
        # CASE: Masking
        if self.trans_train_type.lower() == 'masking':
            """
            inp: input to the model, masked traces. All 12 leads are masked equally
            target: model targets which are values of masked samples
            """
            def repeat_and_expand(mask_individ):
                # repeat each column self.num_masked_samples times
                temp = [col for col in mask_individ.t() for _ in range(self.num_masked_samples)]
                masked_indices = torch.stack(temp).t()
                # expand for N_LEADS
                mask = masked_indices.unsqueeze(1).repeat(1, self.N_LEADS, 1)
                return mask

            batch_size = traces.size(0)
            inp = traces.clone()
            target = traces.clone()

            # mask out random subsequences:
            # generate probability matrix for 1 sample and then scale up to self.num_masked_samples
            shape_individ = [batch_size, self.seq_len // self.num_masked_samples]
            probability_matrix = torch.full(shape_individ, self.perc_masked_samples)
            masked_indices_individ = torch.bernoulli(probability_matrix).bool()
            masked_indices = repeat_and_expand(masked_indices_individ)

            # 80% are masked by replacing the input with MASK_VAL
            indices_replaced_individ = torch.bernoulli(torch.full(shape_individ, 0.8)).bool() & masked_indices_individ
            indices_replaced = repeat_and_expand(indices_replaced_individ)
            inp[indices_replaced] = self.MASK_VAL

            # 10% are disturbed by noise
            indices_random_individ = torch.bernoulli(torch.full(shape_individ, 0.5)).bool() & masked_indices_individ & ~indices_replaced_individ
            indices_random = repeat_and_expand(indices_random_individ)
            num_elem = indices_random.int().sum()
            noise = torch.normal(torch.zeros(num_elem), self.train_noise_std * torch.ones(num_elem)).to(device=inp.device)
            inp[indices_random] = inp[indices_random] + noise

            # remaining 10% are left originally
            return inp, target, masked_indices.detach()

        # CASE: Flipping
        if self.trans_train_type.lower() == 'flipping':
            """how is it flipped:
            target==0: input stays the same (input)
            target==1: mirror on x-axis (-1*input)
            target==2: reverse/flip signal (input.flip(dim=-1))
            target==3: mirror and reverser (-1*input.flip(dim=-1))
            """
            # parameter
            num_flips = 4
            batch_size = traces.shape[0]

            # flipping function
            def f0(x):
                return x

            def f1(x):
                return -x

            def f2(x):
                return x.flip(-1)

            def f3(x):
                return -x.flip(-1)

            # get targets
            m = torch.distributions.categorical.Categorical(
                1 / num_flips * torch.ones(batch_size, self.N_LEADS, num_flips))
            target = m.sample()

            # flip input according to target
            inp = copy.deepcopy(traces)
            for flip_nr in range(num_flips):
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

            target_out = target.view(-1).to(device=inp.device)
            target_mask = torch.ones_like(target_out).bool().detach()
            return inp, target_out, target_mask
