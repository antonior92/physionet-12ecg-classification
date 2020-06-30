import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_pretrain.masks_transformer import *

class PretrainedTransformerXLBlock(nn.Module):
    """  # Get reusable part from MyTransformerXL and return new model.
         # Include Linear block with the given output_size.
    """
    def __init__(self, pretrained, output_size, freeze=False):
        super(PretrainedTransformerXLBlock, self).__init__()
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
    def __init__(self, dmod):
        super(PositionalEncoding, self).__init__()

        self.dmod = dmod

        inv_freq = 1 / (10000 ** (torch.arange(0.0, dmod, 2.0) / dmod))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inp):
        return self.CoreNet(inp)


class MultiHeadAttnRelPos(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0):
        super(MultiHeadAttnRelPos, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.scale = 1 / (d_head ** 0.5)

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        # tot_length is combines length of memory and segment
        tot_len, bsz, _ = w.size()
        seg_len = r.shape[0]

        cat = torch.cat([mems, w], 0)

        w_heads = self.qkv_net(cat)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        w_head_q = w_head_q[-tot_len:]

        w_head_q = w_head_q.view(tot_len, bsz, self.n_head, self.d_head)  # tot_len x bsz x n_head x d_head
        w_head_k = w_head_k.view(seg_len, bsz, self.n_head, self.d_head)  # seg_len x bsz x n_head x d_head
        w_head_v = w_head_v.view(seg_len, bsz, self.n_head, self.d_head)  # seg_len x bsz x n_head x d_head

        r_head_k = self.r_net(r)
        r_head_k = r_head_k.view(seg_len, self.n_head, self.d_head)  # seg_len x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # tot_len x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # tot_len x tot_len x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # tot_len x tot_len x bsz x n_head
        BD = self._rel_shift(BD)

        # [tot_len x tot_len x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None:
            # convert mask
            attn_mask = attn_mask.float().masked_fill(attn_mask == float('-inf'), float(1.0)).bool().to(
                device=cat.device)
            # apply mask
            attn_score.masked_fill_(attn_mask[:, :, None, None], -float('inf'))

        # [tot_len x tot_len x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [tot_len x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        return attn_out

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)
        return x


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(MyTransformerEncoderLayer, self).__init__()

        # Relative Multihead Attention layer
        self.dec_attn = MultiHeadAttnRelPos(n_head, d_model, d_head, dropout, **kwargs)
        # Layer normalization
        self.layer_norm_attn = nn.LayerNorm(d_model)
        # Feedforward network (including residual connection and layer normalization)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)
        # Layer normalization
        self.layer_norm_ff = nn.LayerNorm(d_model)

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        # multihead attention
        output_attn = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias, attn_mask=attn_mask, mems=mems)
        # residual connection and layer normalization
        output_attn = self.layer_norm_attn(dec_inp + output_attn)
        # residual connection and position wise FF network
        output_ff = self.pos_ff(output_attn)
        # residual connection and layer normalization
        output = self.layer_norm_ff(output_attn + output_ff)

        return output


class MyTransformerXL(nn.Module):
    def __init__(self, args):  # n_layer, n_head, d_head, d_inner, d_model, dropout, dropout_attn, mem_len):
        super(MyTransformerXL, self).__init__()

        N_LEADS = 12
        self.num_masked_samples = args['num_masked_samples']
        self.perc_masked_samples = args['perc_masked_samp']

        self.bsz = args["batch_size"]
        self.seg_len = args["seq_length"]

        self.d_model = args["dim_model"]
        self.d_inner = args["dim_inner"]
        self.n_layer = args["num_trans_layers"]
        self.n_head = args["num_heads"]
        self.d_head = self.d_model // self.n_head  # args["dim_head"]
        self.dropout = args["dropout"]
        self.dropout_attn = args["dropout_attn"]
        self.mem_len = args["mem_len"]

        self.input_encoding = nn.Linear(N_LEADS, args["dim_model"])
        self.drop = nn.Dropout(args["dropout"])

        self.layers = nn.ModuleList()
        for i in range(self.n_layer):
            self.layers.append(
                MyTransformerEncoderLayer(self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                                          dropatt=self.dropout_attn)
            )

        self.decoder = nn.Linear(self.d_model, N_LEADS)

        # create parameters
        self.pos_enc = PositionalEncoding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        # initialize
        self.r_w_bias = nn.init.normal_(self.r_w_bias, 0.0, 0.02)
        self.r_r_bias = nn.init.normal_(self.r_r_bias, 0.0, 0.02)

    def reset_length(self, mem_len):
        self.mem_len = mem_len

    def _update_mems(self, hids, mems, mem_len_curr, seg_len):
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mem_len_curr + seg_len` steps that can be cached into mems
        with torch.no_grad():
            new_mems = []
            # indices for memory to take
            end_idx = mem_len_curr + max(0, seg_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                # combine previous hidden states with new ones
                cat = torch.cat([mems[i], hids[i]], dim=0)
                # only take the last self.mem_len hidden states as new memory
                # detach the memory to stop propagating it
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems

    def forward(self, src, mems):
        # input is shape (batch_size, N_LEADS, seq_len)
        # and is reshaped to (seq_len, batch_size, N_LEADS)
        src = src.permute(2, 0, 1)

        # compute the attention mask
        mem_len_curr = mems[0].size(0)
        k_len = mem_len_curr + self.seg_len
        attn_mask = generate_random_sequence_mask(self.seg_len, k_len, self.num_masked_samples,
                                                  self.perc_masked_samples)

        # compute word embeddings
        src_enc = self.input_encoding(src)
        core_out = self.drop(src_enc)

        # compute absolute position within the segment (end_indx : 0)
        pos_seq = torch.arange(k_len - 1, -1, -1.0, device=src_enc.device, dtype=src_enc.dtype)
        pos_enc = self.pos_enc(pos_seq)
        pos_enc = self.drop(pos_enc)

        # store the hidden states (input to transformer and each multihead attention layer)
        hids = []
        hids.append(core_out)

        # loop over all multihead attention layers (each transformer block)
        for i, layer in enumerate(self.layers):
            # get the currently used memory
            mems_i = None if mems is None else mems[i]
            # compute output of on multihead attention layer
            core_out = layer(core_out, pos_enc, self.r_w_bias, self.r_r_bias, attn_mask=attn_mask, mems=mems_i)
            # store hidden state
            hids.append(core_out)
        core_out = self.drop(core_out)

        # decode signal
        pred = self.decoder(core_out)
        # pred is shape (seq_len, batch_size, N_LEADS)
        # and is reshaped to (batch_size, N_LEADS, seq_len)
        pred = pred.permute(1, 2, 0)

        # update the memory
        new_mems = self._update_mems(hids, mems, mem_len_curr, self.seg_len)

        return pred, new_mems

    def get_input_and_targets(self, traces):
        return traces, traces
