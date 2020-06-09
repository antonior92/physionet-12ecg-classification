import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, demb):
        super(PositionalEncoding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_embed, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_embed = d_embed
        self.d_inner = d_inner

        self.CoreNet = nn.Sequential(
            nn.Linear(d_embed, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_embed),
            nn.Dropout(dropout),
        )
    def forward(self, inp):
        return self.CoreNet(inp)


class MultiHeadAttnRelPos(nn.Module):
    def __init__(self, n_head, d_embed, d_head, dropout, dropatt=0):
        super(MultiHeadAttnRelPos, self).__init__()

        self.d_embed = d_embed
        self.n_head = n_head
        self.d_head = d_head

        self.scale = 1 / (d_head ** 0.5)

        self.qkv_net = nn.Linear(d_embed, 3 * n_head * d_head, bias=False)

        self.r_net = nn.Linear(self.d_embed, self.n_head * self.d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_embed, bias=False)


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
        if attn_mask is not None and attn_mask.any().item():
                attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

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
    def __init__(self, n_layer, n_head, d_head, d_inner, d_model, dropout, dropout_attn, mem_len):
        super(MyTransformerXL, self).__init__()

        N_LEADS = 12

        self.d_embed = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.mem_len = mem_len

        self.word_emb = nn.Linear(N_LEADS, d_model)
        self.drop = nn.Dropout(dropout)
        self.n_layer = n_layer

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                MyTransformerEncoderLayer(n_head, d_model, d_head, d_inner, dropout, dropatt=dropout_attn)
            )

        self.decoder = nn.Linear(d_model, N_LEADS)

        # create parameters
        self.pos_emb = PositionalEncoding(self.d_embed)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

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
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems


    def forward(self, data, target, *mems):
        # get the segment length and batch size
        seg_len, bsz = data.size()

        # compute word embeddings
        word_emb = self.word_emb(data)
        core_out = self.drop(word_emb)

        # compute the attention mask
        mem_len_curr = mems[0].size(0)
        k_len = mem_len_curr + seg_len
        dec_attn_mask = torch.triu(word_emb.new_ones(seg_len, k_len), diagonal=1 + mem_len_curr).bool()[:, :, None]

        # compute absolute position within the segment (end_indx : 0)
        pos_seq = torch.arange(k_len - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)

        # store the hidden states (input to transformer and each multihead attention layer)
        hids = []
        hids.append(core_out)

        # loop over all multihead attention layers (each transformer block)
        for i, layer in enumerate(self.layers):
            # get the currently used memory
            mems_i = None if mems is None else mems[i]
            # compute output of on multihead attention layer
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            # store hidden state
            hids.append(core_out)
        core_out = self.drop(core_out)

        # decode signal
        pred = self.decoder(core_out)

        # update the memory
        new_mems = self._update_mems(hids, mems, mem_len_curr, seg_len)

        return pred, new_mems
