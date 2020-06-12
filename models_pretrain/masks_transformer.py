import torch
import numpy as np


def generate_triangular_mask(sz_rows, sz_cols):
    mask = torch.triu(torch.ones(sz_rows, sz_cols), diagonal=1 + sz_cols-sz_rows).byte()
    mask = mask.float().masked_fill(mask == 0, float('0.0')).masked_fill(mask == 1, float('-inf'))
    return mask


def generate_random_sequence_mask(sz_rows, sz_cols, n_samples, perc_masking):
    """
    According to attention definition the same mask is used for all sequences in the batch.
    Mask is a [sz x sz] matrix. If the value [i,j] is masked by a value of -inf, then for the
    computation of output j the input i is masked, meaning that no attention is used for this input.

    sz - sequence size
    n_samples - number of consecutive samples for each masked subsequences
    perc_masking - percentage of all masked samples
    """
    # number of total subsequences of length n_samples to be masked
    num_subseq = int(perc_masking * sz_cols // n_samples)

    # get indices to mask out
    # (better solution would be if each row in idx_ind would be sampled without replacement.
    # Current solution allows overlap of masked out subsequences.
    idx_ind = np.random.choice(sz_cols//n_samples, [sz_rows, num_subseq])
    idx_ind = n_samples * torch.tensor(idx_ind)
    idx = idx_ind
    for i in range(1, n_samples):
        idx = torch.cat([idx, idx_ind+i], 1)

    # mask the indices with infinity
    mask = torch.zeros(sz_rows, sz_cols)
    mask.scatter_(1, idx, float('-inf'))
    return mask
