import itertools
from collections import abc as abc
from collections import OrderedDict
import random

import numpy as np
import torch
from binpacking import to_constant_volume


class SplitLongSignals(object):
    def __init__(self, sample, length=4096, min_length=2000):
        # Get data
        total_length = sample['signal_len']
        # Get number of splits
        n_splits = total_length // length + (1 if total_length % length > min_length else 0)
        n_splits = max(n_splits, 1)
        self.n_splits = n_splits
        # Get variables
        self.offset = (total_length - n_splits * length) // 2 if n_splits * length < total_length else 0
        self.start_i = self.offset
        self.total_length = total_length
        self.sample = sample
        self.length = length
        self.min_length = min_length
        self.sub_id = 0

    def __len__(self):
        return self.n_splits

    def __iter__(self):
        self.start_i = self.offset
        self.sub_id = 0
        return self

    def __next__(self):
        if self.sub_id >= self.n_splits:
            raise StopIteration
        self.sub_id += 1
        subsample = {k: v for k, v in self.sample.items() if k != 'data'}
        subsample['signal_len'] = self.length
        subsample['sub_id'] = self.sub_id
        actual_length = min(self.total_length - self.start_i, self.length)
        if 'data' in self.sample.keys():
            x = np.zeros((self.sample['data'].shape[0], self.length))
            if self.total_length - self.start_i >= self.length:
                x[:, :self.length] = self.sample['data'][:, self.start_i:self.start_i + self.length]
            else:
                pad = (self.length - actual_length) // 2
                x[:, pad:pad + actual_length] = self.sample['data'][:, self.start_i:self.start_i + actual_length]
            subsample['data'] = x
        self.start_i += actual_length
        return subsample



class LoI2IoL(abc.Iterator):
    """List of iterable to iterator of lists"""
    def __init__(self, list_of_iterable, stop_at='last'):
        self.list_of_iterators = [iter(it) for it in list_of_iterable]
        self.stop_at = len(list_of_iterable) if stop_at == 'last' else stop_at
        self.iter_ended = np.zeros(len(list_of_iterable), dtype=bool)

    def __next__(self):
        new_list = []
        for i, l in enumerate(self.list_of_iterators):
            try:
                new_list.append(next(l))
            except StopIteration:
                self.iter_ended[i] = True
                if sum(self.iter_ended) >= self.stop_at:
                    raise StopIteration()
        return new_list


def get_batches(batch_size, ids, counts, drop_last=False):
    # TODO: deal with case the count is larger than `min_n_batches`.
    # TODO: Check if drop last work for the special cases
    # Get minumum number of batches
    min_n_batches = sum(counts) // batch_size
    # Compute dicts with constante voulume
    # Solve a version of "bin packing problem"(https://en.wikipedia.org/wiki/Bin_packing_problem)
    # to sub optimally distribute samples among batches
    dict_ids_counts = OrderedDict(zip(list(ids), list(counts)))
    constant_volume_dicts = to_constant_volume(dict_ids_counts, min_n_batches)
    # Get ids from dicts
    all_ids = [[id for id in ith_ids.keys()] for ith_ids in constant_volume_dicts]
    # Get ids for each of the batches
    batches = all_ids[:batch_size]
    # Distribute remainders
    if not drop_last:
        for i, id in enumerate(itertools.chain(*all_ids[batch_size:])):
            batches[i % batch_size].append(id)
    # Compute total number of samples
    if drop_last:
        n_samples = sum([sum(ith_volume.values())for ith_volume in constant_volume_dicts][:batch_size])
    else:
        n_samples = sum([sum(ith_volume.values())for ith_volume in constant_volume_dicts])
    return batches, n_samples


class ECGBatchloader(abc.Iterable):
    def __init__(self, dset, dx=None, batch_size=32, length=4096, min_length=None, drop_last=False,
                 seed=0):
        self.dset = dset
        self.ids = dset.get_ids()
        if min_length is None:
            min_length = length // 2
        self.transformation = lambda s: SplitLongSignals(s, length, min_length)
        dset.use_only_header(True)
        counts = [len(self.transformation(s)) for s in dset]
        dset.use_only_header(False)

        def collapsing_fn(batch):
            traces = torch.stack([torch.tensor(s['data'], dtype=torch.float32) for s in batch], dim=0)
            if dx is not None:
                target = torch.stack(
                    [torch.tensor(dx.target_from_labels(s['labels']), dtype=torch.long) for s in batch], dim=0)
            ids = [s['id'] for s in batch]
            sub_ids = [s['sub_id'] for s in batch]
            if dx is not None:
                return (traces, target, ids, sub_ids)
            else:
                return (traces, ids, sub_ids)

        self.collapsing_fn = collapsing_fn
        self.counts = counts
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = random.Random(seed)

    def __iter__(self):
        batches, l = get_batches(self.batch_size, self.ids, self.counts, self.drop_last)
        modified_dset = [itertools.chain.from_iterable(map(self.transformation, self.dset[iter(b)])) for b in batches]
        x = LoI2IoL(modified_dset)
        batch_loader = map(self.collapsing_fn, x)
        return batch_loader

    def __len__(self):
        batches, l = get_batches(self.batch_size, self.ids, self.counts, self.drop_last)
        return l

    def shuffle(self):
        idx = list(range(len(self.ids)))
        self.rng.shuffle(idx)
        self.ids = [self.ids[i] for i in idx]
        self.counts = [self.counts[i] for i in idx]