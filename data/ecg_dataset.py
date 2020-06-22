import numpy as np
import os
import torch
from scipy.io import loadmat
from scipy.signal import decimate, resample_poly
import itertools
import collections.abc as abc
from binpacking import to_constant_volume

from output_layer import DxClasses


def resample_ecg(trace, input_freq, output_freq):
    trace = np.atleast_1d(trace).astype(float)
    if input_freq != int(input_freq):
        raise ValueError("input_freq must be an integer")
    if output_freq != int(output_freq):
        raise ValueError("output_freq must be an integer")

    if input_freq == output_freq:
        new_trace = trace
    elif np.mod(input_freq, output_freq) == 0:
        new_trace = decimate(trace, q=input_freq//output_freq,
                             ftype='iir', zero_phase=True, axis=-1)
    else:
        new_trace = resample_poly(trace, up=output_freq, down=input_freq, axis=-1)
    return new_trace


def read_header(header_data):
    # Get attributes
    age = 57
    is_male = 1
    labels = []
    for iline in header_data:
        # Remove \n
        iline = iline.split("\n")[0]
        # Get age sex and label
        if iline.startswith('#Age'):
            tmp_age = iline.split(': ')[1].strip()
            age = int(tmp_age if (tmp_age != 'NaN') else 57)
            if age < 0 or age > 110:
                age = 57
        elif iline.startswith('#Sex'):
            tmp_sex = iline.split(': ')[1]
            if tmp_sex.strip() == 'Female':
                is_male = 0
            else:
                is_male = 1
        elif iline.startswith('#Dx'):
            labels = iline.split(': ')[1].split(',')

    # Traces annotation
    tmp_hea = header_data[0].split(' ')
    # id
    try:
        id = tmp_hea[0]
    except:
        id = 0
    # num leads and freq
    try:
        signal_len = int(tmp_hea[3])
        num_leads = int(tmp_hea[1])
        freq = int(tmp_hea[2])
    except:
        signal_len = 5120
        num_leads = 12
        freq = 500
    gain_lead = 1000*np.ones(num_leads)
    baseline = np.zeros(num_leads)
    for i in range(num_leads):
        tmp_hea = header_data[i + 1].split(' ')
        try:
            gain_lead[i] = int(tmp_hea[2].split('/mV')[0])
            baseline[i] = int(tmp_hea[4])
        except:
            pass

    return {'id': id, 'age': age, 'is_male': is_male, 'labels': labels,
            'baseline': baseline, 'gain_lead': gain_lead, 'freq': freq, 'signal_len': signal_len}


def get_sample(header_data, dx, data=None, new_freq=None):
    # Read header
    attributes = read_header(header_data)
    # Get data
    if data is not None:
        # Get target vector
        attributes['output'] = dx.get_target_from_labels(attributes['labels'])
        # Change scale
        data_with_gain = (data - attributes['baseline'][:, None]) / attributes['gain_lead'][:, None]
        # Resample
        if attributes['freq'] != new_freq:
            data_with_gain = resample_ecg(data_with_gain, attributes['freq'], new_freq)
        # Add data to attribute
        attributes['data'] = data_with_gain
    # Compute new signal len
    if attributes['freq'] != new_freq:
        attributes['signal_len'] = int(np.floor(attributes['signal_len'] * new_freq / attributes['freq']))
    return attributes


class ECGDataset(abc.Sequence):
    def __init__(self, input_folder, dx_classes, freq=500, only_header=False):
        # Save input files and folder
        input_files = []
        for f in os.listdir(input_folder):
            if os.path.isfile(os.path.join(input_folder, f)) and not f.lower().startswith(
                    '.') and f.lower().endswith('mat'):
                input_files.append(f)
        self.input_file = input_files
        self.input_folder = input_folder
        self.freq = freq
        self.only_header = only_header
        self.id_to_idx = dict(zip(self.get_ids(), range(len(self))))
        self.dx = dx_classes

    def use_only_header(self, only_header=True):
        self.only_header = only_header
        return self

    def get_ids(self):
        return [f.split('.mat')[0] for f in self.input_file]

    def get_classes(self):
        classes = set()
        for idx in range(len(self)):
            header = self._getsample(idx, only_header=True)
            for l in header['labels']:
                classes.add(l)
        return sorted(classes)

    def __len__(self):
        return len(self.input_file)

    def _getsample(self, idx, only_header=False):
        filename = os.path.join(self.input_folder, self.input_file[idx])

        if only_header:
            data = None
        else:
            x = loadmat(filename)
            data = np.asarray(x['val'], np.float32)
        # Get header data
        new_file = filename.replace('.mat', '.hea')
        input_header_file = os.path.join(new_file)
        with open(input_header_file, 'r') as f:
            header_data = f.readlines()
        return get_sample(header_data, self.dx, data, self.freq)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.int, np.int64)):
            return self._getsample(idx, self.only_header)
        if isinstance(idx, str):
            return self._getsample(self.id_to_idx[idx], self.only_header)
        elif isinstance(idx, slice):
            return list(self[itertools.islice(range(len(self)), idx.start, idx.stop, idx.step)])
        elif isinstance(idx, abc.Sequence):
            return list(self[iter(idx)])
        elif isinstance(idx, abc.Iterator):
            def my_iterator():
                for i in idx:
                    yield self[i]
            return my_iterator()
        elif isinstance(idx, abc.Iterable):
            return self[iter(idx)]
        else:
            print(idx, type(idx))
            raise IndexError('idx = {} ()'.format(idx, type(idx)))


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
        subsample = {k: v for k, v in self.sample.items() if k != 'data'}
        subsample['signal_len'] = self.length
        subsample['sub_id'] = self.sub_id
        if self.start_i + self.min_length > self.total_length:
            raise StopIteration
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
        self.sub_id += 1
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


def get_batches(batch_size, ids, counts):
    # TODO: deal with case the count is larger than `min_n_batches`.
    # Get minumum number of batches
    min_n_batches = sum(counts) // batch_size
    # Compute dicts with constante voulume
    # Solve a version of "bin packing problem"(https://en.wikipedia.org/wiki/Bin_packing_problem)
    # to sub optimally distribute samples among batches
    dict_ids_counts = dict(zip(list(ids), list(counts)))
    constant_volume_dicts = to_constant_volume(dict_ids_counts, min_n_batches)
    # Get ids from dicts
    all_ids = [[id for id in ith_ids] for ith_ids in constant_volume_dicts]
    # Get ids for each of the batches
    batches = all_ids[:batch_size]
    # Distribute remainders
    for i, id in enumerate(itertools.chain(*all_ids[batch_size:])):
        batches[i % batch_size].append(id)
    return batches


def collapsing_fn(batch):
    traces = torch.stack([torch.tensor(s['data'], dtype=torch.float32) for s in batch], dim=0)
    target = torch.stack([torch.tensor(s['output'], dtype=torch.long) for s in batch], dim=0)
    ids = [s['id'] for s in batch]
    sub_ids = [s['sub_id'] for s in batch]
    return (traces, target, ids, sub_ids)


def get_batchloader(dset, ids, batch_size=32, length=4096, min_length=2000):
    transformation = lambda s: SplitLongSignals(s, length,  min_length)
    dset.use_only_header(True)
    counts = [len(transformation(s)) for s in dset[ids]]
    dset.use_only_header(False)
    batches = get_batches(batch_size, ids, counts)
    modified_dset = [itertools.chain.from_iterable(map(transformation, dset[iter(b)])) for b in batches]
    x = LoI2IoL(modified_dset)
    batch_loader = map(collapsing_fn, x)
    return list(batch_loader)