import numpy as np
import os, sys
from scipy.io import loadmat
from scipy.signal import decimate, resample_poly
import itertools
import collections.abc as abc

CLASSES = ['AF', 'I-AVB', 'RBBB', 'LBBB', 'PAC', 'PVC', 'STD', 'STE']
mututally_exclusive = [
    [0, 1],  # AF and I-AVB
    [2, 3]   # RBBB and LBBB
]
n_classes = len(CLASSES)
n_target_vec = len(CLASSES) - len(mututally_exclusive)
class_to_idx = {'AF': 0, 'I-AVB': 0, 'RBBB': 1, 'LBBB': 1, 'PAC': 2, 'PVC': 3, 'STD': 4, 'STE': 5}
class_to_number = {'AF': 1, 'I-AVB': 2, 'RBBB': 1, 'LBBB': 2, 'PAC': 1, 'PVC': 1, 'STD': 1, 'STE': 1}


def multiclass_to_binaryclass(x):
    x = np.atleast_2d(x)
    n_samples = x.shape[0]
    new_x = np.zeros((n_samples, n_classes), dtype=bool)

    counter = 0
    for i, mask in enumerate(mututally_exclusive):
        for j, id in enumerate(mask):
            new_x[:, id] = x[:, i] == (j + 1)
            counter += 1
    new_x[:, counter:] = x[:, len(mututally_exclusive):]
    return np.squeeze(new_x)


def add_normal_column(x, prob=False):
    x = np.atleast_2d(x)
    n_samples, n_classes = x.shape
    new_x = np.zeros((n_samples, n_classes + 1), dtype=x.dtype)
    new_x[:, :-1] = x[:, :]
    # If x is a vector of zeros and ones
    if not prob:
        new_x[:, -1] = x.sum(axis=1) == 0
    # if x is a vector of probabilities
    else:
        counter = 0
        new_x[:, -1] = 1.0
        for mask in mututally_exclusive:
            new_x[:, -1] = x[:, -1]*(1 - x[:, mask].sum(axis=1))
            counter += len(mask)
        x[:, -1] = x[:, -1]*np.prod(1 - x[:, counter:], axis=1)

    return np.squeeze(new_x)


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

    # labels to vector
    target = np.zeros(n_target_vec)
    for l in labels:
        if l in CLASSES:
            target[class_to_idx[l]] = class_to_number[l]

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

    return {'id': id, 'age': age, 'is_male': is_male, 'output': target,
            'baseline': baseline, 'gain_lead': gain_lead, 'freq': freq, 'signal_len': signal_len}


def get_sample(header_data, data=None, new_freq=None):
    # Read header
    attributes = read_header(header_data)
    # Get data
    if data is not None:
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
    def __init__(self, input_folder, freq=500, only_header=False):
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

    def use_only_header(self):
        self.only_header = True
        return self

    def get_ids(self):
        return [f.split('.mat')[0] for f in self.input_file]

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
        return get_sample(header_data, data, self.freq)

    def __getitem__(self, idx):
        if isinstance(idx, int):
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
            raise IndexError()


def split_long_signals(sample, length=4096, min_length=2000):
    # Get data
    if 'data' in sample.keys():
        data = sample['data']
        total_length = data.shape[1]
        print(total_length, data.shape[1])
    else:
        total_length = sample['signal_len']
    # Get number of splits
    n_splits = total_length // length + (1 if total_length % length > min_length else 0)
    n_splits = max(n_splits, 1)
    # Define where to start picking samples
    offset = (total_length - n_splits * length) // 2 if n_splits * length < total_length else 0
    list_subsamples = []
    start_i = offset
    for ii in range(n_splits):
        subsample = {k: v for k, v in sample.items() if k != 'data'}
        subsample['signal_len'] = length
        if 'data' in sample.keys():
            x = np.zeros((data.shape[0], length))
            if total_length - start_i >= length:
                x[:, :length] = data[:, start_i:start_i + length]
                start_i += length
            else:
                actual_length = total_length - start_i
                pad = (length - actual_length) // 2
                x[:, pad:pad+actual_length] = data[:, start_i:start_i + actual_length]
                start_i += actual_length

            subsample['data'] = x
        # Instanciate dict with data
        list_subsamples.append(subsample)
    return list_subsamples


def to_dict_of_lists(list_of_dicts):
    """Convert list of dict to dict of list"""
    keys = list_of_dicts[0].keys()
    dict_of_lists = {k: [] for k in keys}
    for d in list_of_dicts:
        for k, v in d.items():
            dict_of_lists[k].append(v)
    return dict_of_lists


def apply_to_all_dict_values(d, fn):
    """Apply function to all dict values"""
    return {k: fn(v) for k, v in d.items()}


if __name__ == '__main__':
    # Print classes
    ecg_dataset = ECGDataset('./Training_WFDB', freq=400, only_header=True)
    dset = to_dict_of_lists(ecg_dataset)
    outputs = np.stack(dset['output'])
    outputs_b = multiclass_to_binaryclass(outputs)
    print(outputs_b.sum(axis=0)/6877)

    # Plot example
    import matplotlib.pyplot as plt
    ecg_dataset = ECGDataset('./Training_WFDB', freq=400)
    len(ecg_dataset)

    jj = 1002
    for x in split_long_signals(ecg_dataset[jj]):
        plt.plot(x['data'][0, :])
        plt.show()
    print(ecg_dataset.get_ids())

    samples = to_dict_of_lists(list(itertools.chain(*[split_long_signals(s) for s in ecg_dataset[:10]])))
    dataset = apply_to_all_dict_values(samples, np.stack)
    print(apply_to_all_dict_values(dataset, np.stack))