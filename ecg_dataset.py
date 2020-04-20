import numpy as np
import os, sys
from scipy.io import loadmat
from scipy.signal import decimate, resample_poly
import itertools

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
    n_samples = x.shape[0]
    new_x = np.zeros((n_samples, n_classes), dtype=bool)

    counter = 0
    for i, mask in enumerate(mututally_exclusive):
        for j, id in enumerate(mask):
            new_x[:, id] = x[:, i] == (j + 1)
            counter += 1
    new_x[:, counter:] = x[:, len(mututally_exclusive):]
    return new_x


def add_normal_column(x, prob=False):
    n_samples, n_classes = x.shape
    new_x = np.zeros((n_samples, n_classes + 1), dtype=x.dtype)
    new_x[:, :-1] = x[:, :]
    # If x is a vector of zeros and ones
    if not prob
        new_x[:, -1] = x.sum(axis=1) == 0

    # if x is a vector of probabilities
    else:
        counter = 0
        new_x[:, -1] = 1.0
        for m in mututally_exclusive:
            x[:, -1] *= 1 - x[:, m].sum(axis=1)
            counter += len(mask)
        x[:, -1] *= np.prod(1 - x[:, counter:], axis=1)

    return new_x


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


def get_sample(data, header_data, new_freq):
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

        output = np.zeros(n_target_vec)

        for l in labels:
            if l in CLASSES:
                output[class_to_idx[l]] = class_to_number[l]

        # Get header data
        tmp_hea = header_data[0].split(' ')
        id = int(tmp_hea[0].split('A')[1])
        num_leads = int(tmp_hea[1])
        freq = int(tmp_hea[2])
        gain_lead = np.zeros(num_leads)
        baseline = np.zeros(num_leads)
        for i in range(num_leads):
            tmp_hea = header_data[i + 1].split(' ')
            gain_lead[i] = int(tmp_hea[2].split('/mV')[0])
            baseline[i] = int(tmp_hea[4])
        # Change scale
        data_with_gain = (data - baseline[:, None]) / gain_lead[:, None]
        # Resample
        if freq != new_freq:
            data_with_gain = resample_ecg(data_with_gain, freq, new_freq)

        return {'id': id, 'data': data_with_gain, 'age': age, 'is_male': is_male, 'output': output}


class ECGDataset(object):
    def __init__(self, input_folder, freq=500):
        # Save input files and folder
        input_files = []
        for f in os.listdir(input_folder):
            if os.path.isfile(os.path.join(input_folder, f)) and not f.lower().startswith(
                    '.') and f.lower().endswith('mat'):
                input_files.append(f)
        self.input_file = input_files
        self.input_folder = input_folder
        self.freq = freq

    def get_ids(self):
        return [int(f.split('A')[-1].split('.mat')[0]) for f in self.input_file]

    def __len__(self):
        return len(self.input_file)

    def _getsample(self, idx):
        filename = os.path.join(self.input_folder, self.input_file[idx])

        x = loadmat(filename)
        data = np.asarray(x['val'], np.float32)

        # Get header data
        new_file = filename.replace('.mat', '.hea')
        input_header_file = os.path.join(new_file)
        with open(input_header_file, 'r') as f:
            header_data = f.readlines()

        return get_sample(data, header_data, self.freq)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._getsample(idx)
        elif isinstance(idx, slice):
            return [self._getsample(i) for i in itertools.islice(range(len(self)), idx.start, idx.stop, idx.step)]
        else:
            raise IndexError()


def split_long_signals(sample, length=4096, min_length=2000):
    idx, data, age, is_male, output = sample['id'], sample['data'], sample['age'], sample['is_male'],  sample['output']
    total_length = data.shape[1]
    # Get number of splits
    n_splits = total_length // length + (1 if total_length % length > min_length else 0)
    n_splits = max(n_splits, 1)
    # Define where to start picking samples
    offset = (total_length - n_splits * length) // 2 if n_splits * length < total_length else 0
    list_subsamples = []
    start_i = offset
    for ii in range(n_splits):
        x = np.zeros((data.shape[0], length))
        if total_length - start_i >= length:
            x[:, :length] = data[:, start_i:start_i + length]
            start_i += length
        else:
            actual_length = total_length - start_i
            pad = (length - actual_length) // 2
            x[:, pad:pad+actual_length] = data[:, start_i:start_i + actual_length]
            start_i += actual_length
        # Instanciate dict with data
        list_subsamples.append({'id': idx, 'split': ii, 'n_splits': n_splits, 'data': x,
                               'age': age, 'is_male': is_male, 'output': output})
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
    import matplotlib.pyplot as plt
    ecg_dataset = ECGDataset('./Training_WFDB', freq=400)
    len(ecg_dataset)

    jj = 1002
    plt.plot(ecg_dataset[jj]['data'][0, :])
    plt.show()
    print(ecg_dataset[jj]['data'].shape)
    for x in split_long_signals(ecg_dataset[jj]):
        print(x['data'].shape)
        print(x['split'])
        plt.plot(x['data'][0, :])
        plt.show()

    print(ecg_dataset.get_ids())

    samples = to_dict_of_lists(list(itertools.chain(*[split_long_signals(s) for s in ecg_dataset[:10]])))
    dataset = apply_to_all_dict_values(samples, np.stack)
    print(apply_to_all_dict_values(dataset, np.stack))