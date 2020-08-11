import numpy as np
import os
from scipy.io import loadmat
from scipy.signal import decimate, resample_poly
import itertools
import collections.abc as abc
from collections import OrderedDict


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
            try:
                tmp_age = iline.split(': ')[1].strip()
                age = int(tmp_age if (tmp_age != 'NaN') else 57)
                if age < 0 or age > 110:
                    age = 57
            except:
                age = 57
        elif iline.startswith('#Sex'):
            try:
                tmp_sex = iline.split(': ')[1]
                if tmp_sex.strip() == 'Female':
                    is_male = 0
                else:
                    is_male = 1
            except:
                is_male = 1
        elif iline.startswith('#Dx'):
            try:
                labels = iline.split(': ')[1].split(',')
            except:
                labels = []

    # Traces annotation
    tmp_hea = header_data[0].split(' ')
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

    return {'age': age, 'is_male': is_male, 'labels': labels,
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

    @classmethod
    def from_folder(cls, input_folder, freq=500, only_header=False):
        # Save input files and folder
        input_files = []
        for root, _, file in os.walk(input_folder):
            for ff in file:
                f = os.path.join(root, ff)
                head, tail = os.path.split(f)
                if os.path.isfile(f) and not tail.lower().startswith('.') and \
                        (tail.lower().endswith('mat') or tail.lower().endswith('dat')):
                    input_files.append(os.path.relpath(f, input_folder))
        return cls(input_files, input_folder, freq, only_header)

    def __init__(self, input_files, input_folder='./', freq=500, only_header=False):
        self.input_file = input_files
        self.input_folder = input_folder
        self.freq = freq
        self.only_header = only_header
        self.id_to_idx = OrderedDict(zip(self.get_ids(self.input_file), range(len(self))))

    def use_only_header(self, only_header=True):
        self.only_header = only_header
        return self

    def get_id(self, file):
        if '.dat' in file:
            return os.path.split(file)[1].split('.dat')[0]
        elif '.mat' in file:
            return os.path.split(file)[1].split('.mat')[0]
        else:
            raise ValueError('File should be either a .mat or a .dat')

    def get_header_filename(self, filename):
        if '.dat' in filename:
            return filename.replace('.dat', '.hea')
        elif '.mat' in filename:
            return filename.replace('.mat', '.hea')
        else:
            raise ValueError('File should be either a .mat or a .dat')

    def get_signal_from_path(self, path):
        if '.dat' in path:
            return np.fromfile(path, dtype='<i2').reshape(-1, 12).T
        elif '.mat' in path:
            return loadmat(path)['val']
        else:
            raise ValueError('File should be either a .mat or a .dat')

    def get_ids(self, files=None):
        if files is None:
            files = self.input_file
        return [self.get_id(f) for f in files]

    def get_classes(self):
        classes = set()
        for idx in range(len(self)):
            header = self._getsample(idx, only_header=True)
            for l in header['labels']:
                classes.add(l)
        return sorted(classes)

    def get_occurrences(self, classes):
        class_to_idx = OrderedDict(zip(classes, range(len(classes))))
        counts = np.zeros(len(classes), dtype=int)
        for idx in range(len(self)):
            header = self._getsample(idx, only_header=True)
            for l in header['labels']:
                try:
                    counts[class_to_idx[l]] += 1
                except KeyError:
                    pass
        return counts

    def __len__(self):
        return len(self.input_file)

    def _getsample(self, idx, only_header=False):
        file = self.input_file[idx]
        filename = os.path.join(self.input_folder, file)

        if only_header:
            data = None
        else:
            x = self.get_signal_from_path(filename)
            data = np.asarray(x, np.float32)
        # Get header data
        new_file = self.get_header_filename(filename)
        input_header_file = os.path.join(new_file)
        with open(input_header_file, 'r') as f:
            header_data = f.readlines()
        sample = get_sample(header_data, data, self.freq)
        sample['id'] = self.get_id(file)
        return sample

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
            raise IndexError('idx = {} ()'.format(idx, type(idx)))

    def __iter__(self):
        return self[iter(self.get_ids())]

    def get_subdataset(self, ids):
        id_to_file = dict(zip(self.get_ids(self.input_file), self.input_file))
        files = [id_to_file[id] for id in ids]
        return ECGDataset(files, self.input_folder, self.freq, self.only_header)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dset = ECGDataset.from_folder('training_data/Training_TNMG')
    x = dset._getsample(1)['data']
    plt.plot(x.T)
    plt.show()

    dset2 = ECGDataset.from_folder('training_data/Training_2')
    x = dset2._getsample(1)['data']
    plt.plot(x.T)
    plt.show()
