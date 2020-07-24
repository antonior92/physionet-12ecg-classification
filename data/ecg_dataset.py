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
                if os.path.isfile(f) and not tail.lower().startswith('.') and tail.lower().endswith('mat'):
                    input_files.append(os.path.relpath(f, input_folder))
        return cls(input_files, input_folder, freq, only_header)

    def __init__(self, input_files, input_folder='./', freq=500, only_header=False):
        self.input_file = input_files
        self.input_folder = input_folder
        self.freq = freq
        self.only_header = only_header
        self.id_to_idx = OrderedDict(zip(self.get_ids(self.input_file), range(len(self))))

        if len(self) == 0:
            raise ValueError('Dataset is empty.')

    def use_only_header(self, only_header=True):
        self.only_header = only_header
        return self

    def get_ids(self, files=None):
        if files is None:
            files = self.input_file
        return [os.path.split(f)[1].split('.mat')[0] for f in files]

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

    def __iter__(self):
        return self[iter(self.get_ids())]

    def get_subdataset(self, ids):
        id_to_file = dict(zip(self.get_ids(self.input_file), self.input_file))
        files = [id_to_file[id] for id in ids]
        return ECGDataset(files, self.input_folder, self.freq, self.only_header)


if __name__ == '__main__':
    dset = ECGDataset.from_folder('training_data/Training_2')
    print(len(dset))
    cls = dset.get_classes()
    print(len(cls))
    print(len(dset.get_occurrences(cls)))

    sdset = dset.get_subdataset(['Q3090', 'Q3091'])
    print(len(sdset))
    scls = sdset.get_classes()
    print(scls)
    print(sdset.get_occurrences(scls))
    print(sdset[:])


