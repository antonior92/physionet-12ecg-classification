import numpy as np
import os, sys
from scipy.io import loadmat
from scipy.signal import decimate, resample_poly

CLASSES = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE']


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
                             ftype='iir', zero_phase=True)
    else:
        new_trace = resample_poly(trace, up=output_freq, down=input_freq)
    return new_trace


class ECGDataset(object):
    def __init__(self, input_folder, classes=CLASSES, freq=500):
        # Save classes
        self.classes = classes
        self.class_to_idx = dict(zip(classes, range(len(classes))))
        # Save input files and folder
        input_files = []
        for f in os.listdir(input_folder):
            if os.path.isfile(os.path.join(input_folder, f)) and not f.lower().startswith(
                    '.') and f.lower().endswith('mat'):
                input_files.append(f)
        self.input_file = input_files
        self.input_folder = input_folder
        self.freq = freq

    def __len__(self):
        return len(self.input_file)

    def __getitem__(self, idx):
        filename = os.path.join(self.input_folder, self.input_file[idx])

        x = loadmat(filename)
        data = np.asarray(x['val'], np.float32).T

        # Get header data
        new_file = filename.replace('.mat', '.hea')
        input_header_file = os.path.join(new_file)
        with open(input_header_file, 'r') as f:
            header_data = f.readlines()

        # Get classes
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

        output = np.zeros(len(self.classes))

        for l in labels:
            output[self.class_to_idx[l]] = 1

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
        data_with_gain = (data - baseline[None, :]) / gain_lead[None, :]
        # Resample
        if freq != self.freq:
            data_with_gain = resample_ecg(data_with_gain, freq, self.freq)

        return {'id': id, 'data': data_with_gain, 'age': age, 'is_male': is_male, 'output': output}


def split_long_signals(sample, length=4096, min_length=2000):
    idx, data, age, is_male, output = sample['id'], sample['data'], sample['age'], sample['is_male'],  sample['output']
    total_length = data.shape[0]
    # Get number of splits
    n_splits = total_length // length + (1 if total_length % length > min_length else 0)
    n_splits = max(n_splits, 1)
    # Define where to start picking samples
    offset = (total_length - n_splits * length) // 2 if n_splits * length < total_length else 0
    list_subsamples = []
    start_i = offset
    for ii in range(n_splits):
        x = np.zeros((length, data.shape[1]))
        if total_length - start_i >= length:
            x[:length, :] = data[start_i:start_i + length, :]
            start_i += length
        else:
            actual_length = total_length - start_i
            pad = (length - actual_length) // 2
            x[pad:pad+actual_length] = data[start_i:start_i + actual_length, :]
            start_i += actual_length
        # Instanciate dicst with data
        list_subsamples.append({'id': idx, 'split': ii, 'n_splits': n_splits, 'data': x,
                               'age': age, 'is_male': is_male, 'output': output})
    return list_subsamples


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ecg_dataset = ECGDataset('./Training_WFDB', freq=400)
    len(ecg_dataset)

    jj = 1002
    plt.plot(ecg_dataset[jj]['data'][:, 0])
    plt.show()
    print(ecg_dataset[jj]['data'].shape)
    for x in split_long_signals(ecg_dataset[jj]):
        print(x['data'].shape)
        print(x['split'])
        plt.plot(x['data'][:, 0])
        plt.show()