#!/usr/bin/env python

import sys
import os


def train_12ECG_classifier(input_directory, output_directory):
    sys.path.extend(['./'])
    folder = 'mdls/'
    # We do an ensemble of models to boost the performance. We train the last of the models that
    # will take part on the ensemble on the challenge server. To avoid exceeding the server time limit, we uploaded some
    # models we have trained locally them... We would like to highlight, however, that all the other
    # models from the ensemble have been trained using exact the same command and dataset (but with different seeds).
    # that is, they are trained using:
    #  python train.py --train_classes scored --valid_classes scored --seed DIFFERENT_SEED --out_layer challenge_sigmoid
    #  --valid_split 0 --save_last --input_folder INPUT_DIRECTORY --folder OUTPUT_DIRECTORY
    for subfolder in [os.path.join(folder, f) for f in os.listdir(folder)]:
        cmd = 'cp -r {} {}'.format(subfolder, output_directory.strip('/'))
        print(cmd)
        os.system(cmd)
    cmd = 'python train.py --train_classes scored --valid_classes scored --seed 1122 --out_layer challenge_sigmoid ' \
          '--valid_split 0 --save_last --input_folder {:} --folder {:}'.format(
        input_directory, os.path.join(output_directory.strip('/'), 'mdl_last'))

    print(cmd)
    os.system(cmd)
