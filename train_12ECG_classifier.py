#!/usr/bin/env python

import sys
import os


def train_12ECG_classifier(input_directory, output_directory):
    sys.path.extend(['./'])
    folder = 'multiple/'
    for subfolder in [os.path.join(folder, f) for f in os.listdir(folder)]:
        cmd = 'cp -r {} {}'.format(subfolder, output_directory.strip('/'))
        print(cmd)
        os.system(cmd)
    cmd = 'python train.py --train_classes scored --valid_classes scored --seed 12 --out_layer challenge_sigmoid ' \
          '--valid_split 0 --save_last --input_folder {:} --folder {:}'.format(
        input_directory, os.path.join(output_directory.strip('/'), 'latest_model'))

    print(cmd)
    os.system(cmd)
