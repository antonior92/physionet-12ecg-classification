#!/usr/bin/env python

import sys
import os


def train_12ECG_classifier(input_directory, output_directory):
    sys.path.extend(['./'])
    cmd = 'python train.py --batch_size 4 --train_classes scored --valid_classes scored --input_folder ' \
          '{:} --n_total 10 --seed 3 --folder {:}'.format(input_directory, output_directory)
    print(cmd)
    os.system(cmd)
