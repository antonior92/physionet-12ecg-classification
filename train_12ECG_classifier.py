#!/usr/bin/env python

import sys
import os


def train_12ECG_classifier(input_directory, output_directory):
    sys.path.extend(['./'])
    cmd = 'python train.py --train_classes scored --valid_classes scored --seed 3 --out_layer sigmoid ' \
          '--valid_split 0 --save_last --input_folder {:} --folder {:}'.format(input_directory, output_directory)
    print(cmd)
    os.system(cmd)
