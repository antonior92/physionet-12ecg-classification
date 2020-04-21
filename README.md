# physionet-12ecg-classification


To load this repository run:
```
git clone git@github.com:antonior92/physionet-12ecg-classification.git
# or: git clone https://github.com/antonior92/physionet-12ecg-classification.git
```
The requirements are described in `requirements.txt`.


## Downloading the dataset from PhysioNet
You can download the training data from this link: 
https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_CPSC.tar.gz

Alternatively, from the command line:
```
wget -O PhysioNetChallenge2020_Training_CPSC.tar.gz \
https://cloudypipeline.com:9555/api/download/physionet2020training/PhysioNetChallenge2020_Training_CPSC.tar.gz/
```
And, them, extract the data:
```
tar -xf PhysioNetChallenge2020_Training_CPSC.tar.gz 
```

## Training and evaluating

In order to train train the model use:

```
python train.py
```

By default it looks for the wfdb file (containing the training dataset) in `./Training_WFDB`. The option
``--input_folder PATH`` might be used to specify a different location. By default, does not use the GPU,
but the GPU usage can be ativated using the option `--cuda`. Call
```
python train.py --help
```
To get a complete list of the options.

Unless a the output folder is explicitly specified using the option `--folder`, the script 
will create a new folder ``./output_YYYY-MM-DD_HH_MM_SS_MMMMMM``, for which 
`YYYY-MM-DD_HH_MM_SS_MMMMMM` is the date and time the script was executed. All 
the script output is saved inside this folder. The internal structure of this folder is:
```
./output_YYYY-MM-DD_HH_MM_SS_MMMMMM
    config.json
    model.pth
    history.csv
    (final_model.pth)
```
where `config.json` contain the model hyperparameters and training configurations, `model.pth` contain
the weights of the model for which the best performance was attained, `history.csv` contain the 
performance per epoch during the training, `final_model.pth` contain the weights of the model 
at the last epoch of the training (not necessarily the one with the best validation performance).

## Pretrained model

The output of one successful training procedure is available and can be loaded using:
```
mkdir ./mdl
wget https://www.dropbox.com/s/t4hee1krodllkdn/config.json?dl=0 -O mdl/config.json
wget https://www.dropbox.com/s/rw0idd0da34tmr1/model.pth?dl=0 -O mdl/model.pth
wget https://www.dropbox.com/s/z8x7iawuiz1mers/history.csv?dl=0 -O mdl/history.csv
```
This should create the folder
```
./mdl
    config.json
    model.pth
    history.csv
```
Look at `run_12ECG_classifier.py` to see how this model might be loaded.

## Scripts from the challenge

There are two scripts that are provided by the challenge organizers `driver.py` and `evaluate_12ECG_score.py`.

The script `driver.py` can be used to obtain the model output in all entries of a given directory:
```
python driver.py input_directory output_directory
```
where `input_directory` is a directory for input data files and `utput_directory` is a directory for output
classification files. This script shoud populated the output director with file of the type:
```
#Record ID
 AF, I-AVB, LBBB, Normal, RBBB, PAC,  PVC,  STD, STE
  1,     1,    0,      0,    0,   0,   0,     0,   0
0.9,   0.6,  0.2,   0.05,  0.2, 0.35, 0.35, 0.1, 0.1
```
The PhysioNet/CinC 2020 webpage provides a training database with data files and 
a description of the contents and structure of these files.

This script is available from: https://github.com/physionetchallenges/python-classifier-2020


The script `evaluate_12ECG_score.py` is available in: https://github.com/physionetchallenges/evaluation-2020.
It can use the output from `driver.py` to assess the model performance according to different scores.
````
python evaluate_12ECG_score.py input_directory output_directory scores.csv
````
