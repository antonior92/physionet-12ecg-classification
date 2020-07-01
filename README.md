# physionet-12ecg-classification


To load this repository run:
```
git clone git@github.com:antonior92/physionet-12ecg-classification.git
# or: git clone https://github.com/antonior92/physionet-12ecg-classification.git
```
The requirements are described in `requirements.txt`.


## Downloading the datasets from PhysioNet
The training data can be downloaded from this links (You can use the MD5 hash to verify the integrity of the tar.gz file.):
- CPSC2018 training set, 6,877 recordings: 
[link](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_CPSC.tar.gz); 
MD5-hash: 7b6b1f1ab1b4c59169c639d379575a87
- China 12-Lead ECG Challenge Database (unused CPSC2018 data), 3,453 recordings: 
[link](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_2.tar.gz);
 MD5-hash: 36b409ee2b46aa6f1d2bef99b8451925
- St Petersburg INCART 12-lead Arrhythmia Database, 74 recordings: 
[link](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_StPetersburg.tar.gz);
 MD5-hash: 440ca079f137fb16259511bb6105f134
- PTB Diagnostic ECG Database, 516 recordings: 
[link](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_PTB.tar.gz); 
MD5-hash: 4035a2b496067c4331eecab74695bc67
- PTB-XL electrocardiography Database, 21,837 recordings:
[link](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_PTB-XL.tar.gz);
 MD5-hash: a893319c53f77d8e6a76ed3af38be99e
- Georgia 12-Lead ECG Challenge Database, 10,344 recordings: 
[link](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_E.tar.gz);
 MD5-hash: 594c8cbc02a0aec4c179d2f019b09a7a
 
The data loading procedure used in `train.py` and `pretrain.py` can work with nested directories, so we recommend 
loading and extracting all the datasets into the same folder. Every thing can be done from the command line by:
- Create the new folder and move into it:
```bash
mkdir training_data  # Or whatever name...
cd ./training_data
```
- Load dataset (with appropriate names):
```
for DSET in Training_CPSC Training_2 Training_StPetersburg Training_PTB PTB-XL Training_E;
do
wget -O PhysioNetChallenge2020_$DSET.tar.gz \
https://cloudypipeline.com:9555/api/download/physionet2020training/PhysioNetChallenge2020_$DSET.tar.gz/
done;
mv PhysioNetChallenge2020_PTB-XL.tar.gz PhysioNetChallenge2020_Training_PTB-XL.tar.gz
```
- And, them, extract the data:
```
for DSET in Training_CPSC Training_2 Training_StPetersburg Training_PTB Training_PTB-XL Training_E;
do
mkdir ./$DSET && tar -xf PhysioNetChallenge2020_$DSET.tar.gz -C ./$DSET --strip-components=1
done;
```

## Training and evaluating

In order to train the model use:

```
python train.py
```
By default it looks for the WFDB folder (containing the training dataset) in `./Training_WFDB`. The option
``--input_folder PATH`` might be used to specify a different location. By default, does not use the GPU,
but the GPU usage can be ativated using the option `--cuda`. Call:
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

## Unsupervised pre-training

It is possible to use an unsupervised pre-training stage. Given a partial ECG signal, the model will be 
trained to predict unseen samples of the signal.  To generate a pre-trained model use:
```
python pretrain.py
```
By default it looks for the WFDB folder (containing the training dataset) in `./Training_WFDB`. The option
``--input_folder PATH`` might be used to specify a different location. By default, does not use the GPU,
but the GPU usage can be ativated using the option `--cuda`. Call:
```
python pretrain.py --help
```
To get a complete list of the options.

Unless a the output folder is explicitly specified using the option `--folder`, the script 
will create a new folder ``./output_YYYY-MM-DD_HH_MM_SS_MMMMMM``, for which 
`YYYY-MM-DD_HH_MM_SS_MMMMMM` is the date and time the script was executed. All 
the script output is saved inside this folder. The internal structure of this folder is:
```
./output_YYYY-MM-DD_HH_MM_SS_MMMMMM
    pretrain_config.json
    pretrain_model.pth
    pretrain_history.csv
    pretrain_train_ids.txt
    pretrain_valid_ids.txt
    (pretrain_final_model.pth)
```

where `pretrain_config.json` contain the model hyperparameters and training configurations, 
`pretrain_model.pth` contain the weights of the model for which the best performance (in the unsupervised task)
was attained, `pretrain_history.csv` contain the  performance per epoch during the training, 
`pretrain_final_model.pth` contain the weights of the model  at the last epoch of the training
 (not necessarily the one with the best validation performance). Also, ``pretrain_{train,valid}_ids.txt`` contain
 the ids used for training and validating the unsupervised model


One can load pre-trained weights during the supervised training stage, i.e., ``train.py`` by simply calling:
```
python train.py --folder /PATH/TO/FOLDER/WITH/PRETRAINED/MODEL
```
that is, with an option pointing to any folder containing (at least) 
``pretrain_config.json`` and `pretrain_model.pth`.

## Load pretrained weights and configuration

The output of one successful unsupervised pretraining + the supervised training procedure is available 
and can be loaded using:
```
mkdir ./mdl
wget https://www.dropbox.com/s/1pledtjboriw1fz/config.json?dl=0 -O mdl/config.json
wget https://www.dropbox.com/s/f940fomzmbxmbra/model.pth?dl=0 -O mdl/model.pth
wget https://www.dropbox.com/s/46ombyq4ecgl7oa/pretrain_config.json?dl=0 -O mdl/pretrain_config.json
wget https://www.dropbox.com/s/hv0mj3gwcm43u26/pretrain_model.pth?dl=0 -O mdl/pretrain_model.pth
wget https://www.dropbox.com/s/f08q2wk2wdwehza/history.csv?dl=0 -O mdl/history.csv
wget https://www.dropbox.com/s/3kjsfl8lyak6hau/pretrain_history.csv?dl=0 -O mdl/pretrain_history.csv
```
This should create the folder
```
./mdl
    config.json
    pretrain_config.json
    model.pth
    pretrain_model.pth
    history.csv
    pretrain_history.csv
    
```
Look at `run_12ECG_classifier.py` to see how this model might be loaded.

## Scripts from the challenge

There are two scripts that are provided by the challenge organizers `driver.py` and `evaluate_12ECG_score.py`.

The script `driver.py` can be used to obtain the model output in all entries of a given directory:
```
python driver.py input_directory output_directory
```
where `input_directory` is a directory for input data files and `output_directory` is a directory for output
classification files. This script should populated the output director with file of the type:
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
