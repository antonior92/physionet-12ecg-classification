FROM python:3.7.3-stretch

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER antonior92@gmail.com
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

# Load model pretrained weights
RUN mkdir ./mdl
RUN wget https://www.dropbox.com/s/1pledtjboriw1fz/config.json?dl=0 -O mdl/config.json
RUN wget https://www.dropbox.com/s/f940fomzmbxmbra/model.pth?dl=0 -O mdl/model.pth
RUN wget https://www.dropbox.com/s/46ombyq4ecgl7oa/pretrain_config.json?dl=0 -O mdl/pretrain_config.json

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt
https://www.dropbox.com/s/1pledtjboriw1fz/config.json?dl=0