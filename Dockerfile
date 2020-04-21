FROM python:3.7.3-stretch

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER antonior92@gmail.com
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

# Load model pretrained weights
RUN mkdir ./mdl
RUN wget https://www.dropbox.com/s/t4hee1krodllkdn/config.json?dl=0 -O mdl/config.json
RUN wget https://www.dropbox.com/s/rw0idd0da34tmr1/model.pth?dl=0 -O mdl/model.pth
RUN wget https://www.dropbox.com/s/z8x7iawuiz1mers/history.csv?dl=0 -O mdl/history.csv

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt