FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER antonior92@gmail.com
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

# Load model pretrained weights

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt
