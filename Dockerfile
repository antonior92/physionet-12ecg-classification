FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER antonior92@gmail.com
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

# Load model pretrained weights
RUN mkdir ./mdls
# load mdl1 from folder complete_training_2 on dropbox
RUN mkdir ./mdls/mdl1
RUN wget https://www.dropbox.com/s/y3j7048wcj9jksg/config.json?dl=0 -O mdls/mdl1/config.json
RUN wget https://www.dropbox.com/s/47g3la9gjcvw74j/correction_factor.txt?dl=0 -O mdls/mdl1/correction_factor.txt
RUN wget https://www.dropbox.com/s/mal9mvlt3t3ry9g/history.csv?dl=0 -O mdls/mdl1/history.csv
RUN wget https://www.dropbox.com/s/if4qh47p70l4hv2/model.pth?dl=0 -O mdls/mdl1/model.pth
RUN wget https://www.dropbox.com/s/nvb1tpeq3zxyqmg/out_layer.txt?dl=0 -O mdls/mdl1/out_layer.txt
RUN wget https://www.dropbox.com/s/fvky7p6ca9ipeo1/train_ids.txt?dl=0 -O mdls/mdl1/train_ids.txt

#  load mdl2 from folder complete_training_3 on dropbox
RUN mkdir ./mdls/mdl2
RUN wget https://www.dropbox.com/s/x734tg6zw41rwau/config.json?dl=0 -O mdls/mdl2/config.json
RUN wget https://www.dropbox.com/s/euaeg5ti1547wy0/correction_factor.txt?dl=0 -O mdls/mdl2/correction_factor.txt
RUN wget https://www.dropbox.com/s/911e806lnfzw7v2/history.csv?dl=0 -O mdls/mdl2/history.csv
RUN wget https://www.dropbox.com/s/4s165dijaf70evo/model.pth?dl=0 -O mdls/mdl2/model.pth
RUN wget https://www.dropbox.com/s/el2rcaxqrss7ppw/out_layer.txt?dl=0 -O mdls/mdl2/out_layer.txt
RUN wget https://www.dropbox.com/s/ty5eo7qu6n756pc/train_ids.txt?dl=0 -O mdls/mdl2/train_ids.txt

# load mdl3 from folder complete_training_4 on dropbox
RUN mkdir ./mdls/mdl3
RUN wget https://www.dropbox.com/s/466huxwf19l941l/config.json?dl=0 -O mdls/mdl3/config.json
RUN wget https://www.dropbox.com/s/c39c2es4szhrjtg/correction_factor.txt?dl=0 -O mdls/mdl3/correction_factor.txt
RUN wget https://www.dropbox.com/s/5c9f3x5mqexdgmg/history.csv?dl=0 -O mdls/mdl3/history.csv
RUN wget https://www.dropbox.com/s/e5axa65s32on551/model.pth?dl=0 -O mdls/mdl3/model.pth
RUN wget https://www.dropbox.com/s/xspq1wls3cpeke0/out_layer.txt?dl=0 -O mdls/mdl3/out_layer.txt
RUN wget https://www.dropbox.com/s/p2puyr814lv69pl/train_ids.txt?dl=0 -O mdls/mdl3/train_ids.txt

# load mdl4 from folder complete_mdl5 on dropbox
RUN mkdir ./mdls/mdl4
RUN wget https://www.dropbox.com/s/nt45r982qu7jodl/config.json?dl=0 -O mdls/mdl4/config.json
RUN wget https://www.dropbox.com/s/aqzz3jzl4funp9f/correction_factor.txt?dl=0 -O mdls/mdl4/correction_factor.txt
RUN wget https://www.dropbox.com/s/je51bak370k4kcn/history.csv?dl=0 -O mdls/mdl4/history.csv
RUN wget https://www.dropbox.com/s/b21xtora0ym8uuh/model.pth?dl=0 -O mdls/mdl4/model.pth
RUN wget https://www.dropbox.com/s/6d0d356dmbe7rxn/out_layer.txt?dl=0 -O mdls/mdl4/out_layer.txt
RUN wget https://www.dropbox.com/s/jjx87zicrwbz1kg/train_ids.txt?dl=0 -O mdls/mdl4/train_ids.txt

# load mdl5 from folder complete_mdl6 on dropbox
RUN mkdir ./mdls/mdl5
RUN wget https://www.dropbox.com/s/f0u6vonuqnpyno7/config.json?dl=0 -O mdls/mdl5/config.json
RUN wget https://www.dropbox.com/s/8xufnxjxmiutr3u/correction_factor.txt?dl=0 -O mdls/mdl5/correction_factor.txt
RUN wget https://www.dropbox.com/s/4o2o69bockngeug/history.csv?dl=0 -O mdls/mdl5/history.csv
RUN wget https://www.dropbox.com/s/mojuok2r78cd7go/model.pth?dl=0 -O mdls/mdl5/model.pth
RUN wget https://www.dropbox.com/s/o41yh73e38zwbuj/out_layer.txt?dl=0 -O mdls/mdl5/out_layer.txt
RUN wget https://www.dropbox.com/s/abnxjx8l4czr63u/train_ids.txt?dl=0 -O mdls/mdl5/train_ids.txt

# load mdl6 from folder full_mdl5 on dropbox
RUN mkdir ./mdls/mdl6
RUN wget https://www.dropbox.com/s/fym210f6qiz53oa/config.json?dl=0 -O mdls/mdl6/config.json
RUN wget https://www.dropbox.com/s/m3jkn5e99v02mdk/correction_factor.txt?dl=0 -O mdls/mdl6/correction_factor.txt
RUN wget https://www.dropbox.com/s/nzlodzkdcnsghu0/history.csv?dl=0 -O mdls/mdl6/history.csv
RUN wget https://www.dropbox.com/s/5qtjc58wcxbtvi6/model.pth?dl=0 -O mdls/mdl6/model.pth
RUN wget https://www.dropbox.com/s/9zpn1bsdckcp23j/out_layer.txt?dl=0 -O mdls/mdl6/out_layer.txt
RUN wget https://www.dropbox.com/s/tabl52rkk6myw5w/train_ids.txt?dl=0 -O mdls/mdl6/train_ids.txt

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt
