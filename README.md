# StableDNAm
## Introduction

Any question is welcomed to be asked by issue and I will try my best to solve your problems.

## basic dictionary
You can change parameters in `configuration/config.py` to train models.

You can change model structure in `model/ClassificationDNAbert.py` to train models.

You can change training process and dataset process in `frame/ModelManager.py` and `frame/DataManager.py` .

Besides, dataset in paper  is also included in `data/DNA_MS`.

### pretrain model
You should download pretrain model from relevant github repository.

For example, if you want to use [DNAbert](https://github.com/jerryji1993/DNABERT), you need to put them into the pretrain folder and rename the relevant choice in the model.

### Usage

``python main/train.py``
