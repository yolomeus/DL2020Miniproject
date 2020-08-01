# DL2020Miniproject: Inductive Node Classification using Attention
This project explores the extension of the Graph Attention Network, to attend over higher order neighbours. You can find
a detailed project report in `./report` [link](https://github.com/yolomeus/DL2020Miniproject/blob/master/report/DL_Mini_Project.pdf).
## Requirements
The code has been tested with **Python 3.7**. All requirements are specified in `environment.yml`. If you're using 
[anaconda](https://www.anaconda.com/) you can easily create an environment that meets the requirements by running:

```shell script
$ conda env create -f environment.yml
```
Otherwise you can also install the dependencies manually using `pip3`. If you're training on cpu exclusively, 
`cudatoolkit` is not required. 

## Training
If you want reproduce the experiments on cora, first download the dataset from 
[this](https://github.com/Diego999/pyGAT/tree/master/data/cora.) github page, and place it in `./data/cora`.
Then run:
```shell script
$ cd preprocessing
$ python export_cora.py 
```

Then you can start the training by running:
 ```shell script
$ cd ..
$ python train.py 
```   

For setting hyperparameters see below.

## Project structure

The project uses [hydra](https://hydra.cc/) for managing configuration, including pre-processing and training 
hyperparameters. It allows for a modular configuration composed of individual configuration files which can also be 
overwritten via command line arguments. 

You can find global settings regarding training and testing in `conf/config.yaml`. For model specific configuration, a 
yaml file is placed in `conf/model/` for each model.
Settings can also be passed as command line arguments e.g. :
```shell script
$ python train.py training.epochs=1000 loss.class=torch.nn.CrossEntropyLoss
```