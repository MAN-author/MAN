# MAN
code for paper:"Multi-pretext Attention Network for Few-shot Learning with Self-supervision"

## Installation

```
conda version:4.6.14
python:3.7.4
pytorch:1.3.0
cuda:10.2
```
```
pip install -e .
```
## Dataset
### MiniImagenet 
To download the MiniImagenet dataset go to https://github.com/gidariss/FewShotWithoutForgetting and follow the instructions there. Then, set in man/datasets/mini_imagenet_dataset.py the path to where the dataset resides in your machine.

## Train
```
sh run_miniImageNet/WRN-train.sh
```


## Test
```
sh run_miniImageNet/WRN-test.sh
```
