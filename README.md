# GDConvNet
This repo contains the official training and testing codes for our paper:

### Video Frame Interpolation via Generalized Deformable Convolution

Zhihao Shi, Xiaohong Liu, Kangdi Shi, Linhui Dai, Jun Chen

[[Paper](https://arxiv.org/abs/2008.10680)] 

## Prerequisites
- Python == 3.7.6 
- [Pytorch](https://pytorch.org/) == 1.2.0  
- Torchvision == 0.4.0
- Pillow == 7.1.2
- Numpy == 1.18.1

## Quick Start
### Dataset
[viemo_septuplet](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip) is used as our training and testing dataset. Please download and unzip it somewhere on your device. Then change the training and testing directory in ```./configs/config.py```.

### testing
We have released the model weight in ```./modeldict/```, you can directly use it to do the evaluation.
Using the command:
```bash
$ python3 eval.py
```
to start the testing process.

### training
You can also choose to retrain the model, just use the command:
```bash
$ python3 train_full_model.py
```
to start the training process.

