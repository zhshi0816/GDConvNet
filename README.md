# GDConvNet
This repo contains the official training and testing codes for our paper:

### Video Frame Interpolation via Generalized Deformable Convolution

[Zhihao Shi](https://www.linkedin.com/in/zhihaoshi/), [Xiaohong Liu](https://xiaohongliu.ca/), [Kangdi Shi](https://www.linkedin.com/in/kangdi-shi-5ab37a147/), [Linhui Dai](https://charlie0215.github.io/), [Jun Chen](https://www.ece.mcmaster.ca/~junchen/)

Accepted in IEEE Transactions on Multimedia, 2021.

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

### Testing
We have released the model weight in ```./modeldict/```, you can directly use it to do the evaluation.
Using the command:
```bash
$ python3 eval.py
```
to start the testing process.

### Training
You can also choose to retrain the model, just use the command:
```bash
$ python3 train_full_model.py
```
to start the training process.

## Cite
If you use this code, please kindly cite
```
@article{shi2020video,
  title={Video Interpolation via Generalized Deformable Convolution},
  author={Shi, Zhihao and Liu, Xiaohong and Shi, Kangdi and Dai, Linhui and Chen, Jun},
  journal={arXiv preprint arXiv:2008.10680},
  year={2020}
}
```

