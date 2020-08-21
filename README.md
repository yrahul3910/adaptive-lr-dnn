# Code for "A novel adaptive learning rate scheduler for deep neural networks"  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-novel-adaptive-learning-rate-scheduler-for/handwritten-digit-recognition-on-mnist)](https://paperswithcode.com/sota/handwritten-digit-recognition-on-mnist?p=a-novel-adaptive-learning-rate-scheduler-for)

This repository contains the code for the paper
> Yedida, Rahul, and Snehanshu Saha. "A novel adaptive learning rate scheduler for deep neural networks." arXiv preprint arXiv:1902.07399 (2019).

The main function is `lr_schedule`, that computes the adaptive learning rate for a given dataset and neural network. This method can be used with Keras' `LRScheduler` callback, as used in all the code files.

All results and code from the paper can be found here. Trained models and program outputs are also uploaded here. The top-level directories in this repository correspond to the dataset the experiments were run on. The `paper` directory contains the compiled PDF. The code uses the Keras deep learning library.

## Directory Structure
At the cost of duplicate code, every directory has independently executable code, i.e., to run any experiment, only the files in that directory are required.

The `Unconstrained` or `Nonconstrained` directories under the DenseNet architecture are experiments that we re-did to make the comparison more fair. The only difference is that `nb_filter` is set to -1 in these experiments, as with all the other DenseNet experiments.

The directories are organized by dataset, algorithm, and other options, such as weight decay value, adaptive/baseline, etc. The full directory structure is below (output from `tree -d`)
```
.
├── CIFAR10
│   ├── Adam
│   │   ├── adam-resnet-1e3
│   │   │   ├── Adaptive
│   │   │   └── Baseline
│   │   └── densenet-1e4
│   │       ├── Adaptive
│   │       └── Baseline
│   ├── Momentum
│   │   ├── DenseNet
│   │   │   ├── Adaptive
│   │   │   └── Baseline
│   │   └── ResNet
│   │       ├── Adaptive
│   │       │   ├── 1e-2
│   │       │   └── 1e-3
│   │       └── Baseline
│   ├── RMSprop
│   │   ├── DenseNet
│   │   │   ├── Adaptive
│   │   │   └── Baseline
│   │   └── ResNet
│   │       ├── Adaptive
│   │       └── Baseline
│   └── SGD
│       ├── DenseNet
│       │   ├── Adaptive
│       │   └── Baseline
│       └── ResNet20
│           ├── Adaptive
│           │   ├── wd1e3-adaptive
│           │   └── wd1e3-fixed
│           └── Baseline
├── CIFAR100
│   ├── Adam
│   │   ├── DenseNet
│   │   │   ├── Adaptive
│   │   │   │   ├── Constrained filter size
│   │   │   │   │   ├── 1e4
│   │   │   │   │   └── 1e5
│   │   │   │   └── Unconstrained
│   │   │   └── Baseline
│   │   └── ResNet164
│   │       ├── Adaptive
│   │       └── Baseline
│   ├── Momentum
│   │   ├── DenseNet
│   │   │   ├── Adaptive
│   │   │   └── Baseline
│   │   └── ResNet164
│   │       ├── Adaptive
│   │       └── Baseline
│   ├── RMSprop
│   │   ├── DenseNet
│   │   │   ├── Adaptive
│   │   │   └── Baseline
│   │   └── ResNet
│   │       ├── Adaptive
│   │       └── Baseline
│   └── SGD
│       ├── DenseNet
│       │   ├── Adaptive
│       │   │   ├── Constrained Filters
│       │   │   └── Nonconstrained
│       │   └── Baseline
│       ├── ResNet162v2
│       │   ├── Adaptive
│       │   └── Baseline
│       └── ResNet56v2
├── MNIST
└── paper
```

## Setup
The repository uses standard deep learning libraries in Python:
* `keras`
* `sklearn`
* `numpy`
* `matplotlib`
* `pickle`
* `tqdm`

## Credits
This repository uses code from the following sources:
* [ResNet implementation from Keras docs](https://keras.io/examples/cifar10_resnet/)
* [DenseNet implementation by titu1994](https://github.com/titu1994/DenseNet)
* [AdaMo implementation based on Keras Adam implementation](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L436). Note that this URL is based on commit [a139716](https://github.com/keras-team/keras/commit/a1397169ddf8595736c01fcea084c8e34e1a3884).
* [MNIST architecture from Kaggle Python notebook by Aditya Soni](https://www.kaggle.com/adityaecdrid/mnist-with-keras-for-beginners-99457)

## Citation
If you find this work useful, please cite the paper:
```
@article{yedida2019novel,
  title={A novel adaptive learning rate scheduler for deep neural networks},
  author={Yedida, Rahul and Saha, Snehanshu},
  journal={arXiv preprint arXiv:1902.07399},
  year={2019}
}
```
