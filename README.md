# Code for "A novel adaptive learning rate scheduler for deep neural networks"
All results and code from the paper can be found here. Trained models and program outputs are also uploaded here. The top-level directories in this repository correspond to the dataset the experiments were run on. The `paper` directory contains the paper source and compiled PDF.

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
│   │   │   ├── 1e4
│   │   │   └── 1e5
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
│   └── SGD
│       ├── DenseNet
│       │   ├── Adaptive
│       │   └── Baseline
│       ├── ResNet162v2
│       │   ├── Adaptive
│       │   └── Baseline
│       └── ResNet56v2
├── MNIST
└── paper
```
