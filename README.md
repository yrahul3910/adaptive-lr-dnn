# Code for "A novel adaptive learning rate scheduler for deep neural networks"
All results and code from the paper can be found here. Trained models and program outputs are also uploaded here. The top-level directories in this repository correspond to the dataset the experiments were run on. The `paper` directory contains the compiled PDF.

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
