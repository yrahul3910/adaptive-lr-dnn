# CIFAR-10 Experiments
This directory contains the experiments conducted on CIFAR-10. The files have been divided into two directories, depending on whether the learning rate used a fixed or adaptive policy. Both experiments used a weight decay of 1e-3.

**Fixed:** The model is run for one epoch, and the constant is used to compute the Lipschitz constant.
**Adaptive:** Before each epoch, the constant is computed from the results of the previous epoch. Initially, this is set arbitrarily to 1e-3.

In both cases, the learning rate is the inverse of the Lipschitz constant.
