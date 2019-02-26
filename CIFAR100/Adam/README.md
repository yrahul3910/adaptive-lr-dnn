# ResNet164 Experiments on CIFAR-100 using Adam
The files are described below.

* `nohup-none.out`: The model using no weight decay and adaptive learning rate. Performance decreased after a few epochs, so was stopped.  
* `nohup-fixed.out`: The model using no weight decay and a fixed learning rate. Performance stagnated, so was stopped.
* `epoch-29-fixed.h5`: The above (fixed) saved model after 29 epochs.
* `cifar100_rn.py`: The code for the model.
