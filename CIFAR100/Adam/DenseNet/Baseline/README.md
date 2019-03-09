# Experiments on CIFAR-100 using DenseNet
The files are described below.

* `cif.py`: The code.
* `epoch-240.h5`: The model with the best validation accuracy over 300 epochs, trained with weight decay 1e-4
* `epoch-300.h5`: The model after 300 epochs for the above model
* `history.pkl`: The history object for the above model
* `nohup-1e4.out`: The program output for the above
* `nohup-noreg.out`: The program output for the model with no regularization. Performance was stuck, and was stopped after 75 epochs.
* `cif-bc.py`: The code for DenseNet-BC. No experiments have yet been run with this.
