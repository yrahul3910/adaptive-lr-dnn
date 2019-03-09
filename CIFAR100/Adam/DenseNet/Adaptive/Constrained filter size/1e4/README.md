# Experiments on CIFAR-100 using DenseNet
The files are described below.

* `cif.py`: The code.
* `epoch-240.h5`: The model with the best validation accuracy over 300 epochs, trained with weight decay 1e-4
* `epoch-300.h5`: The model after 300 epochs for the above model
* `history.pkl`: The history object for the above model
* `nohup-1e4.out`: The program output for the above
* `nohup-noreg.out`: The program output for the model with no regularization. Performance was stuck, and was stopped after 75 epochs.

Note that although the code has the computation of weights commented out and a wrong weight decay, the program was modified after execution. This is evidenced by the output in `nohup-1e4.out` showing `tqdm` computing the weights. The code is in fact the one used for the model with no weight decay.
