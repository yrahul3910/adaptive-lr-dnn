# Experiments on CIFAR-10 with ResNet
The files are described below.  

* `cifar.py`: The code for the model.
* `history-fixed.pkl`: The history for the model with a fixed learning rate (computed to be 0.05) and weight decay 1e-4.
* `nohup-1e4.out`: The program output run only for a few epochs with an adaptive learning rate and weight decay 1e-4.
* `nohup-fixed.out`: The program output for 300 epochs for the model with fixed learning rate and weight decay 1e-4.
* `nohup-none.out`: The output for the program run for only a few epochs with an adaptive learning rate and no weight decay.

Clearly, decreasing weight decay in adaptive learning decreased performance, so 1e-3 was chosen to run.
