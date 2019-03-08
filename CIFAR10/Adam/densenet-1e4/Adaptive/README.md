# Experiments on CIFAR-10 using Adam and DenseNet (k=12)
The files are as below.

* `cifar10.py`: The code that uses the other approximation (that diverges).
* `cif.py`: The working code. Adding `lambda ** 2 * max_wt ** 2` to `K2` makes the model slightly poorer.
* `densenet.py`: The DenseNet code, from [this GitHub repo](https://github.com/titu1994/DenseNet)
* `epoch-299.h5`: The model with the best validation performance, at 299 epochs.
* `epoch-300.h5`: The model after 300 epochs.
* `history.pkl`: The history file after 300 epochs.
* `history-sq.pkl`: The history file up to 151 epochs when the squared term was added to `K2`.
* `nohup.out`: The program output.
* `nohup-sq.out`: The program output for the squared term added.
