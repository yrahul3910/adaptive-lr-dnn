from __future__ import print_function

import os.path

import densenet
import numpy as np
import pickle
import sklearn.metrics as metrics
from tqdm import tqdm

from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

lrs = []

def lr_schedule(epoch):
    """Learning Rate Schedule
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    Kz = 0.  # max penultimate activation
    
    sess = K.get_session()
    max_wt = 0.
    for weight in model.weights:
        norm = np.linalg.norm(weight.eval(sess))
        if norm > max_wt:
            max_wt = norm
    
    for i in tqdm(range((len(trainX) - 1) // batch_size + 1)):
        start_i = i * batch_size
        end_i = start_i + batch_size
        xb = trainX[start_i:end_i]
    	
        tmp = np.array(func([xb]))
        activ = np.linalg.norm(tmp)

        if activ > Kz:
            Kz = activ

    K_ = ((nb_classes - 1) * Kz) / (nb_classes * batch_size) + 1e-4 * max_wt
    
    lr = 1 / K_
    lrs.append(lr)
    print('Epoch', epoch+1, 'LR =', lr)
    return lr

batch_size = 128
nb_classes = 100
nb_epoch = 300

img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = -1
bottleneck = False
reduction = 0
dropout_rate = 0.0 # 0.0 for data augmentation

model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                          growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, 
                          bottleneck=bottleneck, reduction=reduction, weights=None)
func = K.function([model.layers[0].input], [model.layers[-2].output])

print("Model created")

optimizer = SGD(lr=1e-3, decay=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

(trainX, trainY), (testX, testY) = cifar100.load_data()

trainX = trainX.astype('float32') / 255.
testX = testX.astype('float32') / 255.

trainX_mean = np.mean(trainX, axis=0)
trainX -= trainX_mean
testX -= trainX_mean

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)

generator = ImageDataGenerator(width_shift_range=0.1,
                               height_shift_range=0.1,
                               horizontal_flip=True)

generator.fit(trainX)

# Load model
save_dir = "."
model_name = 'epoch-{epoch:03d}.h5'
filepath = os.path.join(save_dir, model_name)

model_checkpoint= ModelCheckpoint(filepath, monitor="val_acc", save_best_only=True,
                                  verbose=1)
lr_scheduler = LearningRateScheduler(lr_schedule)

callbacks=[lr_scheduler, model_checkpoint]

model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                    callbacks=callbacks,
                    validation_data=(testX, Y_test),
                    validation_steps=testX.shape[0] // batch_size, verbose=1)

model.save('epoch-300-densenet-k12.h5')
history_300 = model.history.history
with open('history.pkl', 'wb') as f:
    pickle.dump(history_300, f)

print('Saved history.')
print('Learning rate history:\n========================')
print(lrs)
print('==========================\n')

print("Done.")

