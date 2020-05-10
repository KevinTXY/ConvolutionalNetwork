import pickle
import tensorflow as tf

from tensorflow.keras import layers, models
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


class AlexNet(models.Sequential):
    def __init__(self, input_shape, classes):
        super().__init__()
        self.add(layers.Conv2D(filters=96, kernel_size=(11, 11),
                               strides=(4, 4), padding='valid',
                               activation='relu',
                               input_shape=input_shape))
        # should add some normalization
        self.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='valid'))

        self.add(layers.Conv2D(filters=256, kernel_size=(5, 5),
                               strides=(1, 1), padding='same',
                               activation='relu'))
        # should add some normalization
        self.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='valid'))

        self.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                               strides=(1, 1), padding='same',
                               activation='relu'))
        self.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                               strides=(1, 1), padding='same',
                               activation='relu'))
        self.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                               strides=(1, 1), padding='same',
                               activation='relu'))
        self.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='valid'))

        self.add(layers.Flatten())
        self.add(layers.Dense(4096, activation='relu'))
        self.add(layers.Dropout(.5))
        self.add(layers.Dense(4096, activation='relu'))
        self.add(layers.Dropout(.5))
        self.add(layers.Dense(classes, activation='softmax'))


def loadBatch(filename):
    path = 'datasets/cifar10/'
    with open(path + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        y = np.asarray(dict[b'labels'])
        X = dict[b'data']
        X = X / 255
        Y = to_categorical(y)
    return X, Y, y


def normalize(X, mean_train, std_train):
    return (X - mean_train) / std_train


def resize(X_cifar):
    """
    rescale a cifar image to get an image of size 227*227*3
    """
    X_cifar = X_cifar.reshape((len(dict[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    return tf.image.resize(X_cifar, [227, 227])


def main():
    # Load data
    Xtrain, Ytrain, ytrain = loadBatch('data_batch_1')
    Xval, Yval, yval = loadBatch('data_batch_2')
    Xtest, Ytest, ytest = loadBatch('test_batch')

    # normalize the data
    mean_train = np.mean(Xtrain, axis=0)
    std_train = np.std(Xtrain, axis=0)
    Xtrain = normalize(Xtrain, mean_train, std_train)
    Xval = normalize(Xval, mean_train, std_train)
    Xtest = normalize(Xtest, mean_train, std_train)

    # reshape cifar images from 32 * 32 * 3 to 227 * 227 * 3
    # not a very good solution
    Xtrain = resize(Xtrain)
    Xval = resize(Xval)
    Xtest = resize(Xtest)

    # hyper-params
    eta = 0.01
    n_batch = 128
    epochs = 10

    # create & train the model
    model = AlexNet((227, 227, 3), 10)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=eta, momentum=.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    history = model.fit(Xtrain, Ytrain, epochs=epochs,
                        batch_size=n_batch)

    # plots
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    test_loss, test_acc = model.evaluate(Xtest, Ytest, verbose=2)


if __name__ == '__main__':
    main()



        
        
        
                                  
