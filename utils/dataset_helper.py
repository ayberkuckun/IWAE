import numpy as np
import scipy
from tensorflow import keras


def prepare_data(dataset, implementation):
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        image_size = x_train.shape[1]

        x_train = np.reshape(x_train, [-1, image_size ** 2])
        x_test = np.reshape(x_test, [-1, image_size ** 2])

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

    elif dataset == "fashionmnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        image_size = x_train.shape[1]

        x_train = np.reshape(x_train, [-1, image_size ** 2])
        x_test = np.reshape(x_test, [-1, image_size ** 2])

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

    elif dataset == "omniglot":
        path = 'chardata.mat'
        omni_raw = scipy.io.loadmat(path)

        x_train = reshape_data(omni_raw['data'].T.astype("float32"))
        x_test = reshape_data(omni_raw['testdata'].T.astype("float32"))

        image_size = int(np.sqrt(x_train.shape[1]))

    else:
        raise Exception("Not defined dataset!")

    x_train = np.random.binomial(1, x_train).astype('float32')
    x_test = np.random.binomial(1, x_test).astype('float32')

    return x_train, x_test, image_size


def reshape_data(data):
    if data.shape[1] != 28:
        return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='F')
    else:
        return data.reshape((-1, 28 * 28))
