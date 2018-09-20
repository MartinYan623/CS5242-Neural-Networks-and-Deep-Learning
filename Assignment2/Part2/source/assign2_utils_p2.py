import os
import struct
import numpy as np

def _read(dataset = "training", path = "."):
    """
    Source: https://gist.github.com/akesling/5358964 
    Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    which is GPL licensed.
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def mnist_reader():
    from keras.utils import to_categorical
    train_data_iter = _read(dataset = 'training', path='../data/mnist')
    test_data_iter = _read(dataset = 'testing', path='../data/mnist')
    train_x = [] ; train_y = []
    test_x = [] ; test_y = []

    for d in train_data_iter:
        train_x.append(d[1])
        train_y.append(d[0])
    for d in test_data_iter:
        test_x.append(d[1])
        test_y.append(d[0])
    
    class_name = list(range(0,10))
    return np.expand_dims(np.stack(train_x, 0)/255.0, 3), to_categorical(np.array(train_y)), np.expand_dims(np.stack(test_x)/255.0, 3), to_categorical(np.array(test_y)), class_name
    
def cifar10_reader():
    from cifar10 import load_training_data, load_test_data, load_class_names
    train_x, _, train_y = load_training_data()
    test_x, _, test_y = load_test_data()
    class_name = load_class_names()
    return train_x, train_y, test_x, test_y, class_name
