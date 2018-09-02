from NeuralNetworkLayers import *
import numpy as np

def network_forward(network, input_data, label_data=None, phase='train'):
    for layer in network:
        if type(layer) is not SoftmaxOutput_CrossEntropyLossLayer:
            input_data = layer.forward(input_data)
        else:
            layer.eval(input_data, label_data, phase)
    return network

def network_backward(network):
    for layer in reversed(network):
        if type(layer) is SoftmaxOutput_CrossEntropyLossLayer:
            gradient = layer.backward()
        else:
            gradient = layer.backward(gradient)
    return network

def network_SGD(network, decay=1.0):
    for layer in reversed(network):
        if type(layer) is FullyConnectedLayer:
            layer.lr *= decay
            layer.W -= layer.lr * layer.gW
            layer.b -= layer.lr * layer.gb
        else:
            continue
    return network

def network_momentum_SGD(network, decay=1.0, rho=0.99):
    for layer in reversed(network):
        if type(layer) is FullyConnectedLayer:
            layer.vW = layer.vW * rho + layer.gW
            layer.vb = layer.vb * rho + layer.gb
            layer.lr *= decay
            layer.W -= layer.lr * layer.vW
            layer.b -= layer.lr * layer.vb
        else:
            continue
    return network

def load_data(path):
    import csv
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with open(path + '/x_train.csv', 'r') as f:
        x_train_reader = csv.reader(f)
        for row in x_train_reader:
            x_train.append(list(map(int, row)))
    with open(path + '/x_test.csv', 'r') as f:
        x_test_reader = csv.reader(f)
        for row in x_test_reader:
            x_test.append(list(map(int, row)))
    with open(path + '/y_train.csv', 'r') as f:
        y_train_reader = csv.reader(f)
        for row in y_train_reader:
            y_train.append(list(map(int, row)))
    with open(path + '/y_test.csv', 'r') as f:
        y_test_reader = csv.reader(f)
        for row in y_test_reader:
            y_test.append(list(map(int, row)))

    x_train = np.array(x_train, dtype=np.float128)
    y_train = np.array(y_train, dtype=np.float128)
    x_test = np.array(x_test, dtype=np.float128)
    y_test = np.array(y_test, dtype=np.float128)

    return x_train, y_train, x_test, y_test

def train(network, x_train, y_train, n_iter=10000, lr=1e-3, print_every=500, record_every=200,
          x_test=None, y_test=None, batch_size=1, decay_rate=1, require_test=True, optimizer=network_SGD):
    import random
    num_train = x_train.shape[0]

    choice_range = [x for x in range(0, num_train)]
    output_num = len(np.unique(y_train))
    # main training loop
    loss_accu = 0
    train_loss_record = np.array([]).astype(np.float128)
    train_acc_record = np.array([]).astype(np.float128)
    test_loss_record = np.array([]).astype(np.float128)
    test_acc_record = np.array([]).astype(np.float128)

    # def next(current, total):
    #     if current+1 == total:
    #         return 0
    #     else:
    #         return current+1
    # idx = [-1]
    for p in range(1, n_iter + 1):
        idx = []
        for q in range(0, batch_size):
            idx.append(random.choice(choice_range))
        # idx = [next(idx[0], num_train)]
        x_sample = x_train[idx].astype(np.float128).T
        if batch_size == 1:
            y_sample = np.zeros([output_num, 1]).astype(np.float128)
            y_sample[int(y_train[idx])] = 1
        else:
            y_sample = np.zeros([output_num, batch_size]).astype(np.float128)
            for q in range(0, batch_size):
                y_sample[int(y_train[idx[q]])][q] = 1

        if (p-1) % record_every == 0 and require_test:
            _, _, train_acc, train_loss = test(network, x_train, y_train)
            _, _, test_acc, test_loss = test(network, x_test, y_test)
            train_loss_record = np.append(train_loss_record, train_loss)
            train_acc_record = np.append(train_acc_record, train_acc)
            test_loss_record = np.append(test_loss_record, test_loss)
            test_acc_record = np.append(test_acc_record, test_acc)
            print('iter = {:8d}, test acc = {:.5f}'.format(p-1, test_acc))

        network = network_forward(network, x_sample, y_sample)
        network = network_backward(network)
        network = network_SGD(network, decay_rate)
        # network = network_momentum_SGD(network, decay_rate, 0.99)

        loss = network[-1].output_data
        loss_accu += loss

        if p % print_every == 0:
            print('iter = {:8d}, loss = {:.5f}'.format(p, loss_accu / print_every))
            loss_accu = 0

    return network, train_loss_record, train_acc_record, test_loss_record, test_acc_record

def test(network, x_test, y_test, batch_size=1):
    correct = 0
    num_test = x_test.shape[0]
    output_num = len(np.unique(y_test))
    cost = 0
    for p in range(0, num_test):

        x_sample = np.expand_dims(x_test[p], axis=1).astype(np.float128)
        y_sample = np.zeros([output_num, 1]).astype(np.float128)
        y_sample[int(y_test[p])] = 1

        network = network_forward(network, x_sample, y_sample, phase='train')
        cost += 0 if network[-1].output_data is np.nan else network[-1].output_data

        pred = network[-1].y_pred.argmax()
        if pred == y_test[p]:
            correct += 1
    cost /= num_test
    return num_test, correct, correct*1.0/num_test, cost

def num2Vec(label, total_labels):
    result = np.zeros([total_labels, 1]).astype(np.float128)
    result[int(label)] = 1
    return result