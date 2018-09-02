from NeuralNetworks import *
import pickle
import copy
import numpy as np

np.seterr(divide='raise', over='warn', under='warn')

lr = 1e-3
lr_decay = 0.99995
n_iter = 2500
record_every = n_iter / 50

input_num = 14
output_num = 4

batch_sizes = [1, 30]
network_names = ['network_1', 'network_2', 'network_3']
b_size_suffix = {1: '',  30:'_30', 128:'_128'}

network_1 = [
        FullyConnectedLayer(input_num, 100, lr=lr),
        ReLULayer(),
        FullyConnectedLayer(100, 40, lr=lr),
        ReLULayer(),
        FullyConnectedLayer(40, output_num, lr=lr),
        SoftmaxOutput_CrossEntropyLossLayer()
    ]

network_2 = [FullyConnectedLayer(input_num, 28, lr=lr), ReLULayer()]
for p in range(5):
    network_2.append(FullyConnectedLayer(28, 28, lr=lr))
    network_2.append(ReLULayer())
network_2.append(FullyConnectedLayer(28, output_num, lr=lr))
network_2.append(SoftmaxOutput_CrossEntropyLossLayer())

network_3 = [FullyConnectedLayer(input_num, 14, lr=lr, scale=4), ReLULayer()]
for p in range(27):
    network_3.append(FullyConnectedLayer(14, 14, lr=lr, scale=4))
    network_3.append(ReLULayer())
network_3.append(FullyConnectedLayer(14, output_num, lr=lr, scale=4))
network_3.append(SoftmaxOutput_CrossEntropyLossLayer())

x_train, y_train, x_test, y_test = load_data('../data/question_2_1')




networks = {'network_1': network_1, 'network_2': network_2, 'network_3': network_3}

for n_name in network_names:
    for b_size in [1,30]:
        network = copy.deepcopy(networks[n_name])

        network, \
        train_loss_record, train_acc_record, test_loss_record, test_acc_record \
            = train(network, x_train, y_train, n_iter, lr, x_test=x_test, y_test=y_test,
                    record_every=record_every, batch_size=b_size, decay_rate=lr_decay, require_test=True)
        num_test, correct, acc, cost = test(network, x_test, y_test)
        print('test accuracy = {:2f}'.format(acc))

        with open(n_name + b_size_suffix[b_size] + '.bin', 'wb') as f:
            pickle.dump({'network': network,
                         'train_l_r': train_loss_record,
                         'train_a_r': train_acc_record,
                         'test_l_r': test_loss_record,
                         'test_a_r': test_acc_record}, f)
