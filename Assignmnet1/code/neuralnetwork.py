import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from datapreprocess import *

# initializations
np.random.seed(1)

def network_1():
    # network1
    layers_dims = [14, 100, 40, 4]
    train_x, train_y, test_x, test_y = read_data()
    parameters = L_layer_model(train_x, train_y, test_x, test_y, layers_dims, learning_rate=0.1, num_iterations=2000, print_cost=True)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)

def network_2():
    # network2
    layers_dims = [14,28,28,28,28,28,28,4]
    train_x, train_y, test_x, test_y = read_data()
    parameters = L_layer_model(train_x, train_y,test_x, test_y, layers_dims, learning_rate=0.1, num_iterations=1300, print_cost=True, beta=0.9, optimizer="momentum")
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)

def network_3():
    # network3
    layers_dims = [14]
    for i in range(28):
        layers_dims.append(14)
    layers_dims.append(4)
    train_x, train_y, test_x, test_y = read_data()
    parameters = L_layer_model(train_x, train_y, test_x, test_y, layers_dims, learning_rate=0.01, num_iterations=700, print_cost=True, beta=0.9, optimizer="momentum", mini_batch_size=64)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)


network_1()
#network_2()
#network_3()