import numpy as np
from neuralnet_layers import *
from neuralnet_model import *

# initializations
np.random.seed(1)

def network_1():
    # network1
    layers_dims = [14, 100, 40, 4]
    x_train, y_train, x_test, y_test = read_data()
    parameters = L_layer_model('14-100-40-4 network',x_train, y_train, x_test, y_test , layers_dims, learning_rate=0.01, num_iterations=25000, print_cost=True ,beta=0.99, optimizer="momentum")
    pred_train = predict(x_train, y_train, parameters)
    pred_test = predict(x_test, y_test, parameters)

def network_2():
    # network2
    layers_dims = [14,28,28,28,28,28,28,4]
    x_train, y_train, x_test, y_test = read_data()
    parameters = L_layer_model('14-28*6-4 network',x_train, y_train, x_test, y_test , layers_dims, learning_rate=0.01, num_iterations=25000, print_cost=True, beta=0.99, optimizer="momentum")
    pred_train = predict(x_train, y_train, parameters)
    pred_test = predict(x_test, y_test, parameters)

def network_3():
    # network3
    layers_dims = [14]
    for i in range(28):
        layers_dims.append(14)
    layers_dims.append(4)
    x_train, y_train, x_test, y_test = read_data()
    parameters = L_layer_model('14-14*28-4 network',x_train, y_train, x_test, y_test , layers_dims, learning_rate=0.01, num_iterations=25000, print_cost=True, beta=0.99, optimizer="momentum")
    pred_train = predict(x_train, y_train, parameters)
    pred_test = predict(x_test, y_test, parameters)


#network_1()
network_2()
#network_3()