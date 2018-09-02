import numpy as np
from neuralnetworklayers import *


def load_data_from_txt():
    X = [-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]
    X = np.array(X)
    X = X.reshape(X.shape[0], -1)
    Y = [0, 0, 0, 1]
    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0], -1)
    return X, Y

def form_parameters(wname,bname,layer_dims):
    parameters = {}
    L = len(layer_dims)
    lower = 0
    for l in range(1, L):
        upper = layer_dims[l-1] + lower
        w_list = read_parameters_from_file(wname, lower, upper)
        parameters['W' + str(l)] = w_list
        lower = upper
        b_list = read_parameters_from_file(bname,l-1,l)
        parameters['b' + str(l)] = b_list
        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def compute_gradients(X,Y,parameters):
    #forward propagation
    AL, caches = L_model_forward(X, parameters)
    #backward propagation
    grads = L_model_backward(AL, Y, caches)
    return grads

def read_parameters_from_file(name, lower, upper):
    parameter_list = read_file_within_limit(name,lower,upper)
    parameter_list = np.array(parameter_list)
    parameter_list = np.float32(parameter_list)
    parameter_list = parameter_list.reshape(parameter_list.shape[0], -1).T
    return parameter_list

def write_grads_to_csv(name,layers_dims,grads):
    L = len(layers_dims)
    for l in range(1, L):
        dw = grads['dW' + str(l)].T
        write_array_to_file(name[0], dw)
        db = grads['db' + str(l)].T
        write_array_to_file(name[1], db)

def checkgradient_network_1():
    # network1
    layers_dims = [14, 100, 40, 4]
    X, Y = load_data_from_txt()
    wname = "../data/question_2_2c/w-100-40-4.csv"
    bname = "../data/question_2_2c/b-100-40-4.csv"
    parameters = form_parameters(wname,bname,layers_dims)
    grads = compute_gradients(X, Y, parameters)
    dwname = "../Question_2/dw-100-40-4.csv"
    dbname = "../Question_2/db-100-40-4.csv"
    name = [dwname, dbname]
    write_grads_to_csv(name,layers_dims, grads)

def checkgradient_network_2():
    # network2
    layers_dims = [14, 28, 28, 28, 28, 28, 28, 4]
    X, Y = load_data_from_txt()
    wname = "../data/question_2_2c/w-28-6-4.csv"
    bname = "../data/question_2_2c/b-28-6-4.csv"
    parameters = form_parameters(wname, bname, layers_dims)
    grads = compute_gradients(X, Y, parameters)
    dwname = "../Question_2/dw-28-6-4.csv"
    dbname = "../Question_2/db-28-6-4.csv"
    name = [dwname, dbname]
    write_grads_to_csv(name,layers_dims, grads)

def checkgradient_network_3():
    # network3
    layers_dims = [14]
    for i in range(28):
        layers_dims.append(14)
    layers_dims.append(4)
    X, Y = load_data_from_txt()
    wname = "../data/question_2_2c/w-14-28-4.csv"
    bname = "../data/question_2_2c/b-14-28-4.csv"
    parameters = form_parameters(wname, bname, layers_dims)
    grads = compute_gradients(X, Y, parameters)
    dwname = "../Question_2/dw-14-28-4.csv"
    dbname = "../Question_2/db-14-28-4.csv"
    name = [dwname, dbname]
    write_grads_to_csv(name,layers_dims,grads)

checkgradient_network_1()
checkgradient_network_2()
checkgradient_network_3()
