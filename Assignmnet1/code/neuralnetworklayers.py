import csv
import numpy as np
import matplotlib.pyplot as plt


def read_data():
    x_train = []
    y_train= []
    x_test=[]
    y_test=[]
    with open('../data/question_2_1/x_train.csv', newline='') as csvfile:
        x_train_r = csv.reader(csvfile)
        for row in x_train_r:
            x_train.append(list(int(x) for x in list(row)))

    with open('../data/question_2_1/y_train.csv', newline='') as csvfile:
        y_train_r = csv.reader(csvfile)
        for row in y_train_r:
            if row[0] == '0':
                y_train.append([1, 0, 0, 0])
            elif row[0] == '1':
                y_train.append([0, 1, 0, 0])
            elif row[0] == '2':
                y_train.append([0, 0, 1, 0])
            elif row[0] == '3':
                y_train.append([0, 0, 0, 1])

    with open('../data/question_2_1/x_test.csv', newline='') as csvfile:
        x_test_r = csv.reader(csvfile)
        for row in x_test_r:
            x_test.append(list(int(x) for x in list(row)))

    with open('../data/question_2_1/y_test.csv', newline='') as csvfile:
        y_test_r = csv.reader(csvfile)
        for row in y_test_r:
            if row[0] == '0':
                y_test.append([1, 0, 0, 0])
            elif row[0] == '1':
                y_test.append([0, 1, 0, 0])
            elif row[0] == '2':
                y_test.append([0, 0, 1, 0])
            elif row[0] == '3':
                y_test.append([0, 0, 0, 1])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = np.transpose(x_train)
    y_train = np.transpose(y_train)
    x_test = np.transpose(x_test)
    y_test = np.transpose(y_test)

    return x_train, y_train, x_test, y_test

# csv file reading
def read_file_within_limit(name,lower,upper):
    l = list();
    with open(name) as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if ((i >= lower) and (i < upper)):
                row.pop(0)
                l.append(row)
            i = i + 1
    return l

# csv file reading
def read_file(name):
    l =list();
    with open(name,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            l.append(row)
    return l

# csv file writing
def write_array_to_file(name, arr):
    with open(name, 'ab') as f:
        np.savetxt(f, arr, delimiter=",")

def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)  # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2.0/layer_dims[l])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

def relu_forward(z):
    A = np.maximum(0, z)
    assert (A.shape == z.shape)
    cache = z
    return A, cache

def relu_backward(dA, activation_cache):
    #dZ = dA * (activation_cache>0)
    #return dZ
    Z = activation_cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True),z

def stable_softmax(z):
    exps = np.exp(z - np.max(z))
    sum = np.sum(exps,axis=0,keepdims=True)
    exps = exps / sum
    assert (exps.shape == z.shape)
    return exps,z

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b,activation):
    if activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = stable_softmax(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu_forward(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters)//2    # number of layers in the neural network
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],'relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'softmax')
    caches.append(cache)

    assert (AL.shape == (4, X.shape[1]))
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost= - np.multiply(Y, np.log(AL)).sum() / m
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, np.transpose(A_prev)) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(np.transpose(W), dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache):
    linear_cache, activation_cache = cache
    dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    current_cache = caches[L - 1]
    linear_cache, activation_cache = current_cache
    dZ = AL - Y
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZ, linear_cache)

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate,decay=1.0):
    L = len(parameters) // 2  # number of layers in the neural network
    learning_rate *= decay
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters['W' + str(l + 1)].shape))
        v["db" + str(l + 1)] = np.zeros((parameters['b' + str(l + 1)].shape))
    return v

def update_parameters_with_momentum(parameters, grads, v, learning_rate , beta=0.95, decay=1.0):
    L = len(parameters) // 2

    for l in range(L):
        v["dW" + str(l + 1)] = (beta * v["dW" + str(l + 1)]) + ((1 - beta) * grads['dW' + str(l + 1)])
        v["db" + str(l + 1)] = (beta * v["db" + str(l + 1)]) + ((1 - beta) * grads['db' + str(l + 1)])
        learning_rate *= decay
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - (learning_rate * v["dW" + str(l + 1)])
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - (learning_rate * v["db" + str(l + 1)])

    return parameters, v

def L_layer_model(network_name,X, Y,x,y,layers_dims, learning_rate=0.01, num_iterations=3000, print_cost=False, beta = 0.95, optimizer=None):  # lr was 0.009
    # keep track of cost
    train_costs = []
    train_accuracies = []
    test_costs = []
    test_accuracies = []
    iterations=[]

    parameters = initialize_parameters_deep(layers_dims)
    # Loop (gradient descent)
    if optimizer == "momentum":
        v = initialize_velocity(parameters)

    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        # Compute cost.
        train_cost = compute_cost(AL, Y)
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
        # Update parameters.
        if optimizer == "momentum":
            parameters, v = update_parameters_with_momentum(parameters, grads, v,learning_rate,beta, decay=0.99995)
        else:
            parameters = update_parameters(parameters, grads, learning_rate,decay=0.99995)

        train_accuracy = predict(X,Y, parameters)

        AL_test, caches_test = L_model_forward(x, parameters)

        test_cost = compute_cost(AL_test,y)

        test_accuracy = predict(x, y, parameters)


        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Train cost after iteration %i:" % (i) + str(train_cost))
            print("Train accuracy after iteration %i:" % (i) + str(train_accuracy))
        if print_cost and i % 100 == 0:
            train_costs.append(train_cost)
            train_accuracies.append(train_accuracy)


        if print_cost and i % 100 == 0:
            print("Test cost after iteration %i:" % (i) + str(test_cost))
            print("Test accuracy after iteration %i:" % (i) + str(test_accuracy))
            print("---------第"+str(i)+"次迭代结束--------")
        if print_cost and i % 100 == 0:
            test_costs.append(test_cost)
            test_accuracies.append(test_accuracy)
            iterations.append(i)

    # plot the train and test loss
    plt.plot(iterations,train_costs, '-b', label='train data')
    plt.plot(iterations,test_costs,'-r', label='test data')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title("Loss for "+ network_name + ",Learning rate = " + str(learning_rate))
    plt.show()

    # plot the train and test accuracy
    plt.plot(iterations,train_accuracies, '-b', label='train data')
    plt.plot(iterations,np.squeeze(test_accuracies), '-r', label='test data')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title("Accuracy for"+ network_name + ",Learning rate = " + str(learning_rate))
    plt.show()

    return parameters

def predict(X, Y, parameters):
    m = X.shape[1]
    pred = np.zeros((4, m))
    prob, caches = L_model_forward(X, parameters)
    p = np.argmax(prob, axis=0)
    for i in range(0, prob.shape[1]):
        pred[p[i],i]=1
    accuracy = np.sum((pred == Y) / (4 * m))
    return accuracy


