import csv
import numpy as np
import matplotlib.pyplot as plt


def read_data(file_name):
    data_set = []
    with open(file_name, newline='') as csvfile:
        data_file = csv.reader(csvfile)
        for row in data_file:
            data_set.append(list(float(x) for x in list(row)))
    return np.array(data_set)


def initialize_parameters(n_x,n_y):
    #np.random.seed(2)
    W1 = np.random.randn(n_y, n_x) * 0.01
    b1 = np.zeros((n_y, 1))
    #W1 = np.array([0.08,1.2]).reshape(2,1)
    #b1 = np.array([0.3]).reshape(1,1)
    parameters = {"W1": W1,
                  "b1": b1,}
    return parameters

#parameters = initialize_parameters(2,1)
#print("W1 = " + str(parameters["W1"]))
#print("b1 = " + str(parameters["b1"]))


def relu(x):
    n=x.shape[1]
    for i in range(n):
        if x[0,i] > 0:
            pass
        else:
            x[0,i]=0
    return x

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    cache = {"Z1": Z1,
             "A1": A1,}
    return A1, cache
#A1, cache = forward_propagation(x_train, parameters)
#print(np.mean(cache['Z1']) ,np.mean(cache['A1']))
#print(cache['Z1'])
#print(cache['A1'])

def compute_cost(A1, Y, parameters):
    m= y_train.shape[1]
    cost = ( 0.5 / m ) * np.sum(pow(A1-Y,2))
    return cost

#print("cost = " + str(compute_cost(A1,y_train, parameters)))

def backward_propagation(parameters, cache, X, Y):
    m = x_train.shape[1]
    #W1 = parameters['W1']
    A1 = cache['A1']
    dZ1 = A1 - Y
    dW1 = 1 / m * np.dot(dZ1, np.transpose(X))
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,}

    return grads
#grads = backward_propagation(parameters, cache,x_train,y_train)
#print ("dW1 = "+ str(grads["dW1"]))
#print ("db1 = "+ str(grads["db1"]))

def update_parameters(parameters, grads, learning_rate = 0.001):

    W1 = parameters['W1']
    b1 = parameters['b1']
    dW1 = grads['dW1']
    db1 = grads['db1']
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    parameters = {"W1": W1,
                  "b1": b1,}

    return parameters

#print("W1 = " + str(parameters["W1"]))
#print("b1 = " + str(parameters["b1"]))


def nn_model(X, Y, num_iterations = 1000, print_cost=False):
    np.random.seed(3)
    parameters = initialize_parameters(2,1)
    W1 = parameters['W1']
    b1 = parameters['b1']
    for i in range(0, num_iterations):

        ### START CODE HERE ### (≈ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A1, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A1, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)

        ### END CODE HERE ###

        # Print the cost every 1000 iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters

x_train = read_data('data/x_train.csv')
y_train = read_data('data/y_train.csv')
x_train = x_train.transpose()
y_train = y_train.transpose()
# x_train=np.array([-2,2]).reshape(2,1)
# y_train=np.array([0]).reshape(1,1)

parameters = nn_model(x_train, y_train, num_iterations=1000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))

def predict(parameters, X):
    A1, cache = forward_propagation(X, parameters)
    print(A1)
    predictions = (A1 > 0.5)
    return predictions

predictions = predict(parameters, x_train)
print(predictions)
#print("predictions mean = " + str(np.mean(predictions)))
print ('Accuracy: %d' % float((np.dot(y_train,np.transpose(predictions)) + np.dot(1-y_train,np.transpose(1-predictions)))/float(y_train.size)*100) + '%')