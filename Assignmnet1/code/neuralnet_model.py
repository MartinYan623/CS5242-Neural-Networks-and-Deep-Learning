import numpy as np
import matplotlib.pyplot as plt
from neuralnet_layers import *


def L_layer_model(network_name, X, Y, x, y, layers_dims, learning_rate=0.01, num_iterations=3000, print_cost=False,
                  beta=0.95, optimizer=None):
    # keep track of cost
    train_costs = []
    train_accuracies = []
    test_costs = []
    test_accuracies = []
    iterations = []

    parameters = initialize_parameters_deep(layers_dims)
    if optimizer == "momentum":
        v = initialize_v(parameters)

    for i in range(0, num_iterations+1):
        # use full batch learing
        # Forward propagation
        AL, caches = L_model_forward(X, parameters)
        # Compute cost.
        train_cost = compute_cost(AL, Y)
        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
        # Update parameters
        if optimizer == "momentum":
            parameters, v = update_parameters_with_momentum(parameters, grads, v, learning_rate, beta,
                                                                decay=0.99995)
        else:
            parameters = update_parameters(parameters, grads, learning_rate, decay=0.99995)

        train_accuracy = predict(X, Y, parameters)
        AL_test, caches_test = L_model_forward(x, parameters)
        test_cost = compute_cost(AL_test, y)
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
            print("---------第" + str(i) + "次迭代结束--------")
        if print_cost and i % 100 == 0:
            test_costs.append(test_cost)
            test_accuracies.append(test_accuracy)
            iterations.append(i)

    # plot the train and test loss
    plt.plot(iterations, train_costs, '-b', label='cost train data')
    plt.plot(iterations, test_costs, '-r', label='cost test data')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.grid(True)
    plt.title("Loss for " + network_name + ",Learning rate = " + str(learning_rate))
    plt.savefig('/Users/martin_yan/Desktop/loss%s.png'%network_name,dpi=300)
    #plt.show()
    plt.figure()
    # plot the train and tes"t accuracy
    plt.plot(iterations, train_accuracies, '-b', label='acc. train data')
    plt.plot(iterations, test_accuracies, '-r', label='acc. test data')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.grid(True)
    plt.title("Accuracy for" + network_name + ",Learning rate = " + str(learning_rate))
    plt.savefig('/Users/martin_yan/Desktop/acc%s.png' % network_name, dpi=300)
    #plt.show()

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