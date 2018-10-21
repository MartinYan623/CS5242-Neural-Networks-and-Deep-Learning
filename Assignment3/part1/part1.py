from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Activation
from sklearn.metrics import mean_squared_error
from keras.optimizers import RMSprop,Adam
from numpy.random import seed
seed(1)

# Create model
def create_fc_model():
    model = Sequential([
        Dense(20, input_dim=x_train.shape[1]),
        Activation('relu'),
        Dense(1)
    ])
    return model

# split train/test data
def split_data(x, y, ratio=0.8):
    to_train = int(len(x.index) * ratio)
    # tweak to match with batch_size
    to_train -= to_train % batch_size

    x_train = x[:to_train]
    y_train = y[:to_train]
    x_test = x[to_train:]
    y_test = y[to_train:]

    # tweak to match with batch_size
    to_drop = x.shape[0] % batch_size
    if to_drop > 0:
        x_test = x_test[:-1 * to_drop]
        y_test = y_test[:-1 * to_drop]

    # some reshaping
    ##### RESHAPE YOUR DATA BASED ON YOUR MODEL #####
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values
    x_train = x_train.reshape(to_train, length)
    y_train = y_train.reshape(to_train, 1)
    x_test = x_test.reshape(len(x.index) - to_train, length)
    y_test = y_test.reshape(len(x.index) - to_train, 1)

    return (x_train, y_train), (x_test, y_test)

# training parameters passed to "model.fit(...)"
batch_size = 1
epochs = 10

# The input sequence min and max length that the model is trained on for each output point
min_length = 1
max_length = 10

# load data from files
noisy_data = np.loadtxt('../filter_data/noisy_data.txt', delimiter='\t', dtype=np.float)
smooth_data = np.loadtxt('../filter_data/smooth_data.txt', delimiter='\t', dtype=np.float)

print('noisy_data shape:{}'.format(noisy_data.shape))
print('smooth_data shape:{}'.format(smooth_data.shape))
print('noisy_data first 5 data points:{}'.format(noisy_data[:5]))
print('smooth_data first 5 data points:{}'.format(smooth_data[:5]))


# List to keep track of root mean square error for different length input sequences
fc_rmse_list = list()

for num_input in range(min_length, max_length+1):
    length = num_input
    print("*" * 33)
    print("INPUT DIMENSION:{}".format(length))
    print("*" * 33)

    # convert numpy arrays to pandas dataframe
    data_input = pd.DataFrame(noisy_data)
    expected_output = pd.DataFrame(smooth_data)

    # when length > 1, arrange input sequences
    if length > 1:
        # ARRANGE YOUR DATA SEQUENCES
        # lose L-1 input data
        empty = pd.DataFrame(data=None, columns=range(0, length))
        for i in range(length-1, len(data_input)):
            list = []
            for j in range(i, i-length, -1):
                list.append(data_input.iloc[j][0])
            empty.loc[i - length + 1] = list
        data_input = empty
        # lose L-1 output data
        for i in range(length-1):
            expected_output = expected_output.drop(i)
            expected_output = expected_output.reset_index(drop=True)

    print('data_input length:{}'.format(len(data_input.index)))
    # Split training and test data: use first 80% of data points as training and remaining as test
    (x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)
    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('x_test.shape: ', x_test.shape)
    print('y_test.shape: ', y_test.shape)

    print('Input shape:', data_input.shape)
    print('Output shape:', expected_output.shape)
    print('Input head: ')
    print(data_input.head())
    print('Output head: ')
    print(expected_output.head())
    print('Input tail: ')
    print(data_input.tail())
    print('Output tail: ')
    print(expected_output.tail())

    # Create the model
    print('Creating Fully-Connected Model...')
    model_fc = create_fc_model()
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.00005)
    model_fc.compile(optimizer=rmsprop, loss='mean_squared_error')
    # Train the model
    print('Training')
    ##### TRAIN YOUR MODEL #####
    history = model_fc.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                           shuffle=False)

    # Plot and save loss curves of training and test set vs iteration in the same graph
    ##### PLOT AND SAVE LOSS CURVES #####
    loss = history.history['loss']
    test_loss = history.history['val_loss']
    #loss = np.sqrt(loss)
    #val_loss = np.sqrt(val_loss)
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, epochs+1), loss, label='train_loss')
    plt.plot(np.arange(1, epochs+1), test_loss, label='test_loss')
    plt.title('Loss vs Epochs in Training and Validation Set  Length:'+str(num_input))
    plt.xlabel('Epochs')
    plt.ylabel('Loss(MSE)')
    x_label = range(1,11)
    plt.xticks(x_label)
    plt.legend()
    plt.grid()
    #plt.savefig('/Users/martin_yan/Desktop/part1_%d.jpg' % length, dpi=200)
    plt.show()

    # Save your model weights with following convention:
    # For example length 1 input sequences model filename
    # fc_model_weights_length_1.h5
    ##### SAVE MODEL WEIGHTS #####
    filename = 'fc_model_weights_length_%d.h5' % length
    model_fc.save_weights(filename)

    # Predict
    print('Predicting')
    ##### PREDICT #####
    predicted_fc = model_fc.predict(x_test, batch_size=batch_size)
    ##### CALCULATE RMSE #####
    fc_rmse = np.sqrt(mean_squared_error(y_test, predicted_fc))
    fc_rmse_list.append(fc_rmse)

    # print('tsteps:{}'.format(tsteps))
    print('length:{}'.format(length))
    print('Fully-Connected RMSE:{}'.format(fc_rmse))

# save your rmse values for different length input sequence models:
filename = 'fc_model_rmse_values.txt'
np.savetxt(filename, np.array(fc_rmse_list), fmt='%.6f', delimiter='\t')

print("#" * 33)
print('Plotting Results')
print("#" * 33)

# Plot and save rmse vs Input Length
plt.figure()
plt.plot(np.arange(min_length,max_length+1), fc_rmse_list, c='black', label='FC')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('Length of Input Sequences')
x_label = range(1, 11)
plt.xticks(x_label)
plt.ylabel('RMSE')
plt.legend()
plt.grid()
#plt.savefig('/Users/martin_yan/Desktop/part1_rmse.jpg', dpi=200)
plt.show()


