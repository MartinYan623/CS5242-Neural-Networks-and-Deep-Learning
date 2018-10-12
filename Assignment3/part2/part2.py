from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Activation,Embedding
from sklearn.metrics import mean_squared_error


# Create model
def create_rnn_model(stateful):
    ##### YOUR MODEL GOES HERE #####
    model = Sequential()
    model.add(SimpleRNN(20, stateful=stateful, return_sequences=False, batch_input_shape=(1,length,1), activation='relu'))
    model.add(Dense(1))

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
    x_train = x_train.reshape(to_train,length,1)
    y_train = y_train.values
    y_train = y_train.reshape(to_train, 1)

    x_test = x_test.values
    x_test = x_test.reshape(len(x.index) - to_train,length,1)
    y_test = y_test.values
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
rnn_stateful_rmse_list = list()
rnn_stateless_rmse_list = list()

for num_input in range(min_length, max_length + 1):
    length = num_input

    print("*" * 33)
    print("INPUT DIMENSION:{}".format(length))
    print("*" * 33)

    # convert numpy arrays to pandas dataframe
    data_input = pd.DataFrame(noisy_data)
    expected_output = pd.DataFrame(smooth_data)
    empty = pd.DataFrame(data=None, columns=range(0, length))
    # when length > 1, arrange input sequences
    if length > 1:
        # ARRANGE YOUR DATA SEQUENCES
        # lose L-1 input data
        for i in range(length - 1, len(data_input)):
            list = []
            for j in range(i, i - length, -1):
                list.append(data_input.iloc[j][0])
            empty.loc[i - 1] = list
        data_input = empty  # lose L-1 output data
        for i in range(length - 1):
            expected_output = expected_output.drop(i)

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

    # Create the stateful model
    print('Creating Stateful Vanilla RNN Model...')
    model_rnn_stateful = create_rnn_model(stateful=True)

    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.00005)
    model_rnn_stateful.compile(optimizer=rmsprop, loss='mean_squared_error')

    # Train the model
    print('Training')
    loss = []
    val_loss = []
    for i in range(epochs):
        print('Epoch', i + 1, '/', epochs)
        # Note that the last state for sample i in a batch will
        # be used as initial state for sample i in the next batch.

        ##### TRAIN YOUR MODEL #####
        history = model_rnn_stateful.fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_split=0.2, shuffle=False)
        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        # reset states at the end of each epoch
        model_rnn_stateful.reset_states()

    # Plot and save loss curves of training and test set vs iteration in the same graph
    ##### PLOT AND SAVE LOSS CURVES #####

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, epochs + 1), loss, label='train_loss')
    plt.plot(np.arange(1, epochs + 1), val_loss, label='val_loss')
    plt.title('Loss vs Iterations in Training and Validation Set for Stateful RNN  Length:' + str(num_input))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    x_label = range(1, 11)
    plt.xticks(x_label)
    plt.legend()
    plt.grid()
    plt.show()

    # Save your model weights with following convention:
    # For example length 1 input sequences model filename
    # rnn_stateful_model_weights_length_1.h5
    ##### SAVE MODEL WEIGHTS #####
    filename = 'rnn_stateful_model_weights_length_%d.h5' % num_input
    model_rnn_stateful.save_weights(filename)

    # Predict
    print('Predicting')
    ##### PREDICT #####
    predicted_rnn_stateful = model_rnn_stateful.predict(x_test, batch_size=batch_size)
    #predicted_rnn_stateful= predicted_rnn_stateful.reshape(len(data_input.index) - int(len(data_input.index) * 0.8), 1)

    ##### CALCULATE RMSE #####
    rnn_stateful_rmse = np.sqrt(mean_squared_error(y_test, predicted_rnn_stateful))
    rnn_stateful_rmse_list.append(rnn_stateful_rmse)

    # print('tsteps:{}'.format(tsteps))
    print('length:{}'.format(length))
    print('Stateful Vanilla RNN RMSE:{}'.format(rnn_stateful_rmse))

    # Create the stateless model
    print('Creating stateless Vanilla RNN Model...')
    model_rnn_stateless = create_rnn_model(stateful=False)
    model_rnn_stateless.compile(optimizer=rmsprop, loss='mean_squared_error')

    # Train the model
    print('Training')
    ##### TRAIN YOUR MODEL #####
    history = model_rnn_stateless.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=False)

    # Plot and save loss curves of training and test set vs iteration in the same graph
    ##### PLOT AND SAVE LOSS CURVES #####
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # loss = np.sqrt(loss)
    # val_loss = np.sqrt(val_loss)
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, epochs + 1), loss, label='train_loss')
    plt.plot(np.arange(1, epochs + 1), val_loss, label='val_loss')
    plt.title('Loss vs Iterations in Training and Validation Set for Stateless RNN Length:' + str(num_input))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    x_label = range(1, 11)
    plt.xticks(x_label)
    plt.legend()
    plt.grid()
    plt.show()

# Save your model weights with following convention:
    # For example length 1 input sequences model filename
    # rnn_stateless_model_weights_length_1.h5
    ##### SAVE MODEL WEIGHTS #####
    filename = 'rnn_stateless_model_weights_length_%d.h5' % num_input
    model_rnn_stateless.save_weights(filename)

    # Predict
    print('Predicting')
    ##### PREDICT #####
    predicted_rnn_stateless = model_rnn_stateless.predict(x_test, batch_size=batch_size)
    #predicted_rnn_stateless = predicted_rnn_stateless.reshape(len(data_input.index) - int(len(data_input.index) * 0.8), 1)
    ##### CALCULATE RMSE #####
    rnn_stateless_rmse = np.sqrt(mean_squared_error(y_test, predicted_rnn_stateless))
    rnn_stateless_rmse_list.append(rnn_stateless_rmse)

    # print('tsteps:{}'.format(tsteps))
    print('length:{}'.format(length))
    print('Stateless Vanilla RNN RMSE:{}'.format(rnn_stateless_rmse))

# save your rmse values for different length input sequence models - stateful rnn:
filename = 'rnn_stateful_model_rmse_values.txt'
np.savetxt(filename, np.array(rnn_stateful_rmse_list), fmt='%.6f', delimiter='\t')

# save your rmse values for different length input sequence models - stateless rnn:
filename = 'rnn_stateless_model_rmse_values.txt'
np.savetxt(filename, np.array(rnn_stateless_rmse_list), fmt='%.6f', delimiter='\t')

print("#" * 33)
print('Plotting Results')
print("#" * 33)

# Plot and save rmse vs Input Length
plt.figure()
plt.plot(np.arange(min_length, max_length + 1), rnn_stateful_rmse_list, c='blue', label='Stateful RNN')
plt.plot(np.arange(min_length, max_length + 1), rnn_stateless_rmse_list, c='cyan', label='Stateless RNN')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('Length of Input Sequences')
plt.ylabel('RMSE')
x_label = range(1, 11)
plt.xticks(x_label)
plt.legend()
plt.grid()
plt.show()
