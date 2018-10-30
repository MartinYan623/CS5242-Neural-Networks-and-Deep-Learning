import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from sklearn.metrics import mean_squared_error
from keras.optimizers import RMSprop, Adam
import heapq
from preprocess import *
from pretest import *

# Create model
def create_lstm_model(stateful):
    ##### YOUR MODEL GOES HERE #####
    model = Sequential()
    model.add(Masking(mask_value=-999, batch_input_shape=(1, 50, 4)))
    model.add(
        LSTM(20, stateful=stateful, return_sequences=False, batch_input_shape=(1, 50, 4)))
    model.add(Dense(1))
    return model


if __name__ == '__main__':

    with open('../data/middle_data/train_input.bin', 'rb') as f:
        train_input = np.array(pickle.load(f))
    with open('../data/middle_data/train_output.bin', 'rb') as f:
        y_train = np.array(pickle.load(f))

    with open('../data/middle_data/tree_list.bin', 'rb') as f:
         tree_list = pickle.load(f)
    print('Tree info loaded successfully!')

    valid_input, y_valid = create_mlp_valid(tree_list, 2700)

    with open('../data/middle_data/tree_list_test.bin', 'rb') as f:
         tree_list_test = pickle.load(f)
    print('Tree info loaded successfully!')

    test_input = create_mlp_test(tree_list_test)

    x_train =[]
    for i in range(len(train_input)):
        temp = train_input[i][0]
        for j in range(len(train_input[i])):
            temp = np.concatenate((temp, train_input[i][j]), axis=0)
        x_train.append(temp)
    x_train = np.array(x_train)

    x_valid = []
    for i in range(len(valid_input)):
        temp = valid_input[i][0]
        for j in range(len(valid_input[i])):
            temp = np.concatenate((temp, valid_input[i][j]), axis=0)
        x_valid.append(temp)
    x_valid = np.array(x_valid)

    x_test = []
    for i in range(len(test_input)):
        temp = test_input[i][0]
        for j in range(len(test_input[i])):
            temp = np.concatenate((temp, test_input[i][j]), axis=0)
        x_test.append(temp)
    x_test = np.array(x_test)

    # padding sequence
    x_train = sequence.pad_sequences(x_train, maxlen=50, padding='post', dtype=float, value=-999)
    x_valid = sequence.pad_sequences(x_valid, maxlen=50, padding='post', dtype=float, value=-999)
    x_test = sequence.pad_sequences(x_test, maxlen=50, padding='post', dtype=float, value=-999)
    y_valid = np.array(y_valid)

    # create model
    model = create_lstm_model(stateful=False)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00005)
    model.compile(loss='mean_squared_error', optimizer=adam)
    hist = model.fit(x=x_train, y=y_train, epochs=1, batch_size=1, verbose=1, validation_data=(x_valid, y_valid))

    # plot validation result
    # df = DataFrame(data=d)
    # epoch = df.index.map(lambda x: x + 1)
    # df['epoch'] = epoch
    # plot_hist(df, 'lstm', 'acc')
    # plot_hist(df, 'lstm', 'loss')

    # predict testing data
    predicted_lstm = model.predict(x_test, batch_size=1)

    # get the prediction result of testing data set
    predicted_lstm = predicted_lstm.reshape(824, 824)
    print('The result is:')
    print(predicted_lstm)

    # save as txt file
    result = []
    for i in range(len(predicted_lstm)):
        a = np.array(predicted_lstm[i, :])
        line = (heapq.nlargest(10, range(len(a)), a.take))
        # nlargest function returns value from 0, add 1 to change to start from 1
        nl = [i + 1 for i in line]
        nl.append(int(i + 1))
        nl.reverse()
        result.append(nl)
    np.savetxt('../data/result/test_predictions_lstm.txt', result, delimiter='\t', newline='\n', comments='',
               header='pro_id\tlig1_id\tlig2_id\tlig3_id\tlig4_id\tlig5_id\tlig6_id\tlig7_id\tlig8_id\tlig9_id\tlig10_id',
               fmt='%d')
