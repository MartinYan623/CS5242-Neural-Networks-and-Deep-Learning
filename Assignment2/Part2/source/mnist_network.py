from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout, Flatten , regularizers,normalization

def mnist_network(input_shape=(28, 28, 1), class_num=10):
    """MNIST CNN

    Keyword Arguments:
        input_shape {tuple} -- shape of input images. Should be (28,28,1) for MNIST
        class_num {int} -- number of classes. Shoule be 10 for MNIST

    Returns:
        model -- keras.models.Model() object
    """

    im_input = Input(shape=input_shape)
    times = 0
    while times < 5:
        t = Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='valid', data_format='channels_last',
                   kernel_initializer='random_normal',kernel_regularizer=regularizers.l2(0.01))(
            im_input)
        t = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='valid', data_format='channels_last',
                   kernel_initializer='random_normal', kernel_regularizer=regularizers.l2(0.01))(
            t)
        t = Activation('relu')(t)
        t = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='valid', data_format='channels_last')(t)
        t = Dropout(rate=0.6, seed=True)(t)
        times = times + 1

    t = Flatten()(t)
    t = Dense(units=512)(t)
    t = Activation(activation='relu')(t)
    t = Dense(units=class_num)(t)
    output = Activation(activation='softmax')(t)
    model = Model(input=im_input, output=output)
    return model