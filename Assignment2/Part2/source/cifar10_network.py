from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout, Flatten,normalization,regularizers

def cifar10_network(input_shape=(32, 32, 3), class_num=10):
    """CIFAR CNN

    Keyword Arguments:
        input_shape {tuple} -- shape of input images. Should be (32,32,3) for CIFAR
        class_num {int} -- number of classes. Shoule be 10 for CIFAR10

    Returns:
        model -- keras.models.Model() object
    """

    im_input = Input(shape=input_shape)
    times = 0
    while times < 5:
        t = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', data_format='channels_last')(im_input)
        t = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='valid', data_format='channels_last')(t)
        #t= normalization.BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(t)
        t = Activation('relu')(t)
        t = Dropout(rate=0.5, seed=True)(t)
        t = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='valid', data_format='channels_last')(t)
        times = times + 1

    t = Flatten()(t)
    t = Dense(units=class_num)(t)
    output = Activation(activation='softmax')(t)
    model = Model(input=im_input, output=output)
    return model
