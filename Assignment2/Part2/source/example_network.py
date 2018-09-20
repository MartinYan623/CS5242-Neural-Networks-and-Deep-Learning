import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout, Flatten

def example_network(input_shape=(28,28,1), class_num=10):
    """Example CNN
    
    Keyword Arguments:
        input_shape {tuple} -- shape of input images. Should be (28,28,1) for MNIST and (32,32,3) for CIFAR (default: {(28,28,1)})
        class_num {int} -- number of classes. Shoule be 10 for both MNIST and CIFAR10 (default: {10})
    
    Returns:
        model -- keras.models.Model() object
    """

    im_input = Input(shape=input_shape)
    times=0
    while times<3:
        t = Conv2D(filters=32, kernel_size=(3,3),strides=(1,1),padding='valid',data_format='channels_last')(im_input)
        t = Activation('relu')(t)
        t = MaxPool2D(pool_size=(2,2),strides=(1,1), padding='valid', data_format='channels_last')(t)
        t = Dropout(rate=0.5, seed=True)(t)
        times=times+1

    t = Flatten()(t)
    t = Dense(units=512)(t)
    t = Activation(activation='relu')(t)
    t = Dense(units=class_num)(t)
    output = Activation(activation='softmax')(t)
    model = Model(input=im_input, output=output)
    return model

