import numpy as np
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)
import keras
from keras.models import Model
from keras.layers import Input, Conv2D
from matplotlib import pyplot as plt
from PIL import Image
from keras import optimizers, losses
import pickle as pk
from assign2_utils import validate
import pickle as pk

with open('../data/problems.pkl', 'rb') as f:
    problems = pk.load(f)
image_1 = problems['img'][0]
image_f_1 = problems['img_f'][0]
image_2 = problems['img'][1]
image_f_2 = problems['img_f'][1]
image_3 = problems['img'][2]
image_f_3 = problems['img_f'][2]

plt.set_cmap('gray')
plt.subplot(231)
plt.imshow(image_1)
plt.subplot(234)
plt.imshow(image_f_1)
plt.subplot(232)
plt.imshow(image_2)
plt.subplot(235)
plt.imshow(image_f_2)
plt.subplot(233)
plt.imshow(image_3)
plt.subplot(236)
plt.imshow(image_f_3)

# The input image is a 224x224x1 grayscale image
a = Input(shape=(224, 224, 1))
# Read Keras' document about Conv2D:
b = Conv2D(filters=1, kernel_size=(3, 3),strides=(1, 1), padding='valid', data_format='channels_last',
           dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(a)
c = Conv2D(filters=1, kernel_size=(5, 5),strides=(1, 1), padding='valid', data_format='channels_last',
           dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(a)
# Combine things together
model1 = Model(inputs=a, outputs=b)
model2 = Model(inputs=a, outputs=c)

# read documentation on Keras' optimizers
sgd = optimizers.SGD(lr=0.001, momentum=0.99, decay=0.0001, nesterov=False)
rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0001)
adam= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)
# read documentation on how a Keras model is compiled
model1.compile(loss='mean_squared_error', optimizer=sgd)
model2.compile(loss='mean_squared_error', optimizer=sgd)
# expand dimension of input data to make it of shape BxHxWx1,
# B is the batchsize, in our case it's 1.
x_1 = np.expand_dims(np.expand_dims(image_1, 0), 3)
y_1 = np.expand_dims(np.expand_dims(image_f_1, 0), 3)
x_2 = np.expand_dims(np.expand_dims(image_2, 0), 3)
y_2 = np.expand_dims(np.expand_dims(image_f_2, 0), 3)
x_3 = np.expand_dims(np.expand_dims(image_3, 0), 3)
y_3 = np.expand_dims(np.expand_dims(image_f_3, 0), 3)

model1.fit(x=x_1, y=y_1, epochs=10000, verbose=0)
conv_weights = model1.get_layer(index=1).get_weights()[0]
first_filter = conv_weights[:, :, 0, 0]

model1.fit(x=x_2, y=y_2, epochs=10000, verbose=0)
conv_weights = model1.get_layer(index=1).get_weights()[0]
second_filter = conv_weights[:, :, 0, 0]

model2.fit(x=x_3, y=y_3, epochs=10000, verbose=0)
conv_weights = model2.get_layer(index=1).get_weights()[0]
third_filter = conv_weights[:, :, 0, 0]

fig, ax = plt.subplots()
plt.xticks([0, 1, 2])
plt.yticks([0, 1, 2])


cs_1 = plt.imshow(first_filter)
cbar_1 = plt.colorbar(cs_1)
plt.title('first filter')
plt.show()

cs_2 = plt.imshow(second_filter)
cbar_2 = plt.colorbar(cs_2)
plt.title('second filter')
plt.show()

cs_3 = plt.imshow(third_filter)
cbar_3 = plt.colorbar(cs_3)
plt.title('third filter')
plt.show()



answer = {
    # your name as shown in IVLE
    'Name': 'YAN MAITONG',
    # your Matriculation Number, starting with letter 'A'
    'MatricNum': 'A0174365Y',
    # do check the size of filters
    'answer': {'filter': [first_filter, second_filter, third_filter]}
}

with open('A0174365Y.pkl', 'wb') as f:
    pk.dump(answer, f)

validate('A0174365Y.pkl')

