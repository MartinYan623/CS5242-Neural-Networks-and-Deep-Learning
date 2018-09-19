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

plt.set_cmap('gray')
# The input image is a 224x224x1 grayscale image
a = Input(shape=(224, 224, 1))
# Read Keras' document about Conv2D:
b = Conv2D(filters=1, kernel_size=(3, 3),strides=(1, 1), padding='valid', data_format='channels_last',
           dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(a)
# Combine things together
model = Model(inputs=a, outputs=b)

with open('../data/example.pkl', 'rb') as f:
    example = pk.load(f)
image = example['img']
image_f = example['img_f']
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(image_f)

# read documentation on Keras' optimizers
sgd = optimizers.SGD(lr=0.001, momentum=0.99, decay=0.0001, nesterov=False)
# read documentation on how a Keras model is compiled
model.compile(loss='mean_squared_error', optimizer='sgd')
# expand dimension of input data to make it of shape BxHxWx1,
# B is the batchsize, in our case it's 1.
x = np.expand_dims(np.expand_dims(image, 0), 3)
y = np.expand_dims(np.expand_dims(image_f, 0), 3)
model.fit(x=x, y=y, epochs=5000, verbose=0)

conv_weights = model.get_layer(index=1).get_weights()[0]
learned_filter = conv_weights[:, :, 0, 0]
fig, ax = plt.subplots()
plt.xticks([0, 1, 2])
plt.yticks([0, 1, 2])
cs = plt.imshow(learned_filter)
cbar = plt.colorbar(cs)
plt.show()

# you won't have real_filter in the problem set
real_filter = example['filter'][:, :, 0, 0]
err = np.linalg.norm(real_filter - learned_filter) / (real_filter.shape[0]** 2)
print('MSE between real filter and learned filter is {:.6f}'.format(err))
plt.subplot(121)
plt.imshow(real_filter)
plt.subplot(122)
plt.imshow(learned_filter)