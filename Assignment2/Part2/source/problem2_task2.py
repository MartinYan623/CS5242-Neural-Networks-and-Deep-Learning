from assign2_utils_p2 import mnist_reader
from assign2_utils_p2 import cifar10_reader
import matplotlib.pyplot as plt
from keras import optimizers, losses
from keras.models import model_from_json
from mnist_network import mnist_network
from cifar10_network import cifar10_network


# load data
train_x, train_y, test_x, test_y, class_name = mnist_reader()
plt.set_cmap('gray')
#plt.show(plt.imshow(train_x[0, :, :, 0]))
#print('The label is {}'.format(class_name[list(train_y[0]).index(1)]))
model = mnist_network(input_shape=(28, 28, 1))
adam= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer=adam,metrics=['accuracy'])
model.fit(x=train_x, y=train_y, epochs=1, verbose=1)
loss, acc = model.evaluate(x=test_x, y=test_y)
print('Test accuracy is {:.4f}'.format(acc))


# load data
train_x, train_y, test_x, test_y, class_name = cifar10_reader()
#plt.show(plt.imshow(train_x[1,:,:,:]))
#print('The label is {}'.format(class_name[list(train_y[1]).index(1)]))
model = cifar10_network(input_shape=(32, 32, 3))
sgd = optimizers.SGD(lr=0.01, momentum=0.95, decay=0.00005,nesterov=False)
adam= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer=adam,metrics=['accuracy'])
model.fit(x=train_x, y=train_y, epochs=1, verbose=1)
loss, acc = model.evaluate(x=test_x, y=test_y)
print('Test accuracy is {:.4f}'.format(acc))


# save as HDF5 file
model.save('A0174365Y_mnist.h5')
model.save('A0174365Y_cifar10.h5')

