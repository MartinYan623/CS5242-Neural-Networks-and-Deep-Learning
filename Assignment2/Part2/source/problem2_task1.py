from assign2_utils_p2 import mnist_reader
from assign2_utils_p2 import cifar10_reader
import matplotlib.pyplot as plt
from keras import optimizers, losses
from example_network import example_network
from keras.models import model_from_json


# load data
train_x, train_y, test_x, test_y, class_name = mnist_reader()
plt.set_cmap('gray')
#plt.show(plt.imshow(train_x[0, :, :, 0]))
#print('The label is {}'.format(class_name[list(train_y[0]).index(1)]))
model = example_network(input_shape=(28, 28, 1))
sgd = optimizers.SGD(lr=0.01, momentum=0.95, decay=0.00005,nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['accuracy'])
model.fit(x=train_x, y=train_y, epochs=1, verbose=1)
loss, acc = model.evaluate(x=test_x, y=test_y)
print('Test accuracy is {:.4f}'.format(acc))

# load data
train_x, train_y, test_x, test_y, class_name = cifar10_reader()
#plt.show(plt.imshow(train_x[1,:,:,:]))
#print('The label is {}'.format(class_name[list(train_y[1]).index(1)]))
model = example_network(input_shape=(32, 32, 3))
sgd = optimizers.SGD(lr=0.01, momentum=0.95, decay=0.00005,nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['accuracy'])
model.fit(x=train_x, y=train_y, epochs=1, verbose=1)
loss, acc = model.evaluate(x=test_x, y=test_y)
print('Test accuracy is {:.4f}'.format(acc))

# save as jason file
json_string = model.to_json()
with open("A0174365Y.json", "w") as json_file:
    json_file.write(json_string)