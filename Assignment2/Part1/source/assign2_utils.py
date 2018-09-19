import numpy as np
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)

import keras
from keras.models import Model
from keras.layers import Input, Conv2D
from matplotlib import pyplot as plt
from PIL import Image

import pickle as pk

def load_image(path):
    im = Image.open(path)
    im = im.resize((224,224), Image.ANTIALIAS)
    im = im.convert('L')
    return np.array(im) / 255.0

def save_image(path, arr):
    plt.imsave(path, arr)
    
def test_model_with_weight(weights, x, conv_shape=(3,3)):
    """Test the weights you get using input image
    
    Arguments:
        weights {np.array} -- filter weights obtained from the conv layer, should have shape (3,3,1,1)
        x {np.array} -- test image in shape (B,H,W,C), in our case B=1, H=W=224, C=1
    
    Returns:
        y_pred -- filtered image
    """
    a = Input(shape=(224,224,1))
    b = Conv2D(1, conv_shape, padding='valid', use_bias=False)(a)
    model = Model(inputs=a, outputs=b)
    model.get_layer(index=1).set_weights([weights])
    y_pred = model.predict(x)
    return y_pred

def validate(answer_file):
    with open('../data/val.pkl', 'rb') as f:
        val = pk.load(f)
    with open(answer_file, 'rb') as f:
        ans = pk.load(f)
    x = np.expand_dims(np.expand_dims(val['val_img'],0),3)
    weights = ans['answer']['filter']
    val_img_output = []
    for i in range(0, 3):
        val_img_output.append(test_model_with_weight(np.expand_dims(np.expand_dims(weights[i],2),3), x, conv_shape=weights[i].shape))
    plt.set_cmap('gray')
    plt.subplot(231)
    plt.imshow(val_img_output[0][0,:,:,0])
    plt.subplot(232)
    plt.imshow(val_img_output[1][0,:,:,0])
    plt.subplot(233)
    plt.imshow(val_img_output[2][0,:,:,0])
    plt.subplot(234)
    plt.imshow(val['val_img_f'][0][0,:,:,0])
    plt.subplot(235)
    plt.imshow(val['val_img_f'][1][0,:,:,0])
    plt.subplot(236)
    plt.imshow(val['val_img_f'][2][0,:,:,0])
    plt.show()