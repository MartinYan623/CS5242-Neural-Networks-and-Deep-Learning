import numpy as np

class FullyConnectedLayer:
    def __init__(self, num_input, num_output, lr=1e-3, scale=2):
        # layer parameters
        self.W = np.random.randn(num_output, num_input) * np.sqrt(scale/(num_input+num_output)).astype(np.float128)
        #print(self.W)
        self.b = np.random.randn(num_output, 1) * np.sqrt(scale/(num_input+num_output)).astype(np.float128)
        # gradients and momentum
        self.gW = np.zeros(self.W.shape).astype(np.float128)
        self.gb = np.zeros(self.b.shape).astype(np.float128)
        self.gI = np.array([]).astype(np.float128)
        self.vW = np.zeros(self.W.shape).astype(np.float128)
        self.vb = np.zeros(self.b.shape).astype(np.float128)
        # layer input and output
        self.input_data = np.array([]).astype(np.float128)
        self.output_data = np.array([]).astype(np.float128)
        # learning rate
        self.lr = lr
    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = np.dot(self.W, input_data) + self.b
        return self.output_data
    def backward(self, gradient_data):
        self.gW = np.dot(gradient_data, np.transpose(self.input_data))
        self.gb = np.expand_dims(np.mean(gradient_data, axis=1), axis=1) if gradient_data.shape[1] > 1 else gradient_data
        self.gI = np.dot(np.transpose(self.W), gradient_data)
        return self.gI

def sigmoid(x):
    return 1/(1+np.exp(-x))

class SigmoidLayer:
    def __init__(self):
        self.gI = np.array([]).astype(np.float128)
        self.input_data = np.array([]).astype(np.float128)
        self.output_data = np.array([]).astype(np.float128)
    def forward(self, x):
        self.input_data = x
        self.output_data = sigmoid(x)
        return self.output_data
    def backward(self, gradient):
        self.gI = (gradient *
                  sigmoid(self.input_data) *
                  (1 - sigmoid(self.input_data))).astype(np.float128)
        return self.gI

class ReLULayer:
    def __init__(self):
        self.gI = np.array([]).astype(np.float128)
        self.input_data = np.array([]).astype(np.float128)
        self.output_data = np.array([]).astype(np.float128)
    def forward(self, x):
        self.input_data = x
        self.output_data = x.clip(0)
        return self.output_data
    def backward(self, gradient):
        self.gI = gradient * (self.input_data > 0).astype(np.float128)
        return self.gI

class LeakyReLULayer:
    def __init__(self, epsilon=0.01):
        self.gI = np.array([]).astype(np.float128)
        self.input_data = np.array([]).astype(np.float128)
        self.output_data = np.array([]).astype(np.float128)
        self.epsilon = epsilon
    def forward(self, x):
        self.input_data = x
        self.output_data = x * (x > 0) + x * (x<0) * self.epsilon
        return self.output_data
    def backward(self, gradient):
        self.gI = gradient * ((self.input_data > 0) + self.epsilon*(self.input_data < 0)).astype(np.float128)
        return self.gI

# Not robust, not used
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def softmax_robust(x):
    result = np.zeros(x.shape, dtype=np.float128)
    for q in range(x.shape[1]):
        for p in range(0, x.shape[0]):
            result[p][q] = 1/np.sum(np.exp(x[:,q] - x[p,q]))
    return result

class SoftmaxOutput_CrossEntropyLossLayer:
    def __init__(self):
        self.gI = np.array([]).astype(np.float128)
        self.x = np.array([]).astype(np.float128)
        self.y_pred = np.array([]).astype(np.float128)
        self.y_label = np.array([]).astype(np.float128)
        self.output_data = np.array([]).astype(np.float128)
    def eval(self, x, y_label, phase='train'):
        self.x = x
        self.y_pred = softmax_robust(x.astype(np.float128))
        self.y_label = y_label
        #print(self.y_pred)
        if phase is 'train':
            self.output_data = np.sum(- y_label * np.log(self.y_pred.clip(1e-30))) / self.y_label.shape[1]
        elif phase is 'test':
            self.output_data = np.array([])
        return self.output_data
    def backward(self):
        self.gI = self.y_pred - self.y_label
        return self.gI
