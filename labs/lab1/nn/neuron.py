import numpy as np
from .activation_functions import sigmoid

class Neuron():
    def __init__(self,
                 input_size: int,
                 activation=sigmoid):

        self.weights = np.random.rand(input_size)
        self.bias = 0.0
        self.activation = activation
        self.output = None

    def forward(self, inputs):
        inputs = np.asarray(inputs)
        self.output = self.activation(np.dot(self.weights, inputs) + self.bias)
        return self.output

    def set_weights(self, weights):
        self.weights = weights

    def set_bias(self, bias):
        self.bias = bias

    def set_activation(self, activation):
        self.activation = activation

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def get_activation(self):
        return self.activation

    def __str__(self):
        return f"Neuron(weights={self.weights}, bias={self.bias}, activation={self.activation.__name__})"