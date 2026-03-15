import numpy as np
from .neuron import Neuron

class Layer():
    def __init__(self,
                 n_neurons: int,
                 input_size: int):

        self.n_neurons = n_neurons
        self.neurons = [Neuron(input_size=input_size) for _ in range(n_neurons)]

    def forward(self, inputs):
        self.outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return self.outputs


    def set_neuron_weights(self, neuron_index, weights):
        self.neurons[neuron_index].set_weights(weights)
    def set_neuron_bias(self, neuron_index, bias):
        self.neurons[neuron_index].set_bias(bias)

    def set_weights(self, weights_list : list[np.ndarray]):
        for neuron_index in range(len(self.neurons)):
            self.set_neuron_weights(neuron_index, weights_list[neuron_index])
    def set_bias(self, bias_list):
        for neuron_index in range(len(self.neurons)):
            self.set_neuron_bias(neuron_index, bias_list[neuron_index])

    def get_weights(self):
        weights = []
        for neuron in self.neurons:
            weights.append(neuron.weights)
        return weights
    def get_bias(self):
        bias = []
        for neuron in self.neurons:
            bias.append(neuron.bias)
        return bias

    def __str__(self):
        return f"Layer(n_neurons={self.n_neurons}, neurons = {str(self.neurons)}"