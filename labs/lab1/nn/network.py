import numpy as np
from .layer import Layer

class SimpleNetwork():
    def __init__(self,
                 input_size: int,
                 hidden_layers_sizes: list,
                 output_size: int = 1):

        self.input_size = input_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.output_size = output_size

        self.hidden_layers = []
        prev_size = input_size

        for size in hidden_layers_sizes:
            layer = Layer(n_neurons=size, input_size=prev_size)
            self.hidden_layers.append(layer)
            prev_size = size

        self.output_layer = Layer(n_neurons=output_size, input_size=prev_size)

    def predict(self, inputs):
        current_inputs = np.asarray(inputs)

        for layer in self.hidden_layers:
            current_inputs = layer.forward(current_inputs)

        output = self.output_layer.forward(current_inputs)
        return output

    def __str__(self):
        return f"SimpleNetwork(input_size={self.input_size}, hidden_layers_sizes={self.hidden_layers_sizes})"