import numpy as np


class Layer:
    def __init__(self, input_size, n_neurons, activation_func, activation_deriv, init_method="uniform"):
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.activation_func = activation_func
        self.activation_deriv = activation_deriv
        if init_method == "uniform":
            self.weights = np.random.uniform(0, 1, (input_size, n_neurons))
        elif init_method == "he":
            self.weights = np.random.randn(input_size, n_neurons) * np.sqrt(2.0 / input_size)
        elif init_method == "xavier":
            self.weights = np.random.randn(input_size, n_neurons) * np.sqrt(1.0 / input_size)
        else:
            raise ValueError("Nieznana metoda inicjalizacji")
        self.biases = np.zeros((1, n_neurons))
        self.last_input = None
        self.output = None

    def forward(self, inputs):
        self.last_input = inputs
        z = (inputs @ self.weights) + self.biases
        self.output = self.activation_func(z)
        return self.output

    def backward(self, d_output, learning_rate):
        delta = d_output * self.activation_deriv(self.output)
        d_weights = self.last_input.T @ delta
        d_biases = np.sum(delta, axis=0, keepdims=True)
        d_input = delta @ self.weights.T
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input