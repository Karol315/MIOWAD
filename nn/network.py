import numpy as np
from .layer import Layer
from .activation_functions import *


class SimpleNetwork:
    def __init__(self, input_size,
                 output_size,
                 hidden_layers_count=None,
                 hidden_layers_sizes=None,
                 activation_func=sigmoid,
                 activation_deriv=sigmoid_deriv,
                 init_method="uniform"):
        self.layers = []

        final_sizes = []
        if hidden_layers_count is not None and hidden_layers_sizes is not None:
            if hidden_layers_count != len(hidden_layers_sizes):
                raise ValueError("The number of hidden layers must equal the number of hidden layers sizes.")
            final_sizes = hidden_layers_sizes
        elif hidden_layers_sizes is not None:
            final_sizes = hidden_layers_sizes
        elif hidden_layers_count is not None:
            final_sizes = [5] * hidden_layers_count

        current_input = input_size
        for n_neurons in final_sizes:
            self.layers.append(Layer(current_input, n_neurons, activation_func, activation_deriv, init_method))
            current_input = n_neurons
        self.layers.append(Layer(current_input, output_size, linear, linear_deriv, init_method))

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        current_output = np.asarray(inputs)
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def fit(self, X, y, epochs, learning_rate, batch_size=None):

        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples

        for epoch in range(epochs):
            # Przetasowanie danych
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                output = self.predict(X_batch)

                # Backward pass
                d_output = 2 * (output - y_batch) / len(X_batch)

                for layer in reversed(self.layers):
                    d_output = layer.backward(d_output, learning_rate)

    def fit_with_history(self, X, y, epochs, learning_rate, batch_size=None):
        """
        Wersja do analizy - uczy sieć i zapamiętuje wartości wag wszystkich warstw w każdej epoce.
        """
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples

        history = {
            'loss': [],
            'weights': {i: [] for i in range(len(self.layers))}
        }

        for epoch in range(epochs):
            # Przetasowanie danych
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                output = self.predict(X_batch)
                loss = np.mean((y_batch - output) ** 2)
                epoch_loss += loss * len(X_batch)

                # Backward pass
                d_output = 2 * (output - y_batch) / len(X_batch)
                for layer in reversed(self.layers):
                    d_output = layer.backward(d_output, learning_rate)


            history['loss'].append(epoch_loss / n_samples)
            for idx, layer in enumerate(self.layers):
                history['weights'][idx].append(layer.weights.copy())

        return history