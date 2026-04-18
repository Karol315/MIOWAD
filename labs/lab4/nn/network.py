import numpy as np
from .layer import Layer
from .activation_functions import Softmax, Sigmoid, ReLU, Linear


class Network:
    def __init__(self, layers, task="classification"):
        """
        Inicjalizuje sieć na podstawie przekazanej listy warstw.

        Przykład:
        net = Network([
                        Layer(2, 16, ReLU(), init_method="he"),
                        Layer(16, 3, Softmax(), init_method="he")
                    ],
                    task="classification")
        """
        self.layers = layers
        self.task = task

        last_activation = self.layers[-1].activation

        if self.task == "classification":
            if not isinstance(last_activation, (Softmax, Sigmoid)):
                raise ValueError("W trybie klasyfikacji ostatnia warstwa musi używać funkcji Softmax lub Sigmoid.")
        elif self.task == "regression":
            if not isinstance(last_activation, (Linear, Sigmoid, ReLU)):
                raise ValueError("W trybie regresji ostatnia warstwa musi używać funkcji Linear, Sigmoid lub ReLU.")
        else:
            raise ValueError("Nieznany parametr 'task'. Użyj 'classification' lub 'regression'.")

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        current_output = np.asarray(inputs)
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def fit(self, X, y, epochs, learning_rate, batch_size=None, mode="simple", momentum=0.9, decay_rate=0.9):
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward
                output = self.predict(X_batch)

                # Backward
                if self.task == "classification":
                    d_output = (output - y_batch) / len(X_batch)        # Pochodna dla Cross-Entropy z Softmax/Sigmoid
                else:
                    d_output = 2 * (output - y_batch) / len(X_batch)    # Pochodna dla MSE

                for layer in reversed(self.layers):
                    d_output = layer.backward(d_output, learning_rate, mode=mode, momentum=momentum,
                                              decay_rate=decay_rate)


    def fit_with_history(self, X, y, epochs, learning_rate, batch_size=None, mode="simple", momentum=0.9,
                         decay_rate=0.9):
        """
        Wersja do analizy, fit z zapamiętaniem wartości wag wszystkich warstw oraz błędu w każdej epoce.
        """
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples

        history = {
            'loss': [],
            'weights': {i: [] for i in range(len(self.layers))}
        }

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward
                output = self.predict(X_batch)

                # Backward
                if self.task == "classification":
                    loss = -np.sum(y_batch * np.log(output + 1e-9)) / len(X_batch)  # Cross-Entropy
                    d_output = (output - y_batch) / len(X_batch)
                else:
                    loss = np.mean((y_batch - output) ** 2)                         # MSE
                    d_output = 2 * (output - y_batch) / len(X_batch)

                epoch_loss += loss * len(X_batch)

                for layer in reversed(self.layers):
                    d_output = layer.backward(d_output, learning_rate, mode=mode, momentum=momentum,
                                              decay_rate=decay_rate)

            history['loss'].append(epoch_loss / n_samples)
            for idx, layer in enumerate(self.layers):
                history['weights'][idx].append(layer.weights.copy())

        return history