import numpy as np


class Layer:
    def __init__(self,
                 input_size,
                 n_neurons,
                 activation_func,
                 activation_deriv,
                 init_method="uniform",
                 random_state=42
                 ):

        self.input_size = input_size
        self.n_neurons = n_neurons
        self.activation_func = activation_func
        self.activation_deriv = activation_deriv
        self.biases = np.zeros((1, n_neurons))
        self.last_input = None
        self.output = None

        np.random.seed(random_state)
        if init_method == "uniform":
            self.weights = np.random.uniform(0, 1, (input_size, n_neurons))
        elif init_method == "he":
            self.weights = np.random.randn(input_size, n_neurons) * np.sqrt(2.0 / input_size)
        elif init_method == "xavier":
            self.weights = np.random.randn(input_size, n_neurons) * np.sqrt(1.0 / input_size)
        else:
            raise ValueError("Nieznana metoda inicjalizacji")

        self.v_w = np.zeros_like(self.weights)
        self.v_b = np.zeros_like(self.biases)

        # Zmienne dla RMSprop (bufor średniej kwadratów gradientów)
        self.mean_square_w = np.zeros_like(self.weights)
        self.mean_square_b = np.zeros_like(self.biases)


    def forward(self, inputs):
        self.last_input = inputs
        z = (inputs @ self.weights) + self.biases
        self.output = self.activation_func(z)
        return self.output

    def backward(self, d_output, learning_rate, mode="simple",  momentum=0.9, decay_rate=0.9):
        delta = d_output * self.activation_deriv(self.output)
        d_weights = self.last_input.T @ delta
        d_biases = np.sum(delta, axis=0, keepdims=True)
        d_input = delta @ self.weights.T

        if mode == "simple":
            self.weights -= learning_rate * d_weights
            self.biases -= learning_rate * d_biases

        elif mode == "momentum":
            self.v_w = momentum * self.v_w + learning_rate * d_weights
            self.v_b = momentum * self.v_b + learning_rate * d_biases
            self.weights -= self.v_w
            self.biases -= self.v_b

        elif mode == "rmsprop":
            epsilon = 1e-6
            self.mean_square_w = decay_rate * self.mean_square_w + (1 - decay_rate) * (d_weights ** 2)
            self.mean_square_b = decay_rate * self.mean_square_b + (1 - decay_rate) * (d_biases ** 2)

            self.weights -= learning_rate * d_weights / (np.sqrt(self.mean_square_w) + epsilon)
            self.biases -= learning_rate * d_biases / (np.sqrt(self.mean_square_b) + epsilon)

        return d_input