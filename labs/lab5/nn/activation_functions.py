import numpy as np

class Activation:
    def func(self, x):
        raise NotImplementedError("Metoda 'func' musi zostać nadpisana.")

    def deriv(self, out):
        raise NotImplementedError("Metoda 'deriv' musi zostać nadpisana.")

    def __call__(self, x):
        return self.func(x)


class Sigmoid(Activation):
    def func(self, x):
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def deriv(self, out):
        return out * (1 - out)


class ReLU(Activation):
    def func(self, x):
        return np.maximum(0, x)

    def deriv(self, out):
        return (out > 0).astype(float)


class Linear(Activation):
    def func(self, x):
        return x

    def deriv(self, out):
        return np.ones_like(out)


class Softmax(Activation):
    def func(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def deriv(self, out):
        # tylko dla Cross-Entropy jako loss
        return np.ones_like(out)

class Tanh(Activation):
    def func(self, x):
        return np.tanh(x)

    def deriv(self, out):
        return 1 - out ** 2