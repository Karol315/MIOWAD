import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(output):
    # Ważne: ta funkcja zakłada, że 'output' to już obliczona wartość sigmoid(x)
    return output * (1 - output)

def linear(x):
    return x
def linear_deriv(output):
    return np.ones_like(output)