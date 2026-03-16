import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(output):
    return output * (1 - output)

def linear(x):
    return x
def linear_deriv(output):
    return np.ones_like(output)

def relu(x):
    return np.maximum(0, x)
def relu_deriv(output):
    return (output > 0).astype(float)