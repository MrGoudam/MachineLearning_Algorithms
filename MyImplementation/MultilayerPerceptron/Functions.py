import numpy as np


def logistic(x):
    return 1 / (1 + np.exp(-x))


def gradient_logistic(x):
    f = logistic(x)
    return f * (1 - f)


def relu(x):
    return x * (x > 0)


def gradient_relu(x):
    return x > 0


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def gradient_softmax(x):
    smax = softmax(x)
    return smax * (1 - smax)


ACTIVATIONS = {'logistic': logistic,
               'relu': relu,
               'softmax': softmax}

GRADIENTS = {'logistic': gradient_logistic,
             'relu': gradient_relu,
             'softmax': gradient_softmax}