import numpy as np

def linear(x: np.ndarray) -> np.ndarray:
    return x

def dlinear(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def drelu(x: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(x)
    grad[x > 0] = 1
    return grad

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x: np.ndarray) -> np.ndarray:
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def dtanh(x: np.ndarray) -> np.ndarray:
    return (2 / (np.exp(x) + np.exp(-x))) ** 2

def softmax(x: np.ndarray) -> np.ndarray:
    exps = np.exp(x)
    return exps / np.sum(exps)

def dsoftmax(x: np.ndarray) -> np.ndarray:
    s = softmax(x)
    return np.diag(s) - np.outer(s, s)