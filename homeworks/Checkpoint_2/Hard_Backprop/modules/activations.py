import numpy as np
from scipy.special import expit, log_softmax, softmax
from .base import Module


class ReLU(Module):
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(input, 0)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (input > 0).astype(input.dtype)


class Sigmoid(Module):
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        return expit(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        s = expit(input)
        return grad_output * s * (1 - s)


class Softmax(Module):
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        return softmax(input, axis=-1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        s = softmax(input, axis=-1)
        return s * (grad_output - (grad_output * s).sum(axis=-1, keepdims=True))


class LogSoftmax(Module):
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        return log_softmax(input, axis=-1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        s = softmax(input, axis=-1)
        return grad_output - s * grad_output.sum(axis=-1, keepdims=True)
