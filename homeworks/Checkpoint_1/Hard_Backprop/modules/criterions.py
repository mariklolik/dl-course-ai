import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        assert input.shape == target.shape, 'input and target shapes not matching'
        return np.mean((input - target) ** 2)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert input.shape == target.shape, 'input and target shapes not matching'
        return 2.0 * (input - target) / input.size


class CrossEntropyLoss(Criterion):
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        log_probs = self.log_softmax(input)
        B = input.shape[0]
        return -np.mean(log_probs[np.arange(B), target.astype(int)])

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        B = input.shape[0]
        log_probs = self.log_softmax.output
        grad_log_probs = np.zeros_like(input)
        grad_log_probs[np.arange(B), target.astype(int)] = -1.0 / B
        return self.log_softmax.backward(input, grad_log_probs)
