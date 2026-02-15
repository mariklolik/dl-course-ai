import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        output = input @ self.weight.T
        if self.bias is not None:
            output += self.bias
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        return grad_output @ self.weight

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        self.grad_weight += grad_output.T @ input
        if self.bias is not None:
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        self.mean = None
        self.input_mean = None
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        if self.training:
            B = input.shape[0]
            self.mean = input.mean(axis=0)
            self.var = ((input - self.mean) ** 2).mean(axis=0)
            self.input_mean = input - self.mean
            self.sqrt_var = np.sqrt(self.var + self.eps)
            self.inv_sqrt_var = 1.0 / self.sqrt_var
            self.norm_input = self.input_mean * self.inv_sqrt_var

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * (B / (B - 1)) * self.var

            output = self.norm_input
        else:
            output = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)

        if self.affine:
            output = output * self.weight + self.bias
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        if self.training:
            B = input.shape[0]
            if self.affine:
                grad_output = grad_output * self.weight

            grad_input = (1.0 / B) * self.inv_sqrt_var * (
                B * grad_output
                - grad_output.sum(axis=0)
                - self.norm_input * (grad_output * self.norm_input).sum(axis=0)
            )
            return grad_input
        else:
            if self.affine:
                return grad_output * self.weight / np.sqrt(self.running_var + self.eps)
            else:
                return grad_output / np.sqrt(self.running_var + self.eps)

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        if self.affine:
            if self.training:
                self.grad_weight += (grad_output * self.norm_input).sum(axis=0)
            else:
                norm_input = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)
                self.grad_weight += (grad_output * norm_input).sum(axis=0)
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = (np.random.random(input.shape) >= self.p).astype(input.dtype)
            return self.mask * input / (1 - self.p)
        else:
            return input

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        if self.training:
            return self.mask * grad_output / (1 - self.p)
        else:
            return grad_output

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        output = input
        for module in self.modules:
            output = module(output)
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        inputs = [input]
        for module in self.modules[:-1]:
            inputs.append(module.output)

        for module, inp in zip(reversed(self.modules), reversed(inputs)):
            grad_output = module.backward(inp, grad_output)
        return grad_output

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
