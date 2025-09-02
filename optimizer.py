import torch

class Optimizer_simple:

    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad

    def zero_grad(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.zero_()