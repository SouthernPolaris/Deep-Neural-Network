import numpy as np

class ReLU:
    def activate(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Softmax:
    def activate(self, inputs):
        exponentiated = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exponentiated / np.sum(exponentiated, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()