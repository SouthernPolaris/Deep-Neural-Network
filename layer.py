import numpy as np

class Layer:
    def __init__(self, numInputs, numNeurons):
        self.numInputs = numInputs
        self.numNeurons = numNeurons
        self.neuronWeights = 0.1 * np.random.randn(numInputs, numNeurons)
        self.neuronBiases = np.zeros((1, numNeurons), dtype=float, order='C')

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.matmul(inputs, self.neuronWeights) + self.neuronBiases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.neuronWeights.T)