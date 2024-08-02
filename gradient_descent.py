class SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        layer.neuronWeights -= self.learning_rate * layer.dweights
        layer.neuronBiases -= self.learning_rate * layer.dbiases
