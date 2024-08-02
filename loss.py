import numpy as np

class CrossEntropyLoss:
    def loss_calclation(self, predictions, labels):
        samples = labels.shape[0]
        prediction_clip = np.clip(predictions, 1e-15, 1 - 1e-15)
        confidence_positive = prediction_clip[range(samples), labels]
        loss = -np.log(confidence_positive)
        return np.mean(loss)
    
    def backward(self, predictions, labels):
        samples = labels.shape[0]
        grad = predictions.copy()
        grad[range(samples), labels] -= 1
        grad = grad / samples
        return grad