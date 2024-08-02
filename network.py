import numpy as np
from activation import *
from layer import *
from loss import *
from gradient_descent import *
from save import *
import time

def train(X, y, epochs, batch_size, learning_rate):
    
    file_exists = False
    try:
        with open('trained_model', 'rb'):
            file_exists = True
    except FileNotFoundError:
        pass
    
    if file_exists:
        overwrite = input(f"The file 'trained_model' already exists. Do you want to overwrite it? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Training aborted. File not overwritten")
            return None
    
    start = time.time()

    layer1 = Layer(784, 128)
    activation1 = ReLU()
    layer2 = Layer(128, 10)
    activation2 = Softmax()
    loss_function = CrossEntropyLoss()
    gradient_descent = SGD(learning_rate)

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        epoch_loss = 0
        correct_predictions = 0
        total_predections = 0
        
        for batch_start in range(0, X.shape[0], batch_size):
            X_batch = X[batch_start : (batch_start + batch_size)]
            y_batch = y[batch_start : (batch_start + batch_size)]

            layer1.forward(X_batch)
            activation1.activate(layer1.output)
            layer2.forward(activation1.output)
            activation2.activate(layer2.output)

            loss = loss_function.loss_calclation(activation2.output, y_batch)
            predictions = np.argmax(activation2.output, axis=1)

            epoch_loss += loss * X_batch.shape[0]
            correct_predictions += np.sum(predictions == y_batch)
            total_predections += y_batch.shape[0]

            accuracy = np.mean(predictions == y_batch)

            loss_grad = loss_function.backward(activation2.output, y_batch)
            layer2.backward(loss_grad)
            activation1.backward(layer2.dinputs)
            layer1.backward(activation1.dinputs)

            gradient_descent.update_params(layer1)
            gradient_descent.update_params(layer2)

        if(total_predections != 0):
            print(f"Accurary for Epoch {epoch + 1}: {correct_predictions/total_predections}")
        else:
            print("0 Predictions Made this Epoch")

    save_model('trained_model', [layer1, activation1, layer2, activation2])

    end = time.time()
    print(f'Training took {end - start} seconds')

    return

def predict(X):
    
    layers = load_model('trained_model')
    layer1, activation1, layer2, activation2 = layers

    layer1.forward(X)
    activation1.activate(layer1.output)
    layer2.forward(activation1.output)
    activation2.activate(layer2.output)

    return np.argmax(activation2.output, axis=1)
