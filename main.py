import numpy as np
from matplotlib import pyplot as plt
from network import *

def load_data(file_path):
    with open(file_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

train_images = load_data('archive/train-images.idx3-ubyte')
train_labels = load_labels('archive/train-labels.idx1-ubyte')

test_images = load_data('archive/t10k-images.idx3-ubyte')
test_labels = load_labels('archive/t10k-labels.idx1-ubyte')


print(f'Training data shape: {train_images.shape}, Training labels shape: {train_labels.shape}')
print(f'Test data shape: {test_images.shape}, Test labels shape: {test_labels.shape}')


train_images = train_images.reshape(train_images.shape[0], -1) / 255.0
test_images = test_images.reshape(test_images.shape[0], -1) / 255.0

train(train_images, train_labels, epochs=2, batch_size=32, learning_rate=0.1)

def plot_test_predictions():
    num_images = 5
    idx = np.random.choice(test_images.shape[0], num_images, replace=False)

    if(num_images > 20):
        print("Too many plots to display")
        return
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    for i, ax in zip(idx, axes):
        image = test_images[i].reshape(28, 28)
        prediction = predict(test_images[i : i+1])[0]
        true_label = test_labels[i]

        ax.imshow(image, cmap='gray')
        ax.set_title(f'Prediction: {prediction}, True Label: {true_label}')
        ax.set_xlabel(f"True Label: {true_label}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

plot_test_predictions()