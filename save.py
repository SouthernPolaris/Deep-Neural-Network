import pickle

def save_model(filename, layers):
    with open(filename, 'wb') as f:
        pickle.dump(layers, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        layers = pickle.load(f)
    return layers