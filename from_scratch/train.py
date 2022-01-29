import numpy as np
from data_preparation import *
from utils import CBOW
from train import *
import pprint

hyperparameters = {'window_size': 2, 'n': 10, 'epochs': 10, 'learning_rate': 0.1}

if __name__ == '__main__':
    text = read_data('doc.txt')
    tokens = tokenize(text)
    X, y = generate_vectors(tokens, windows_size=2)
    cbow = CBOW(5, 50)
    cbow.fit(X, y, epochs=20, learning_rate=0.01)