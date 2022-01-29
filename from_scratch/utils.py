from turtle import back, backward, forward
import numpy as np
from data_preparation import *

def softmax(z):
    e_x = np.exp(z - np.max(z))
    return e_x / e_x.sum(axis=0)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0) * 1

class CBOW:
    def __init__(self, word_vocab: int, word_emb: int):
        self.params = {}
        self.params['W1'] = np.random.randn(word_emb, word_vocab) * 0.01
        self.params['W2'] = np.random.randn(word_vocab, word_emb) * 0.01
        self.params['b1'] = np.zeros((1, word_emb))
        self.params['b2'] = np.zeros((1, word_vocab))

    def forward(self, x, y):
        """
        Input layer -> Hidden layer -> Output layer
        """
        self.m = len(x)
        z1 = x.dot(self.params['W1'].T) + self.params['b1']
        a1 = relu(z1)
        z2 = a1.dot(self.params['W2'].T) + self.params['b2']
        y_hat = softmax(z2)
        # Cross entropy loss
        J = np.sum(-y * np.log(y_hat)) / self.m
        cache = (z1, a1, x, y, y_hat)
        return J, cache

    def backward(self, cache):
        z1, a1, x, y, y_hat = cache
        self.grads = {}
        self.grads['W2'] = (y_hat - y).T.dot(a1)
        #print(self.params['W2'].shape, self.grads['W2'].shape)
        self.grads['b2'] = np.sum(y_hat - y, axis=0, keepdims=True)
        #print(self.params['b2'].shape, self.grads['b2'].shape)
        self.grads['W1'] = (y_hat - y).dot(self.params['W2']).T.dot(x)
        #print(self.params['W1'].shape, self.grads['W1'].shape)
        self.grads['b1'] = np.sum((y_hat - y).dot(self.params['W2']), axis=0, keepdims=True)
        #print(self.params['b1'].shape, self.grads['b1'].shape)

    def step(self, learning_rate):
        self.params['W1'] -= learning_rate * self.grads['W1'] / self.m
        self.params['W2'] -= learning_rate * self.grads['W2'] / self.m
        self.params['b1'] -= learning_rate * self.grads['b1'] / self.m
        self.params['b2'] -= learning_rate * self.grads['b2'] / self.m

    def fit(self, x, y, epochs=20, learning_rate=0.1):
        for e in range(epochs):
            # Forwad propagation
            cost, cache = self.forward(x, y)
            # Backward propagation
            self.backward(cache)
            # Update the parameters
            self.step(learning_rate)
            print(f"Epoch {e+1}: Cost: {cost}")
        
