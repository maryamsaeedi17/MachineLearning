import numpy as np

class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.w=None
        self.b=None
        self.learning_rate=learning_rate
        self.epochs=epochs

    def fit(self, X_train, Y_train):
        ...
​
    def predict(self, X_test):
        ...
​
    def evaluate(self, X_test, Y_test):
        ...