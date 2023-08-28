import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_length, learning_rate, epochs):
        self.w=np.random.rand(input_length)
        self.b=np.random.rand(1)
        self.learning_rate=learning_rate
        self.epochs=epochs

    def activation_function(self, X, func_name):
        if func_name == "sigmoid":
            return 1 / (1 + np.exp(-X))
        elif func_name == "relu":
            return np.maximum(0, X)
        elif func_name == "tanh":
            return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
        elif func_name == "linear":
            return X
        elif func_name == "binarystep":
            return np.heaviside(X, 1)
        elif func_name == "softmax":
            return np.exp(X) / np.sum(np.exp(X))

    def predict(self, X_test):
        Y_pred = []
        for x in X_test:
            y_pred = x @ self.weights + self.bias
            y_pred = self.activation(y_pred, "sigmoid")
            y_pred = np.where(y_pred > 0.5, 1, 0)
            Y_pred.append(y_pred)
        return np.array(Y_pred)

    def evaluate(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        loss = np.sum(np.abs(Y_test - Y_pred))
        Y_pred = Y_pred.reshape(-1,1)
        accuracy = np.mean(Y_pred == Y_test)
        return loss, accuracy

    def fit(self, X_train, Y_train, X_test, Y_test, epochs):
        train_losses, test_losses, train_accuracies, test_accuracies = []
        for epoch in range(self.epochs):
            for x, y in zip(X_train, Y_train):
                # forwarding
                y_pred = x @ self.w + self.b
                y_pred = self.activation(y_pred, "sigmoid")

                # backpropagation
                error = y - y_pred

                # updating
                self.w += self.learning_rate * x * error
                self.b += self.learning_rate * error

            train_losses.append(self.evaluate(X_train, Y_train)[0])
            test_losses.append(self.evaluate(X_test, Y_test)[0])
            train_accuracies.append(self.evaluate(X_train, Y_train)[1])
            test_accuracies.append(self.evaluate(X_test, Y_test)[1])