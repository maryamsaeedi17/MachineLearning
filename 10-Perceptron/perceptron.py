import numpy as np

class Perceptron:
    def __init__(self, input_length, learning_rate, epochs):
        # self.w=np.zeros(input_lenght)
        # self.b=0
        self.w=np.random.rand(input_length)
        self.b=np.random.rand(1)
        self.learning_rate=learning_rate
        self.epochs=epochs

    def fit(self, X_train, Y_train):
        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                x=X_train[j]
                y=Y_train[j]
                y_pred= self.w * x + self.b
                error=y - y_pred
                self.w += self.learning_rate * error * x
                #self.b += self.learning_rate * error
                self.b += 0.1 * error


    def predict(self, X_test):
        Y_pred=[]
        for x_test in X_test:
            y_pred=x_test * self.w + self.b
            Y_pred.append(y_pred)
        return np.array(Y_pred)

    def evaluate(self, X_test, Y_test):
        Y_pred=self.predict(X_test)

        loss= np.mean(np.square(Y_test, Y_pred))
        accuracy=np.sum(Y_pred == Y_test) / len(Y_test)

        return loss, accuracy