import numpy as np

class KNN:
    def __init__(self, k):
        self.k= k

    # training
    def fit(self, X, Y):
        self.X_train= X
        self.Y_train= Y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

        
    def predict(self, X):
        Y=[]
        for x in X:
            distances=[]
            for x_train in self.X_train:
                d=self.euclidean_distance(x, x_train)
                distances.append(d)

            nearest_neighbors= np.argsort(distances)[0:self.k]
            result= np.bincount(self.Y_train[nearest_neighbors])
            y=np.argmax(result)
            Y.append(y)
        return Y

    def evaluate(self, X, Y):
        Y_pred=self.predict(X)
        accuracy=np.sum(Y_pred == Y) / len(Y)
        return accuracy
