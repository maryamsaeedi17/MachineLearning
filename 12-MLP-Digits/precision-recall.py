import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from perceptron import Perceptron


df = pd.read_csv("Input/Surgical-deepnet.csv")
df.isnull().sum()
X = df.drop('complication', axis=1).copy().values
Y = df['complication'].copy().values
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

model = perceptron(learning_rate=0.001, input_length=X_train.shape[1])
model.fit(X_train, y_train, X_test, y_test, 64)
Y_pred = model.predict(X_test)
y_test = y_test.reshape(-1, 1)


def sklearn_scratch(Y_pred, y_test):
    TP = FP = FN = 0
    for (y_pred, y) in zip(Y_pred, y_test):
        if y_pred == 1:
            if y == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y == 1:
                FN += 1
    return TP, FP, FN


TP, FP, FN = sklearn_scratch(Y_pred, y_test)
print("Percision from scratch: ", TP / (TP + FP))
print("Recall from scratch:", TP / (TP + FN))
print("Percision from Scikit-Learn:", precision_score(y_test, Y_pred, average='binary'))
print("Recall from Scikit-Learn:", recall_score(y_test, Y_pred, average='binary'))