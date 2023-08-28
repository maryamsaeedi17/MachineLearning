import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from perceptron import Perceptron


x, y, coef = make_regression(n_samples=100,
                                      n_features=1,
                                      n_informative=1, 
                                      noise=10,
                                      coef=True,
                                      random_state=0)

x = np.interp(x, (x.min(), x.max()), (0, 20))
y = np.interp(y, (y.min(), y.max()), (20000, 150000))

# plt.ion()
# plt.plot(x,y,'.',label='training data')
# plt.xlabel('Years of experience')
# plt.ylabel('Salary $')
# plt.title('Experience Vs. Salary')
# plt.show()

df = pd.DataFrame(data={'experience':x.flatten(),'salary':y})
print(df.head(10))

X=df["experience"].values
Y=df["salary"].values

#X = np.c_[X, np.ones(X.shape[0])]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

X_train=X_train.reshape(-1, 1)
X_test=X_test.reshape(-1, 1)
Y_train=Y_train.reshape(-1, 1)
Y_test=X_test.reshape(-1, 1)

print(X_train.shape)
# print(X_test.shape)

# print(Y_train.shape)
# print(Y_test.shape)

model=Perceptron(input_length=X_train.shape[1], learning_rate= 0.001, epochs=128)
model.fit(X_train, Y_train)

loss, accuracy=model.evaluate(X_test, Y_test)

print("loss: ", loss)
print("accuracy: ", accuracy)