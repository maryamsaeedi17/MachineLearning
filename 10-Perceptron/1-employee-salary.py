import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


x, y, coef = make_regression(n_samples=100,
                                      n_features=1,
                                      n_informative=1, 
                                      noise=10,
                                      coef=True,
                                      random_state=0)

x = np.interp(x, (x.min(), x.max()), (0, 20))
y = np.interp(y, (y.min(), y.max()), (20000, 150000))

# plt.ion()
plt.plot(x,y,'.',label='training data')
plt.xlabel('Years of experience')
plt.ylabel('Salary $')
plt.title('Experience Vs. Salary')
plt.show()

df = pd.DataFrame(data={'experience':x.flatten(),'salary':y})
print(df.head(10))