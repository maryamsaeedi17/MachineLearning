import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

#baloon
xs = np.random.normal(30, 5, 200)
ys = np.random.normal(45, 5, 200)
zs = np.random.normal(0.005, 0.01, 200)
ax.scatter(xs, ys, zs, marker='D', color="r")

#melon
xs = np.random.normal(20, 2, 200)
ys = np.random.normal(20, 2, 200)
zs = np.random.normal(2, 0.2, 200)
ax.scatter(xs, ys, zs, marker='o', color="y")

ax.set_xlabel("Length(cm)")
ax.set_ylabel("Width(cm)")
ax.set_zlabel("Weight(kg)")

plt.title("Balloon-Melon War")
plt.show()
