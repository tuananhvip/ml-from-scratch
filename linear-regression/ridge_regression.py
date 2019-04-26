import numpy as np
from sklearn.model_selection import train_test_split

X = np.loadtxt('prostate.data.txt', skiprows=1)

y = X[:, -1]
X = X[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

