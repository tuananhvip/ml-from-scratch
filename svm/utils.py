from scipy import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_mat_file(mat_file):
    mat = io.loadmat(mat_file)
    return mat['X'], mat['y']


class Graph:

    def __init__(self, X, y, interactive='off'):
        if interactive == 'on':
            plt.ion()
        self.X = X
        self.y = y

    def plot_data(self):
        y = self.y.reshape((-1, ))
        class_1 = self.X[y == 0]
        class_2 = self.X[y == 1]
        plt.scatter(class_1[:, 0], class_1[:, 1], c='r', marker='+')
        plt.scatter(class_2[:, 0], class_2[:, 1], c='b', marker='o')
        plt.show()

    def visualize_boundary_linear(self, W, b):
        W = W.T.reshape((2, ))
        xp = np.linspace(np.min(self.X[:, 0]), np.max(self.X[:, 0]), 100)
        yp = -(W[0] * xp + b)/W[1]
        plt.plot(xp, yp)
        plt.pause(10)

    def visualize_boundary(self, svm_model):
        x1_plot = np.linspace(np.min(self.X[:, 0]), np.max(self.X[:, 0]), 100)
        x2_plot = np.linspace(np.min(self.X[:, 1]), np.max(self.X[:, 1]), 100)
        X1, X2 = np.meshgrid(x1_plot, x2_plot)
        vals = np.zeros(X1.shape)
        for i in range(0, X1.shape[1]):
            this_X = [X1[:, i], X2[:, i]]
            vals[:, i] = svm_model.predict(this_X)
        print(vals)