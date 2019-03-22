from scipy import io
import matplotlib.pyplot as plt


def load_mat_file(mat_file):
    mat = io.loadmat(mat_file)
    return mat['X'], mat['y']


def plot_data(X, y):
    y = y.reshape((-1, ))
    class_1 = X[y == 0]
    class_2 = X[y == 1]
    plt.scatter(class_1[:, 0], class_1[:, 1], c='r', marker='+')
    plt.scatter(class_2[:, 0], class_2[:, 1], c='b', marker='o')
    plt.show()