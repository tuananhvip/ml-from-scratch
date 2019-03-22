from utils import load_mat_file, Graph
from sklearn.svm import SVC

X, y = load_mat_file('ex6data1.mat')
plot = Graph(X, y, 'on')
plot.plot_data()

C = 1.0
linear_svm = SVC(C=C, kernel='linear')

linear_svm.fit(X, y)

W = linear_svm.coef_
b = linear_svm.intercept_
plot.visualize_boundary(W, b)