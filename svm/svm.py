from utils import load_mat_file, plot_data
from sklearn.svm import SVC

X, y = load_mat_file('ex6data1.mat')
plot_data(X, y)