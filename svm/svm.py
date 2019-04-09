from cvxopt import matrix, solvers
import numpy as np
import os

class SVM:

    def __init__(self, C=1.0, kernel='linear', debug=False, is_saved=False):
        self.C = C
        self.kernel = kernel
        self.debug = debug
        self.is_saved = is_saved

    def _lagrange_duality(self, X_train, y_train, V):
        """
        note: maximize f(x) <==> minimize -f(x)

        ==> Using cvxopt.qp

        minimize: f(x) = (1/2)*x'*P*x + q'*x
        s.t: G*x <= h
             A*x = b

        P = V.T * V
        q = (-1).T

        G = [[-1, 0, 0, ..., 0],
             [0, -1, 0, ..., 0],
             [0, 0, -1, ..., 0],
             ....              ,
             [0, 0, 0, ..., -1],
             --------------
             [1, 0, 0, ..., 0],
             [0, 1, 0, ..., 0],
             [0, 0, 1, ..., 0],
             ....             ,
             [0, 0, 0, ..., 1]]

            => G.shape = (2*N, 2*N)

        h = [[0, 0, 0, ..., 0].T, [C, C, C, ..., C].T]

            => h.shape = (2*N, 1)

        A = y

        b = np.zeros(N)

        """
        N, D = X_train.shape

        P = matrix(V.dot(V.T))  # shape = (N, N)

        q = matrix(-np.ones((N, 1)))

        G = matrix(np.concatenate((-np.eye(N), np.eye(N)), axis=0))  # shape = (2N, N)
        h = matrix(np.array([0] * N + [self.C] * N).reshape(-1, 1))  # shape = (2N, 1)

        A = matrix(y_train.T)
        b = matrix(np.zeros((1, 1)))

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)

        lambda_ = np.array(sol['x'])

        return lambda_

    def _coef(self, X, y, V, lambda_):
        epsilon = 1e-6
        S = np.where(lambda_ > epsilon)[0]
        V_S = V[S, :]
        X_S = X[S, :]
        y_S = y[S, :]
        lambda_S = lambda_[S]
        w = V_S.T.dot(lambda_S)
        b = np.mean(y_S - X_S.dot(w))
        return w, b

    def _train(self, X_train, y_train):
        """
        Solve SVM by using Lagrange duality

        V = [x_1*y_1, x_2*y_2, ..., x_n*y_n]

        g(z) = -1/2*z'*V'*V*z + 1'*z

        z = argmax_z g(z) <==> argmin_z -g(z)

        z = argmin_z (1/2)*z'*V'*V*z - 1'*z
        s.t: -z <= 0
              z <= C
              y'*z = 0

        After get z, find w, b.
        ----------------------------------------
        """
        V = X_train * y_train  # shape = (N, D)
        lambda_ = self._lagrange_duality(X_train, y_train, V)
        self.w, self.b = self._coef(X_train, y_train, V, lambda_)

    def train(self, X_train, y_train):
        assert len(np.unique(y_train)) == 2, "This SVM assumes only work for binary classification."
        assert type(X_train) is np.ndarray and type(y_train) is np.ndarray, "Expect numpy array but got %s" % (type(X_train))
        try:
            self.w, self.b = self._load()
        except FileNotFoundError:
            self._train(X_train, y_train)
        if self.is_saved:
            self._save()
        if self.debug:
            self._check_with_sklearn(X_train, y_train)

    def _save(self):
        if "model" not in os.listdir():
            os.mkdir("model")
        if "weights" not in os.listdir("model"):
            np.save("model/weights", {'weights': self.w, 'bias': self.b})

    def _load(self):
        data = np.load("model/weights.npy").item()
        return data['weights'], data['bias']

    def _check_with_sklearn(self, X, y):
        print("-"*50)
        print("------------ Training phrase --------------")
        print("My SVM weights:", self.w)
        print("My SVM bias:", self.b)

        from sklearn.svm import SVC
        self.sk_svm = SVC(C=self.C, kernel='linear')
        self.sk_svm.fit(X, y)

        W = self.sk_svm.coef_
        b = self.sk_svm.intercept_
        print("Sk-learn SVM weights:", W)
        print("Sk-learn SVM bias:", b)
        print("-"*50)

    def predict(self, X_test):
        assert type(X_test) is np.ndarray, "Expect numpy array but got %s" % (type(X_test))
        pred = X_test.dot(self.w) + self.b
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        return pred