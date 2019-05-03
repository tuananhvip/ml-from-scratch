import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class RidgeRegression:

    def __init__(self, alpha=0.01, epochs=1000, lambda_=0.1, debug=False):
        self.epochs = epochs
        self.lambda_ = lambda_
        self.alpha = alpha
        self.debug = debug
        if debug:
            plt.ion()

    def _hypothesis(self, X, theta):
        return np.dot(X, theta) + self.bias

    def _loss(self, X, y):
        m = y.shape[0]
        return np.linalg.norm(self._hypothesis(X, self.theta) - y, 2) ** 2 / (2*m) + \
               self.lambda_*np.linalg.norm(self.theta, 2) ** 2 / (2*m)

    def _gradient(self, X, y):
        m = X.shape[0]
        return 1/m * np.dot(X.T, self._hypothesis(X, self.theta) - y) + (self.lambda_/m*self.theta)

    def _gradient_bias(self, X, y):
        m = X.shape[0]
        ones = np.ones((m, 1))
        return 1 / m * np.dot(ones.T, self._hypothesis(ones, self.bias) - y)

    def _gradient_descent(self, X, y):
        train_loss = []
        val_loss = []
        for e in range(self.epochs):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
            self.theta = self.theta - self.alpha * self._gradient(X_train, y_train)
            if self.debug:
                print("Train loss at iterations %d: %f " % (e + 1, self._loss(X_train, y_train)))
                print("Val loss at iterations %d: %f " % (e + 1, self._loss(X_val, y_val)))
                train_loss.append(self._loss(X_train, y_train))
                val_loss.append(self._loss(X_val, y_val))
                plt1, = plt.plot(train_loss, color='b', label='Train loss')
                plt2, = plt.plot(val_loss, color='r', label='Validation loss')
                plt.legend(handles=[plt1, plt2])
                plt.show()
                plt.pause(0.5)
            if abs(np.mean(self._gradient(X_train, y_train))) < 1e-3:
                break

    def _train(self, X_train, y_train):
        self._gradient_descent(X_train, y_train)

    def train(self, X_train, y_train):
        self.theta = np.random.normal(size=(X_train.shape[1], 1))
        self.bias = np.mean(y_train)
        self._train(X_train, y_train)

    def predict(self, X_test):
        assert X_test.shape[1] == self.theta.shape[0], "Incorrect shape."
        return self._hypothesis(X_test, self.theta)

    def r2_score(self, pred, y_test):
        total_sum_squares = np.sum((y_test - np.mean(y_test))**2)
        residual_sum_squares = np.sum((y_test - pred)**2)
        return 1 - residual_sum_squares/total_sum_squares

    def ridge_sklearn(self, X_train, y_train):
        from sklearn.linear_model.ridge import Ridge

        ridge = Ridge(self.lambda_, solver='lsqr')
        ridge.fit(X_train, y_train)

        print(ridge.coef_)
        print(ridge.intercept_)


def standardize_regression(X, y):
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    y_mean = np.mean(y)
    y_std = np.std(y)
    return (X - x_mean) / x_std, (y - y_mean) / y_std


def main():
    X = np.loadtxt('prostate.data.txt', skiprows=1)
    columns = ['lcavol', 'lweight',	'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    y = X[:, -1]
    X = X[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, y_train = standardize_regression(X_train, y_train)
    y_train = y_train.reshape((-1, 1))
    experiment = True

    alpha = 0.01
    epochs = 500
    if experiment:
        plt.ion()
        from sklearn.linear_model.ridge import Ridge
        lambda_ = np.arange(1, 10000, 50)
        coefs = []
        for l in lambda_:
            ridge = RidgeRegression(alpha, epochs, l, False)
            ridge.train(X_train, y_train)
            coefs.append(ridge.theta.T)
            # ridge = Ridge(l)
            # ridge.fit(X_train, y_train)
            # coefs.append(ridge.coef_)
        coefs = np.array(coefs).T
        for ind, c in enumerate(coefs):
            c = c.reshape((-1,))
            plt.plot(c, label=columns[ind])
        plt.xlabel('lambda (regularization)')
        plt.ylabel('theta (coefficient)')
        plt.legend()
        plt.show()
        plt.pause(10)
    else:
        lambda_ = 1
        ridge = RidgeRegression(alpha, epochs, lambda_, True)
        ridge.train(X_train, y_train)
        X_test, y_test = standardize_regression(X_test, y_test)
        pred = ridge.predict(X_test)
        y_test = y_test.reshape((-1, 1))
        if ridge.debug:
            print("Test score: %f" % ridge.r2_score(pred, y_test))


if __name__ == '__main__':
    main()

