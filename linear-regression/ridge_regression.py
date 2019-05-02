import numpy as np
from sklearn.model_selection import train_test_split


class RidgeRegression:

    def __init__(self, alpha=0.01, epochs=1000, lambda_=0.1, debug=False):
        self.epochs = epochs
        self.lambda_ = lambda_
        self.alpha = alpha
        self.debug = debug

    def _hypothesis(self, X, theta):
        assert X.shape[1] == theta.shape[0], "Incorrect shape."
        return np.dot(X, theta)

    def _loss(self, X, y):
        m = y.shape[0]
        return np.linalg.norm(self._hypothesis(X, self.theta) - y, 2) ** 2 / (2*m) + \
               self.lambda_*np.linalg.norm(self.theta, 2) ** 2 / 2

    def _gradient(self, X, y):
        m = X.shape[0]
        return 1 / m * np.dot(X.T, self._hypothesis(X, self.theta) - y) + (self.lambda_*self.theta)

    def _gradient_bias(self, X, y):
        m = X.shape[0]
        ones = np.ones((m, 1)).T
        return 1 / m * np.dot(ones, self._hypothesis(X, self.theta) - y)

    def _gradient_descent(self, X, y):
        for e in range(self.epochs):
            self.theta = self.theta - self.alpha * self._gradient(X, y)
            self.bias = self.bias - self.alpha * self._gradient_bias(X, y)
            if self.debug:
                print("Loss at iterations %d: %f " % (e + 1, self._loss(X, y)))
            if abs(np.mean(self._gradient(X, y))) < 1e-3:
                break

    def _train(self, X_train, y_train):
        self._gradient_descent(X_train, y_train)

    def train(self, X_train, y_train):
        self.theta = np.random.normal(size=(X_train.shape[1], 1))
        self.bias = np.random.normal()
        self._train(X_train, y_train)

    def predict(self, X_test):
        assert X_test.shape[1] == self.theta.shape[0], "Incorrect shape."
        return self._hypothesis(X_test, self.theta)

    def score(self, pred, y_test):
        assert pred.shape == y_test.shape, "Prediction and Label must be the same shape."
        return abs(np.mean(pred - y_test))

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
    y = X[:, -1]
    X = X[:, :-1]
    X, y = standardize_regression(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_train = y_train.reshape((-1, 1))

    alpha = 0.01
    epochs = 500
    lammbda_ = 0
    ridge = RidgeRegression(alpha, epochs, lammbda_, False)
    ridge.train(X_train, y_train)
    print(ridge.theta.T)
    print(ridge.bias)
    pred = ridge.predict(X_test)
    y_test = y_test.reshape((-1, 1))
    print("Test score: %f" % ridge.score(pred, y_test))

    ridge.ridge_sklearn(X_train, y_train)


if __name__ == '__main__':
    main()

