import numpy as np


class SoftmaxRegression:

    def __init__(self, batch_size=64, epochs=100, learning_rate=0.01, ):
        self.W = None
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def cross_entropy_loss(self, y, y_hat):
        """
        Compute cross entropy loss.

        Parameters
        ----------
        y: matrix of 1-hot label vectors.
                Shape = (N, num_class).
        y_hat: predicted matrix from softmax function (probability over num_class). Sum(y_hat[i, :]) = 1.
                Shape = (N, num_class)

        Returns
        -------
        Loss = y*log(y_hat). (element-wise).
                Shape = (N, num_class)
        """
        assert y.shape == y_hat.shape, "y and y_hat must have the same shape."
        y_hat[y_hat == 0] = 1e-9
        loss = -np.sum(np.sum(y * np.log(y_hat), axis=1))/y.shape[0]
        return loss

    def softmax_function(self, X):
        """
        Compute softmax function.
        z = X.dot(W) ==> shape = (N, num_class)

        Parameters
        ----------
        X: input variable.
            Shape = (N, D).

        Returns
        -------
        Softmax: e^z / sum(e^z).
            Shape = (N, num_class)
        """
        assert self.W is not None, "Must call train function first."
        z = X.dot(self.W)
        z = z - np.max(z, axis=1).reshape((z.shape[0], 1))
        return np.exp(z)/np.sum(np.exp(z), axis=1).reshape((z.shape[0], 1))

    def gradient(self, X, y, y_hat):
        """
        Compute gradient matrix.

        Parameters
        ----------
        X: training set.
            Shape = (N, D).
        y: training label (1-hot).
            Shape = (N, num_class).
        y_hat: predicted.
            Shape = (N, num_class).

        Returns
        -------
        Gradient of cross entropy loss respect to W: 1/N*(X.T.dot(y_hat - y))
            Shape = (D, num_class).
        """
        assert y.shape == y_hat.shape, "y and y_hat must be same shape."
        return X.T.dot(y_hat - y)/X.shape[0]

    def gradient_descent(self, grad):
        """
        Update weights by gradient descent

        Parameters
        ----------
        grad: gradient matrix of W.

        """
        self.W = self.W - self.learning_rate * grad

    def _train(self, X_train, y_train):
        for e in range(self.epochs):
            batch_loss = 0
            num_batches = 0
            it = 0
            while it < X_train.shape[0]:
                y_hat = self.softmax_function(X_train[it:it+self.batch_size])
                loss = self.cross_entropy_loss(y_train[it:it+self.batch_size], y_hat)
                batch_loss += loss
                grad = self.gradient(X_train[it:it+self.batch_size], y_train[it:it+self.batch_size], y_hat)
                self.gradient_descent(grad)
                it += self.batch_size
                num_batches += 1
            print("Loss at epoch %s %.2f" % (e + 1 , batch_loss / num_batches))

    def train(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], "X and y must have the same data points."
        self.W = np.random.normal(size=(X_train.shape[1], y_train.shape[1]))
        self._train(X_train, y_train)

    def predict(self, X_test):
        return np.argmax(X_test.dot(self.W), axis=1)


if __name__ == '__main__':
    from utils import load_dataset_mnist, preprocess_data
    from mnist import MNIST

    load_dataset_mnist()
    mndata = MNIST('data_mnist')

    images, labels = mndata.load_training()
    images, labels = preprocess_data(images, labels)
    softmax = SoftmaxRegression(epochs=20)
    softmax.train(images, labels)

    images_test, labels_test = mndata.load_testing()
    images_test, labels_test = preprocess_data(images_test, labels_test, test=True)

    pred = softmax.predict(images_test)

    print("Accuracy:", len(pred[labels_test == pred]) / len(pred))
    from sklearn.metrics.classification import confusion_matrix

    print("Confusion matrix: ")
    print(confusion_matrix(labels_test, pred))
