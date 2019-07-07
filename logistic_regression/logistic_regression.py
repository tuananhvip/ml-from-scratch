"""
Author: Giang Tran
Email: giangtran240896@gmail.com
Docs: https://giangtranml.github.io/ml/machine-learning/logistic-regression
"""
import pandas as pd
import re
import json
import numpy as np
from sklearn.model_selection import train_test_split
from math import ceil

class LogisticRegression:

    def __init__(self, epochs, batch_size, lr):
        """
        Constructor for logistic regression.

        Parameter
        ---------
        :epochs: number of epoch to train logistic regression.
        :batch_size: number of batch size each iteration.
        :lr: learning rate hyperparameter.
        """
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.w = None

    def _sigmoid(self, X):
        """
        Sigmoid activation function:
            s(z) = 1/(1+e^-x)

        Parameter
        ---------
        :X: matrix of dataset. shape = (n, d) with n is number of training, d
            is dimension of each vector.

        Return
        ---------
        s(x): value of activation.
        """
        assert X.shape[1] == self.w.shape[0], "Invalid shape."
        z = X.dot(self.w)
        return 1/(1+np.exp(-z))

    def _cross_entropy_loss(self, y_true, y_pred):
        """
        Compute cross entropy loss.
        """
        m = y_true.shape[0]
        y_pred[y_pred == 0] = 1e-9
        y_pred[y_pred == 1] = 1 - 1e-9
        return -np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1 - y_pred))/m

    def _gradient(self, X, y_true, y_pred):
        """
        Compute gradient of J with respect to `w`.
        """
        m = X.shape[0]
        return (X.T.dot(y_pred - y_true))/m

    def _gradient_descent(self, grad):
        """
        Apply gradient to `w` and update it.
        """
        self.w = self.w - self.lr*grad

    def _train(self, train_X, train_y):
        """
        Main training function. 
        """
        n = train_X.shape[0]
        num_batches = ceil(n/self.batch_size)
        for e in range(self.epochs):
            it = 0
            batch_loss = 0
            while it < num_batches:
                logits = self._sigmoid(train_X[it*self.batch_size:(it+1)*self.batch_size])
                grad = self._gradient(train_X[it*self.batch_size:(it+1)*self.batch_size], 
                                      train_y[it*self.batch_size:(it+1)*self.batch_size], logits)
                self._gradient_descent(grad)
                batch_loss += self._cross_entropy_loss(
                    train_y[it*self.batch_size:(it+1)*self.batch_size], logits)
                it += 1
            print("Loss at epoch %s: %.2f" % (e+1, batch_loss/num_batches))

    def train(self, train_X, train_y):
        """
        Wrapper training function, check the prior condition first.
        """
        assert type(train_X) is np.ndarray, "Expected train X is numpy array but got %s" % type(train_X)
        assert type(train_y) is np.ndarray, "Expected train y is numpy array but got %s" % type(train_y)
        train_y = train_y.reshape((-1, 1))
        self.w = np.random.normal(size=(train_X.shape[1], 1))
        self._train(train_X, train_y)

    def predict(self, test_X):
        """
        Output sigmoid value of trained parameter w, b.
        Choose threshold 0.5
        """
        pred = self._sigmoid(test_X)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

def clean_sentences(string):
    label_chars = re.compile("[^A-Za-z \n]+")
    string = string.lower()
    return re.sub(label_chars, "", string)


def main():
    df = pd.read_csv("data/amazon_baby_subset.csv")
    reviews = df.loc[:, 'review'].values
    for ind, review in enumerate(reviews):
        if type(review) is float:
            reviews[ind] = ""

    reviews = clean_sentences("\n".join(reviews))
    with open("data/important_words.json") as f:
        important_words = json.load(f)
    reviews = reviews.split("\n")
    n = len(reviews)
    d = len(important_words)
    X = np.zeros((n, d))
    y = df.loc[:, 'sentiment'].values
    y[y == -1] = 0

    for ind, review in enumerate(reviews):
        for ind_w, word in enumerate(important_words):
            X[ind, ind_w] = review.count(word)
    ones = np.ones((n, 1))
    X = np.concatenate((X, ones), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    epochs = 50
    learning_rate = 0.1
    batch_size = 64
    logistic = LogisticRegression(epochs, batch_size, learning_rate)
    logistic.train(X_train, y_train)
    pred = logistic.predict(X_test)
    y_test = y_test.reshape((-1, 1))
    print("Accuracy: " + str(len(pred[pred == y_test])/len(pred)))

if __name__ == '__main__':
    main()
