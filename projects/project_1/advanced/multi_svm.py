from svm import SVM
import numpy as np


class OneVsRestSVM:

    def __init__(self, C=1.0, kernel='linear'):
        self.C = C
        self.kernel = kernel
        self.classifiers = dict()

    def _train(self, X_train, y_train):
        for klass in self.klasses:
            y_clone = y_train[:]
            y_clone = y_clone.astype(np.short)
            y_clone[y_clone == klass] = 1
            y_clone[y_clone != 1] = -1
            svm = self.classifiers[klass]
            svm.train(X_train, y_clone)

    def train(self, X_train, y_train):
        if type(X_train) is not np.ndarray:
            X_train = np.array(X_train)
        if type(y_train) is not np.ndarray:
            y_train = np.array(y_train)
        self.klasses = np.unique(y_train)
        for klass in self.klasses:
            svm_classifier = SVM(self.C, self.kernel)
            self.classifiers[klass] = svm_classifier
        self._train(X_train, y_train)
        print(self.classifiers)

    def predict(self, X_test):
        pass