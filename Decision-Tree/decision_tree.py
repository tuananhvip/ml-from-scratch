"""
Author: Giang Tran.
"""

import numpy as np
from math import log2


class NodeDT:
    """
    Class Node represents in Decision Tree
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.is_leaf = False
        self.used = []

    def entropy(self):
        n = len(self.X)
        sum_ = 0
        for i in np.unique(self.X):
            v = len(self.X[self.X == i])
            sum_ += -((v/n) * log2(v/n))
        return sum_

    def classification_error(self):
        pass


class DecisionTree:
    """
    Algorithms:
        + Metrics: either entropy/information gain or classification error.
        - Start from root node. Scan through all features, choose the best feature base on the metric we've just found.
        ....
        - At any level, the number of elements in the set reduces corresponding to that chosen feature.
    """
    _metrics = {'ce': '_classification_error', 'ig': '_information_gain'}

    def __init__(self, criterion='ce'):
        self.criterion = criterion

    def _build_dt(self, root):
        # loop through features
        # compute entropy parent node vs entropy child node to get information gain.
        """
        Algorithm:
            - Start from the root. Find the best feature that has optimum entropy/information gain or classification error.
            - From that best feature, loop through all categories to build subtree.
            - If entropy/classification erorr is 0, or reach all features then that node is leaf, stop and move to
                other subtrees
        :param root:
        :return:
        """
        N, D = root.X.shape
        best_coef = 0.0
        best_feature = 0
        for d in range(D):
            if d in root.used:
                continue
            feature = root.X[:, d]
            # if category? then count
            categories = np.unique(feature)
            entropy_feature = 0
            for category in categories:
                # for each category in that feature
                num_category = len(feature[feature == category])
                for c in self.num_class:
                    # count the number of each class
                    num_category_class = len(feature[np.logical_and(feature == category, root.y == c)])
                    # compute entropy/information gain or classification error
                    if self.criterion == 'ig':
                        entropy_feature += num_category/N * self._entropy(num_category, num_category_class)
                    else:
                        pass
            if self.criterion == 'ig':
                information_gain = root.entropy() - entropy_feature
                if best_coef < information_gain:
                    best_coef = information_gain
                    best_feature = d
            else:
                pass
        # after choose the best feature to split.
        # loop through all its categories to build subtree
        feature = root.X[:, best_feature]
        categories = np.unique(feature)
        for i, category in enumerate(categories):
            node = NodeDT(root.X[feature == category], root.y[feature == category])
            node.used = root.used + [best_feature]
            setattr(root, 'child_' + str(i), node)
            if not self._stop(node):
                self._build_dt(node)
            else:
                node.is_leaf = True

    def _train(self, X_train, y_train):
        self.tree = NodeDT(X_train, y_train)
        self._build_dt(self.tree)

    def _is_numerical(self, feature):
        return len(np.unique(feature)) >= 100

    def _convert_numerical_to_categorical(self, feature, y_train, num_class):
        """
        :param feature:
        :param num_class:
        :return:
        """
        assert num_class == 2, "This function only assumes work with binary classification."
        feature = sorted(feature)
        best_threshold = 0.0
        for i in range(len(feature)-1):
            threshold = (feature[i] + feature[i+1]) / 2
            feature[feature < threshold]

    def train(self, X_train, y_train):
        self.num_class = np.unique(y_train)
        _, D = X_train.shape
        for d in range(D):
            feature = X_train[:, d]
            if self._is_numerical(feature):
                X_train[:, d] = self._convert_numerical_to_categorical(feature, y_train, self.num_class)
        self._train(X_train, y_train)

    def _entropy(self, n, v):
        """
        E(X) = - sum_v(p(X_v) * log_2(p(X_v))) with X_v is a subset of X = (X_1, X_2, ..., X_n)
        :return: an entropy scalar that measure the uncertainty of a feature in data.
        """
        return - (v/n) * log2(v/n)

    def _information_gain(self):
        pass

    def _classification_error(self):
        pass

    def _stop(self, node):
        """
        Stop condition:
            - Reach max depth or already reach all features.
            - If entropy of that node is 0
        :return:
        """
        pass


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('data/titanic_train.csv')
    X = df.loc[:, :].drop(['Survived', 'PassengerId'], axis=1).values
    y = df.loc[:, 'Survived'].values

    dt = DecisionTree(criterion='ig')
    dt.train(X, y)


