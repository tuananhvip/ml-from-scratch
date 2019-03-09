"""
Author: Giang Tran.
"""

import numpy as np
from math import log2


class NodeDT:
    """
    Class Node represents in Decision Tree
    """

    def __init__(self, X_feature):
        self.X_feature = X_feature
        self.is_leaf = False
        self.used = []

    def entropy(self):
        n = len(self.X_feature)
        sum_ = 0
        for i in np.unique(self.X_feature):
            v = len(self.X_feature[self.X_feature == i])
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

    def __init__(self, X_train, y_train, criterion='ce'):
        self.X_train = X_train
        self.y_train = y_train
        self.criterion = criterion
        self.num_class = np.unique(self.y_train)

    def _build_dt(self, root):
        # loop through features
        # compute entropy parent node vs entropy child node to get information gain.
        """
        Algorithm:
            - Start from the root. Find the best feature that has optimum entropy/information gain or classification error.
            - From that best feature, loop through all categories in to build subtree.
            - If entropy/classification erorr is 0, or reach all features then that node is leaf, stop and move to
                other subtrees
        :param root:
        :return:
        """
        N, D = self.X_train.shape
        best_coef = 0.0
        best_feature = 0
        for d in range(D):
            if d in root.used:
                continue
            feature = self.X_train[:, d]
            num_training = len(feature)
            # if category? then count
            categories = np.unique(feature)
            entropy_feature = 0
            for category in categories:
                # for each category in that feature
                num_category = len(feature[feature == category])
                for c in self.num_class:
                    # count the number of each class
                    num_category_training = len(feature[np.logical_and(feature == category, self.y_train == c)])
                    # compute entropy/information gain or classification error
                    if self.criterion == 'ig':
                        entropy_feature += num_category/num_training * self._entropy(num_category, num_category_training)
                    else:
                        pass
            # after choose the best feature to split.
            # loop through all its categories to build subtree
            if self.criterion == 'ig':
                information_gain = root.entropy() - entropy_feature
                if best_coef < information_gain:
                    best_coef = information_gain
                    best_feature = d
            else:
                pass
        feature = self.X_train[:, best_feature]
        categories = np.unique(feature)
        for category in categories:
            node = NodeDT(feature[feature == category])
            if not self._stop(node):
                self._build_dt(node)
            else:
                node.is_leaf = True


    def _train(self):
        self.tree = NodeDT(self.y_train)
        self._build_dt(self.tree)

    def train(self):
        pass

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




