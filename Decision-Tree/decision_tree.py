"""
Author: Giang Tran.
"""

import numpy as np

class NodeDT:
    """
    Class Node represents in Decision Tree
    """
    _metrics = {'ce': '_classification_error', 'ig': '_information_gain'}

    def __init__(self, X_feature):
        self.X_feature = X_feature
        self.is_leaf = False

        for i in np.unique(X_feature):
            setattr(self, 'child_' + str(i), None)


class DecisionTree:
    """
    Algorithms:
        + Metrics: either entropy/information gain or classification error.
        - Start from root node. Scan through all features, choose the best feature base on the metric we've just found.
        ....
        - At any level, the number of elements in the set reduces corresponding to that chosen feature.
    """
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _build_dt(self, root):
        # loop through features
        # compute entropy parent node vs entropy child node to get information gain.
        for j in self.X_train.shape[1]:
            pass

    def _train(self):
        root_node = NodeDT
        pass

    def train(self):
        pass

    def _entropy(self):
        """
        E(X) = - sum_v(p(X_v) * log_2(p(X_v))) with X_v is a subset of X = (X_1, X_2, ..., X_n)
        :return: an entropy scalar that measure the uncertainty of a feature in data.
        """
        pass

    def _entropy_node(self):
        pass

    def _information_gain(self):
        pass

    def _classification_error(self):
        pass




