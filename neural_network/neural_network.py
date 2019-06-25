"""
Author: Giang Tran
Email: giangtran240896@gmail.com
"""

import numpy as np
import sys
sys.path.append("..")

from optimizations_algorithms.optimizers import SGD, SGDMomentum, RMSProp, Adam
from nn_components.layers import FCLayer, ActivationLayer

class NeuralNetwork:

    def __init__(self, epochs, batch_size, optimizer, nn_structure, batch_norm):
        """
        Deep neural network architecture.

        Parameters
        ----------
        epochs: (integer) number of epochs to train.
        batch_size: (integer) number of batch size.
        optimizer: (object) optimizer object uses to optimize the loss.
        nn_structure: A list of 2-element tuple (num_neuron, activation)
                 represents neural network architecture.
        batch_norm: use batch normalization in neural network.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.batch_norm = batch_norm
        self.layers = self._structure(nn_structure)

    def _structure(self, nn_structure):
        """
        Structure function that initializes neural network architecture.
        """
        layers = []
        for struct in nn_structure:
            num_neurons = struct["num_neurons"]
            weight_init = struct["weight_init"]
            fc = FCLayer(num_neurons=num_neurons, optimizer=self.optimizer, weight_init=weight_init)
            layers.append(fc)
            if "activation" in struct:
                activation = struct["activation"]
                act_layer = ActivationLayer(activation=activation)
                layers.append(act_layer)
        return layers

    def _loss(self, Y, Y_hat):
        """
        Compute cross-entropy loss.

        Parameters
        ----------
        Y: one-hot encoding label. shape=(num_dataset, num_classes)
        Y_hat: softmax probability distribution over each data point. 
            shape=(num_dataset, num_classes)

        Returns
        -------
        J: cross-entropy loss.
        """
        assert Y.shape == Y_hat.shape, "Unmatch shape."
        return -np.sum(np.sum(Y*np.log(Y_hat), axis=1), axis=0)/Y.shape[0]

    def _forward(self, train_X):
        """
        NN forward propagation level.

        Parameters
        ----------
        train_X: training dataset X.
                shape = (N, D)

        Returns
        -------
        Probability distribution of softmax at the last layer.
            shape = (N, C)
        """
        inputs = train_X
        for layer in self.layers:
            inputs = layer.forward(inputs)
        output = inputs
        return output

    def _backward_last(self, Y, Y_hat):
        """
        Special formula of backpropagation for the last layer.
        """
        m = Y.shape[0]
        delta = (Y_hat - Y)/m # shape = (N, C)
        dW = self.layers[-3].output.T.dot(delta)
        self.layers[-2].update_params(dW)
        dA_prev = delta.dot(self.layers[-2].W.T)
        return dA_prev

    def _backward(self, Y, Y_hat, X):
        """
        NN backward propagation level. Update weights of the neural network.

        Parameters
        ----------
        Y: one-hot encoding label.
            shape = (N, C).
        Y_hat: output values of forward propagation NN.
            shape = (N, C).
        X: training dataset.
            shape = (N, D).
        """
        dA_prev = self._backward_last(Y, Y_hat)

        for i in range(len(self.layers)-3, 0, -1):
            if isinstance(self.layers[i], ActivationLayer):
                dA_prev = self.layers[i].backward(dA_prev)
                continue
            dA_prev = self.layers[i].backward(dA_prev, self.layers[i-1])
        _ = self.layers[i-1].backward(dA_prev, X)

    def train(self, train_X, train_Y):
        """
        Training function.

        Parameters
        ----------
        train_X: training dataset X.
        train_Y: one-hot encoding label.
        """
        for e in range(self.epochs):
            Y_hat = self._forward(train_X)
            self._backward(train_Y, Y_hat, train_X)
            loss = self._loss(train_Y, Y_hat)
            print("Loss epoch %d: %f" % (e+1, loss))

    def predict(self, test_X):
        """
        Predict function.
        """
        y_hat = self._forward(test_X)
        return np.argmax(y_hat, axis=1)

if __name__ == '__main__':
    from utils import load_dataset_mnist, preprocess_data
    from mnist_lib import MNIST

    load_dataset_mnist()
    mndata = MNIST('data_mnist')
    training_phase = True
    if training_phase:
        images, labels = mndata.load_training()
        images, labels = preprocess_data(images, labels)
        epochs = 20
        batch_size = 64
        learning_rate = 0.1

        sgd = SGD(learning_rate)
        archs = [
            {"num_neurons": 100, "weight_init": "he", "activation": "sigmoid"},
            {"num_neurons": 125, "weight_init": "he", "activation": "sigmoid"},
            {"num_neurons": 50, "weight_init": "he", "activation": "sigmoid"},
            {"num_neurons": labels.shape[1], "weight_init": "he", "activation": "softmax"}]
        nn = NeuralNetwork(epochs, batch_size, sgd, archs, False)
        nn.train(images, labels)
    else:
        images_test, labels_test = mndata.load_testing()
        images_test, labels_test = preprocess_data(images_test, labels_test, test=True)

        pred = nn.predict(images_test)

        print("Accuracy:", len(pred[labels_test == pred]) / len(pred))
        from sklearn.metrics.classification import confusion_matrix

        print("Confusion matrix: ")
        print(confusion_matrix(labels_test, pred))

