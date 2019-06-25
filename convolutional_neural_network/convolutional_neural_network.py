"""
Author: Giang Tran
Email: giangtran204896@gmail.com
"""
import numpy as np
import sys
sys.path.append("..")
from neural_network.neural_network import NeuralNetwork
from nn_components.layers import ConvLayer, ActivationLayer, PoolingLayer, FlattenLayer, FCLayer
from optimizations_algorithms.optimizers import SGD, SGDMomentum, RMSProp, Adam

class CNN(NeuralNetwork):
    def __init__(self, epochs, batch_size, optimizer, cnn_structure):
        """
        A Convolutional Neural Network.

        Parameters
        ----------
        epochs: (integer) number of epochs to train model.
        batch_size: (integer) number of batch size to train at each iterations.
        optimizer: (object) optimizer class to use (gsd, gsd_momentum, rms_prop, adam)
        cnn_structure: (list) a list of dictionary of cnn architecture.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.layers = self._structure(cnn_structure) 

    def _structure(self, cnn_structure):
        """
        Structure function that initializes cnn architecture.
        """
        layers = []
        for struct in cnn_structure:
            filter_size = struct["filter_size"]
            filters = struct["filters"]
            padding = struct["padding"]
            stride = struct["stride"]
            conv_layer = ConvLayer(filter_size, filters, padding, stride)
            layers.append(conv_layer)
            if "activation" in struct:
                activation = struct["activation"]
                act_layer = ActivationLayer(activation=activation)
                layers.append(act_layer)
        return layers

    def _forward(self, train_X):
        pass

    def _backward(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass


def main():
    lenet_arch = [{"filter_size": (5,5), "filters": 6, "padding": "SAME", "stride": 1, "activation": "sigmoid"},
                  {"filter_size": (5,5), "filters": 6, "padding": "SAME", "stride": 1, "activation": "sigmoid"}]

if __name__ == "__main__":
    main()