"""
Author: Giang Tran
Email: giangtran204896@gmail.com
"""
import numpy as np
import sys
sys.path.append("..")

debug = True
debug_path = "D:\ml_from_scratch"
if debug:
    sys.path.append(debug_path)

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
            if type(struct) is str and struct == "flatten":
                flatten_layer = FlattenLayer()
                layers.append(flatten_layer)
                continue
            if struct["type"] == "conv":
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
            elif struct["type"] == "pool":
                filter_size = struct["filter_size"]
                stride = struct["stride"]
                mode = struct["mode"]
                pool_layer = PoolingLayer(filter_size=filter_size, stride=stride, mode=mode)
                layers.append(pool_layer)
            else:
                num_neurons = struct["num_neurons"]
                weight_init = struct["weight_init"]
                fc_layer = FCLayer(num_neurons=num_neurons, weight_init=weight_init)
                layers.append(fc_layer)
                if "activation" in struct:
                    activation = struct["activation"]
                    act_layer = ActivationLayer(activation)
                    layers.append(act_layer)
        return layers

    def _forward(self, train_X):
        """
        Forward propagation all layers in convolutional neural network.

        Parameters
        ----------
        train_X: input training image X. shape = (m, iW, iH, iC)

        Returns
        -------
        Output value of the last layer. 
        """
        input_X = train_X
        for layer in self.layers:
            input_X = layer.forward(input_X)
        output = input_X
        return output

    def _backward(self, Y, Y_hat, X):
        """
        CNN backward propagation.

        Parameters
        ----------
        Y: one-hot encoding label.
            shape = (m, C).
        Y_hat: output values of forward propagation NN.
            shape = (m, C).
        X: training dataset.
            shape = (m, iW, iH, iC).
        """
        dA_prev = self._backward_last(Y, Y_hat)
        for i in range(len(self.layers)-3, 0, -1):
            print(i)
            if isinstance(self.layers[i], FCLayer) or isinstance(self.layers[i], ConvLayer):
                dA_prev = self.layers[i].backward(dA_prev, self.layers[i-1], self.optimizer)
                continue
            dA_prev = self.layers[i].backward(dA_prev, self.layers[i-1])
        _ = self.layers[i-1].backward(dA_prev, X, self.optimizer)

    def predict(self):
        pass


def main():
    from libs.utils import load_dataset_mnist, preprocess_data
    from libs.mnist_lib import MNIST
    if debug:
        load_dataset_mnist(debug_path + "/libs")
        mndata = MNIST(debug_path + "/libs/data_mnist")
    else:
        load_dataset_mnist("../libs")
        mndata = MNIST('../libs/data_mnist')
    lenet_arch = [{"type": "conv", "filter_size": (5, 5), "filters": 6, "padding": "SAME", "stride": 1, "activation": "relu"},
                {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
                {"type": "conv", "filter_size": (5, 5), "filters": 16, "padding": "SAME", "stride": 1, "activation": "relu"},
                {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
                "flatten",
                {"type": "fc", "num_neurons": 120, "weight_init": "std", "activation": "tanh"},
                {"type": "fc", "num_neurons": 84, "weight_init": "std", "activation": "tanh"},
                {"type": "fc", "num_neurons": 10, "weight_init": "std", "activation": "softmax"}
                ]
    epochs = 20
    batch_size = 5
    learning_rate = 0.1
    sgd = SGD(learning_rate)
    cnn = CNN(epochs=epochs, batch_size=batch_size, optimizer=sgd, cnn_structure=lenet_arch)
    training_phase = True
    if training_phase:
        images, labels = mndata.load_training()
        images, labels = preprocess_data(images, labels, nn=True)
        cnn.train(images[:batch_size, ], labels[:batch_size, ])

if __name__ == "__main__":
    main()