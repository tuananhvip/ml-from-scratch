"""
Author: Giang Tran
Email: giangtran204896@gmail.com
Docs: https://giangtranml.github.io/ml/machine-learning/convolutional-neural-network
"""
import numpy as np
from neural_network.neural_network import NeuralNetwork
from nn_components.layers import ConvLayer, ActivationLayer, PoolingLayer, FlattenLayer, FCLayer, BatchNormLayer


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
        super().__init__(epochs, batch_size, optimizer, cnn_structure)

    def _structure(self, cnn_structure):
        """
        Structure function that initializes CNN architecture.

        Parameters
        ----------
        cnn_structure: (list) a list of dictionaries define CNN architecture.
            Each dictionary element is 1 kind of layer (ConvLayer, FCLayer, PoolingLayer, FlattenLayer, BatchNormLayer).

        - Convolutional layer (`type: conv`) dict should have following key-value pair:
            + filter_size: (tuple) define conv filter size (fH, fW)
            + filters: (int) number of conv filters at the layer.
            + stride: (int) stride of conv filter.
            + weight_init: (str) choose which kind to initialize the filter, either `he` `xavier` or `std`.
            + padding: (str) padding type of input corresponding to the output, either `SAME` or `VALID`.
            + activation (optional): (str) apply activation to the output of the layer. LINEAR -> ACTIVATION.
            + batch_norm (optional): (any) apply batch norm to the output of the layer. LINEAR -> BATCH NORM -> ACTIVATION
        
        - Pooling layer (`type: pool`) dict should have following key-value pair:
            + filter_size: (tuple) define pooling filter size (fH, fW).
            + mode: (str) choose the mode of pooling, either `max` or `avg`.
            + stride: (int) stride of pooling filter.

        - Fully-connected layer (`type: fc`) dict should have following key-value pair:
            + num_neurons: (int) define number of neurons in the dense layer.
            + weight_init: (str) choose which kind to initialize the weight, either `he` `xavier` or `std`.
            + activation (optional): (str) apply activation to the output of the layer. LINEAR -> ACTIVATION.
            + batch_norm (optional): (any) apply batch norm to the output of the layer. LINEAR -> BATCH NORM -> ACTIVATION
        
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
                weight_init = struct["weight_init"]
                conv_layer = ConvLayer(filter_size, filters, padding, stride, weight_init)
                layers.append(conv_layer)
                if "batch_norm" in struct:
                    bn_layer = BatchNormLayer()
                    layers.append(bn_layer)
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
                if "batch_norm" in struct:
                    bn_layer = BatchNormLayer()
                    layers.append(bn_layer)    
                if "activation" in struct:
                    activation = struct["activation"]
                    act_layer = ActivationLayer(activation)
                    layers.append(act_layer)
        return layers

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
            if isinstance(self.layers[i], (FCLayer, ConvLayer, BatchNormLayer)):
                dA_prev = self.layers[i].backward(dA_prev, self.layers[i-1], self.optimizer)
                continue
            dA_prev = self.layers[i].backward(dA_prev, self.layers[i-1])
        _ = self.layers[i-1].backward(dA_prev, X, self.optimizer)
