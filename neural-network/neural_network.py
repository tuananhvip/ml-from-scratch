"""
Author: Giang Tran
Email: giangtran240896@gmail.com
"""

import numpy as np


class HiddenLayer:

    def __init__(self, num_neurons, activation, alpha):
        """
        Abstract class represents for all hidden layers in class neural network.

        Parameters
        ----------
        num_neurons: hyperparameter for number of neurons in the layer.
        activation: available activation functions. Must be in [sigmoid, tanh,
            relu, softmax]. Softmax activation must be at the last layer.
        alpha: hyperparameter learning rate for gradient descent.
        """
        assert activation in ["sigmoid", "tanh", "relu", "softmax"], "Unknown activation."
        self.num_neurons = num_neurons
        self.activation = activation
        self.alpha = alpha
        self.output = None
        self.W = None
 
    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def _tanh(self, z):
        return np.tanh(z)

    def _relu(self, z):
        return z*(z > 0)

    def _softmax(self, z):
        z_prime = z - np.max(z, axis=1, keepdims=True)
        return np.exp(z_prime) / np.sum(np.exp(z_prime), axis=1, keepdims=True)

    def _sigmoid_grad(self, z):
        return z*(1-z)

    def _tanh_grad(self, z):
        return 1 - z**2

    def _relu_grad(self, z):
        return 1*(z > 0)

    def forward(self, inputs):
        """
        Layer forward level. LINEAR -> ACTIVATION. 
        Initialize weights according `He initialization` and `Xavier
        initialization`

        Parameters
        ----------
        inputs: inputs of the current layer. This is equivalent to the output of the
        previous layer.

        Returns
        -------
        output: Output value LINEAR -> ACTIVATION of the current layer.
        """
        if self.W is None:
            scale = (np.sqrt(2/inputs.shape[1]) if self.activation == "relu" else
                     np.sqrt(1/inputs.shape[1]))
            self.W = np.random.normal(0, scale,
                                     (inputs.shape[1], self.num_neurons))
        Z = inputs.dot(self.W)
        self.A = getattr(self, "_" + self.activation)(Z)
        return self.A

    def backward(self, dZ, dA_prev, X, input_layer=False):
        """
        Layer backward level. Compute gradient respect to W and update it.
        Also compute gradient respect to X for computing gradient of previous
        layers.

        Parameters
        ----------
        dZ: gradient of J respect to Z at the current layer.
        dA_prev: gradient of J respect to X at the after layer.
            None if at the first layer.
        X: inputs.
        input_layer: whether at input layer or not.

        Returns
        -------
        dA_prev: gradient of J respect to X at the current layer.
        """
        grad = X.T.dot(dZ)
        if input_layer:
            dA_prev = None
        else:
            dA_prev = dZ.dot(self.W.T)
        self.W = self.W - self.alpha*grad
        return dA_prev

class NeuralNetwork:

    def __init__(self, epochs, batch_size, learning_rate, nn_structure):
        """
        Deep neural network architecture.

        Parameters
        ----------
        epochs: (integer) number of epochs to train.
        batch_size: (integer) number of batch size.
        nn_structure: A list of 2-element tuple (num_neuron, activation)
                 represents neural network architecture.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.nn_structure = nn_structure
        self.learning_rate = learning_rate
        self.layers = self._structure()

    def _structure(self):
        """
        Structure function that initializes neural network architecture.
        """
        layers = []
        for struct in self.nn_structure:
            num_neurons = struct[0]
            activation = struct[1]
            layer = HiddenLayer(num_neurons, activation, learning_rate)
            layers.append(layer)
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
        m = Y.shape[0]
        dZ = (Y_hat - Y)/m # shape = (N, C)
        dA_prev = None
        for i in range(len(self.layers)-1, 0, -1):
            dA_prev = self.layers[i].backward(dZ, dA_prev, self.layers[i-1].A)
            dZ = dA_prev * getattr(self.layers[i-1], "_" + self.layers[i-1].activation +
                                  "_grad")(self.layers[i-1].A)
        _ = self.layers[i-1].backward(dZ, dA_prev, X, True)

    def train(self, train_X, train_Y):
        for e in range(self.epochs):
            Y_hat = self._forward(train_X)
            self._backward(train_Y, Y_hat, train_X)
            loss = self._loss(train_Y, Y_hat)
            print("Loss epoch %d: %.2f" % (e+1, loss))

    def predict(self, test_X):
        y_hat = self._forward(test_X)
        return np.argmax(y_hat, axis=1)

if __name__ == '__main__':
    from utils import load_dataset_mnist, preprocess_data
    from mnist import MNIST

    # load_dataset_mnist()
    mndata = MNIST('../data_mnist')
    training_phase = True
    if training_phase:
        images, labels = mndata.load_training()
        images, labels = preprocess_data(images, labels)
        epochs = 20
        batch_size = 64
        learning_rate = 0.1
        archs = [(100, "relu"), (125, "relu"), (50, "relu"), (labels.shape[1],
                                                             "softmax")]
        nn = NeuralNetwork(epochs, batch_size, learning_rate, archs)
        nn.train(images, labels)
    else:
        images_test, labels_test = mndata.load_testing()
        images_test, labels_test = preprocess_data(images_test, labels_test, test=True)

        pred = nn.predict(images_test)

        print("Accuracy:", len(pred[labels_test == pred]) / len(pred))
        from sklearn.metrics.classification import confusion_matrix

        print("Confusion matrix: ")
        print(confusion_matrix(labels_test, pred))

