"""
Author: Giang Tran
Email: giangtran240896@gmail.com
"""

import numpy as np


class HiddenLayer:

    def __init__(self, num_neurons, activation, alpha, keep_prob=1,
                 use_batch_norm=False):
        """
        Abstract class represents for all hidden layers in class neural network.

        Parameters
        ----------
        num_neurons: hyperparameter for number of neurons in the layer.
        activation: available activation functions. Must be in [sigmoid, tanh,
            relu, softmax]. Softmax activation must be at the last layer.
        alpha: hyperparameter learning rate for gradient descent.
        keep_prob: probability to keep neurons in network, use for dropout technique.
        use_batch_norm: whether use batch normalization. If true, the NN
            provides 2 new parameters gamma and beta.
        """
        assert activation in ["sigmoid", "tanh", "relu", "softmax"], "Unknown activation."
        self.num_neurons = num_neurons
        self.activation = activation
        self.alpha = alpha
        self.keep_prob = keep_prob
        self.use_batch_norm = use_batch_norm
        self.output = None
        self.W = None
 
    def _sigmoid(self, z):
        """
        Sigmoid activation function.
            g(z) = 1 / (1 + e^-z)
        """
        return 1/(1+np.exp(-z))

    def _tanh(self, z):
        """
        Tanh activation function.
            g(z) = tanh(z)
        """
        return np.tanh(z)

    def _relu(self, z):
        """
        Relu activation function.
            g(z) = max(0, z)
        """
        return z*(z > 0)

    def _softmax(self, z):
        """
        Softmax activation function. Use at the output layer.
            g(z) = e^z / sum(e^z)
        """
        z_prime = z - np.max(z, axis=1, keepdims=True)
        return np.exp(z_prime) / np.sum(np.exp(z_prime), axis=1, keepdims=True)

    def _sigmoid_grad(self, z):
        """
        Sigmoid derivative.
            g'(z) = g(z)(1-g(z))
        """
        return z*(1-z)

    def _tanh_grad(self, z):
        """
        Tanh derivative.
            g'(z) = 1 - g^2(z).
        """
        return 1 - z**2

    def _relu_grad(self, z):
        """
        Relu derivative.
            g'(z) = 0 if g(z) <= 0
            g'(z) = 1 if g(z) > 0
        """
        return 1*(z > 0)

    def _dropout(self, A):
        np.random.uniform(0, self.keep_prob, A.shape)
        pass

    def _batch_norm_forward(self, Z):
        if not self.use_batch_norm:
            return Z
        if not hasattr(self, "gamma") and not hasattr(self, "beta"):
            self.gamma = np.ones((1, self.num_neurons))
            self.beta = np.zeros((1, self.num_neurons))
        self.mu = np.mean(Z, axis=0, keepdims=True)
        self.sigma = np.std(Z, axis=0, keepdims=True)
        self.Znorm = (Z - self.mu)/np.sqrt(self.sigma)
        return self.gamma*self.Znorm + self.beta

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
        self.Z = inputs.dot(self.W)
        Ztrans = self._batch_norm_forward(self.Z)
        self.A = getattr(self, "_" + self.activation)(Ztrans)
        return self.A

    def _batch_norm_backward(self, dZ):
        if not self.use_batch_norm:
            return dZ
        m = self.Z.shape[0]
        dZnorm = dZ * self.gamma
        self.gamma = np.sum(dZ * self.Znorm, axis=0, keepdims=True)
        self.beta = np.sum(dZ, axis=0, keepdims=True)
        dSigma = np.sum(dZnorm * (-((self.Z - self.mu)*self.sigma**(-3/2))/2),
                       axis=0, keepdims=True)
        dMu = np.sum(dZnorm*(-1/np.sqrt(self.sigma)), axis=0, keepdims=True) +\
                dSigma*((-2/m)*np.sum(self.Z - self.mu, axis=0, keepdims=True))
        dZ = dZnorm*(1/np.sqrt(self.sigma)) + dMu/m +\
                dSigma*((2/m)*np.sum(self.Z - self.mu, axis=0, keepdims=True))
        return dZ

    def backward(self, dZ, dA_prev, prev_layer):
        """
        Layer backward level. Compute gradient respect to W and update it.
        Also compute gradient respect to X for computing gradient of previous
        layers.

        Parameters
        ----------
        dZ: gradient of J respect to Z at the current layer.
        dA_prev: gradient of J respect to X at the after layer.
            None if at the first layer as the backward pass.
        prev_layer: previous layer as the backward pass.

        Returns
        -------
        dA_prev: gradient of J respect to X at the current layer.
        """
        if type(prev_layer) is np.ndarray:
            grad = prev_layer.T.dot(dZ)
            self.W = self.W - self.alpha*grad
            return None
        dZ = self._batch_norm_backward(dZ)
        grad = prev_layer.A.T.dot(dZ)
        dA_prev = dZ.dot(self.W.T)
        self.W = self._gradient_descent(self.W, grad)
        dZ = dA_prev * getattr(prev_layer, "_" + prev_layer.activation +
                                "_grad")(prev_layer.A)
        return dZ, dA_prev

    def _gradient_descent(self, param, grad):
        return param - self.alpha * grad

class NeuralNetwork:

    def __init__(self, epochs, batch_size, learning_rate, nn_structure,
                 batch_norm):
        """
        Deep neural network architecture.

        Parameters
        ----------
        epochs: (integer) number of epochs to train.
        batch_size: (integer) number of batch size.
        nn_structure: A list of 2-element tuple (num_neuron, activation)
                 represents neural network architecture.
        batch_norm: use batch normalization in neural network.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.nn_structure = nn_structure
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.layers = self._structure()

    def _structure(self):
        """
        Structure function that initializes neural network architecture.
        """
        layers = []
        for struct in self.nn_structure:
            num_neurons = struct[0]
            activation = struct[1]
            layer = HiddenLayer(num_neurons, activation, learning_rate,
                                1.0, self.batch_norm)
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
            dZ, dA_prev = self.layers[i].backward(dZ, dA_prev, self.layers[i-1])
        _ = self.layers[i-1].backward(dZ, dA_prev, X)

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
        archs = [(100, "sigmoid"), (125, "sigmoid"), (50, "sigmoid"), (labels.shape[1],
                                                             "softmax")]
        nn = NeuralNetwork(epochs, batch_size, learning_rate, archs, True)
        nn.train(images, labels)
    else:
        images_test, labels_test = mndata.load_testing()
        images_test, labels_test = preprocess_data(images_test, labels_test, test=True)

        pred = nn.predict(images_test)

        print("Accuracy:", len(pred[labels_test == pred]) / len(pred))
        from sklearn.metrics.classification import confusion_matrix

        print("Confusion matrix: ")
        print(confusion_matrix(labels_test, pred))

