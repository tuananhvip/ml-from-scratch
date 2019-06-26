import numpy as np
from nn_components.initializations import he_initialization, xavier_initialization, standard_normal_initialization
from nn_components.activations import relu, sigmoid, tanh, softmax, relu_grad, sigmoid_grad, tanh_grad

initialization_mapping = {"he": he_initialization, "xavier": xavier_initialization, "std": standard_normal_initialization}

class Layer:
    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError("Child class must implement forward() function")

    def backward(self):
        raise NotImplementedError("Child class must implement backward() function")

class FCLayer(Layer):

    def __init__(self, num_neurons, weight_init="std"):
        """
        The fully connected layer.

        Parameters
        ----------
        num_neurons: (integer) number of neurons in the layer.
        weight_init: (string) either `he` initialization, `xavier` initialization or standard normal distribution.
        """
        assert weight_init in ["std", "he", "xavier"], "Unknown weight intialization algorithm."
        self.num_neurons = num_neurons
        self.weight_init = weight_init
        self.output = None
        self.W = None

    def forward(self, inputs):
        """
        Layer forward level. 

        Parameters
        ----------
        inputs: inputs of the current layer. This is equivalent to the output of the previous layer.

        Returns
        -------
        output: Output value LINEAR of the current layer.
        """
        if self.W is None:
            self.W = initialization_mapping[self.weight_init](weight_shape=(inputs.shape[1], self.num_neurons))
        self.output = inputs.dot(self.W)
        return self.output

    def backward(self, dA_prev, prev_layer, optimizer):
        """
        Layer backward level. Compute gradient respect to W and update it.
        Also compute gradient respect to X for computing gradient of previous
        layers.

        Parameters
        ----------
        dA_prev: gradient of J respect to X at the after layer.
            None if at the first layer as the backward pass.
        prev_layer: previous layer as the backward pass.
        optimizer: (object) optimizer uses to optimize the loss function.
        Returns
        -------
        dA_prev: gradient of J respect to X at the current layer.
        """
        if type(prev_layer) is np.ndarray:
            grad = prev_layer.T.dot(dA_prev)
            self.update_params(grad, optimizer)
            return None
        grad = prev_layer.output.T.dot(dA_prev)
        self.update_params(grad, optimizer)
        dA_prev = dA_prev.dot(self.W.T)
        return dA_prev

    def update_params(self, grad, optimizer):
        updated_grad = optimizer.minimize(grad)
        self.W -= updated_grad


class ConvLayer(Layer):

    def __init__(self, filter_size, filters, padding='SAME', stride=1, weight_init="std"):
        """
        The convolutional layer.

        Parameters
        ----------
        filter_size: a 2-elements tuple (width `fW`, height `fH`) of the filter. 
        filters: an integer specifies number of filter in the layer.
        padding: use padding to keep output dimension = input dimension .
                    whether 'SAME' or 'VALID'.
        stride: stride of the filters.
        weight_init: (string) either `he` initialization, `xavier` initialization or standard normal distribution.
        """
        assert len(filter_size) == 2, "Filter size must be a 2-elements tuple (width, height)."
        self.filter_size = filter_size
        self.filters = filters
        self.padding = padding
        self.stride = stride
        self.W = None

    def _conv_op(self, slice_a, slice_b):
        """
        Convolutional operation between 2 slices.
        """
        return np.sum(slice_a*slice_b)

    def forward(self, X):
        """
        Forward propagation of conv layer. 
        If padding is 'SAME', we must solve this equation to find appropriate number p:
            oW = (iW - fW + 2p)/s + 1
            oH = (iH - fH + 2p)/s + 1

        Parameters
        ----------
        X: the input to this layer. shape = (m, iW, iH, iC)

        Returns
        -------
        Output value of the layer. shape = (m, oW, oH, filters)
        """
        assert len(X.shape) == 4, "The shape of input image must be a 4-elements tuple (batch_size, height, width, channel)."
        if self.W is None:
            self.W = np.random.normal(size=self.filter_size + (X.shape[-1], self.filters))
        m, iW, iH, iC = X.shape
        fW, fH = self.filter_size
        oW = (iW - fW)/self.stride + 1
        oH = (iH - fH)/self.stride + 1
        if self.padding == "SAME":
            oW = iW
            oH = iH
            p = int(((oW - 1)*self.stride + fW - iW)/2)
            X = np.pad(X, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')
            m, iW, iH, iC = X.shape

        self.conv_out = np.zeros(shape=(m, oW, oH, self.filters))
        for f in range(self.filters):
            for w in range(oW):
                for h in range(oH):
                    self.conv_out[:, w, h, f] = self._conv_op(X[:, w:w+fW, h:h+fH, :], self.W[:, :, :, f])
        return self.conv_out

    def backward(self):
        pass

    def update_params(self, grad, optimizer):
        updated_grad = optimizer.minimize(grad)
        self.W -= updated_grad


class PoolingLayer(Layer):

    def __init__(self, filter_size=(2, 2), stride=2, mode="max"):
        """
        The pooling layer.

        Parameters
        ----------
        filter_size: a 2-elements tuple (width `fW`, height `fH`) of the filter. 
        stride: strides of the filter.
        mode: either average pooling or max pooling.
        """
        self.filter_size = filter_size
        self.stride = stride

    def _pool_op(self, slice_a):
        return np.max(slice_a, axis=(1, 2))

    def forward(self, X):
        m, iW, iH, iC = X.shape
        fW, fH = self.filter_size
        oW = int((iW - fW)/self.stride) + 1
        oH = int((iH - fH)/self.stride) + 1
        self.output = np.zeros(shape=(m, oW, oH, iC))
        for w in range(oW):
            for h in range(oH):
                w_step = w*self.stride
                h_step = h*self.stride
                self.output[:, w, h, :] = self._pool_op(X[:, w_step:w_step+fW, h_step:h_step+fH, :])
        return self.output

class FlattenLayer(Layer):

    def __init__(self):
        pass

    def forward(self, X):
        m, iW, iH, iC = X.shape
        self.output = np.reshape(X, (m, iW*iH*iC))
        return self.output


class ActivationLayer(Layer):

    def __init__(self, activation):
        """
        activation: (string) available activation functions. Must be in [sigmoid, tanh,
                                relu, softmax]. Softmax activation must be at the last layer.
        
        """
        assert activation in ["sigmoid", "tanh", "relu", "softmax"], "Unknown activation."
        self.activation = activation

    def forward(self, X):
        self.output = eval(self.activation)(X)
        return self.output

    def backward(self, dA):
        dA = dA * eval(self.activation + "_grad")(self.output)
        return dA


class DropoutLayer(Layer):

    def __init__(self, keep_prob):
        """
        keep_prob: (float) probability to keep neurons in network, use for dropout technique.
        """
        self.keep_prob = keep_prob


class BatchNormLayer(Layer):

    def __init__(self):
        pass

    def forward(self, X):
        """
        Compute batch norm forward.
        LINEAR -> BATCH NORM -> ACTIVATION.

        Returns
        -------
        Output values of batch normalization.
        """
        if not hasattr(self, "gamma") and not hasattr(self, "beta"):
            self.gamma = np.ones((1, self.num_neurons))
            self.beta = np.zeros((1, self.num_neurons))
        self.mu = np.mean(Z, axis=0, keepdims=True)
        self.sigma = np.std(Z, axis=0, keepdims=True)
        self.Znorm = (Z - self.mu)/np.sqrt(self.sigma)
        return self.gamma*self.Znorm + self.beta

    def backward(self, dZ):
        """
        Computex batch norm backward.
        LINEAR <- BATCH NORM <- ACTIVATION.
        https://giangtranml.github.io/ml/machine-learning/batch-normalization

        Returns
        -------
        dZ: Gradient w.r.t LINEAR function Z.
        """
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