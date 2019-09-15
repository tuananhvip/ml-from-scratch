import numpy as np
np.seterr(all="raise")
from nn_components.initializations import he_initialization, xavier_initialization, standard_normal_initialization
from nn_components.activations import relu, sigmoid, tanh, softmax, relu_grad, sigmoid_grad, tanh_grad

initialization_mapping = {"he": he_initialization, "xavier": xavier_initialization, "std": standard_normal_initialization}

class Layer:
    def __init__(self):
        pass

    def forward(self, X):
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
        assert weight_init in ["std", "he", "xavier"], "Weight initialization must be in either `he` or `xavier` or `std`."
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

    def backward(self, d_prev, prev_layer, optimizer):
        """
        Layer backward level. Compute gradient respect to W and update it.
        Also compute gradient respect to X for computing gradient of previous
        layers as the forward direction [l-1].

        Parameters
        ----------
        d_prev: gradient of J respect to A[l+1] of the previous layer according the backward direction.
        prev_layer: previous layer according the forward direction.
        optimizer: (object) optimizer uses to optimize the loss function.
        
        Returns
        -------
        d_prev: gradient of J respect to A[l] at the current layer.
        """
        if type(prev_layer) is np.ndarray:
            grad = prev_layer.T.dot(d_prev)
            self.update_params(grad, optimizer)
            return None
        grad = prev_layer.output.T.dot(d_prev)
        self.update_params(grad, optimizer)
        d_prev = d_prev.dot(self.W.T)
        return d_prev

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
        weight_init: (string) either `he` initialization, `xavier` initialization or `std` standard normal distribution.
        """
        assert len(filter_size) == 2, "Filter size must be a 2-elements tuple (width, height)."
        assert weight_init in ["he", "xavier", "std"], "Weight initialization must be in either `he` or `xavier` or `std`."
        self.filter_size = filter_size
        self.filters = filters
        self.padding = padding
        self.stride = stride
        self.weight_init = weight_init
        self.W = None

    def _conv_op(self, input_slice, kernel):
        """
        Convolutional operation of 2 slices.

        Parameters
        ----------
        input_slice: Input slice, shape = (m, fW, fH, in_filters)
        kernel: Kernel shape = (fW, fH, in_filters, out_filters)

        Returns
        -------
        output_slice shape = (m, out_filters)
        """
        return np.einsum("mijk,ijkf->mf", input_slice, kernel)

    def _conv_backward_op(self, input_slice, d_prev_slice, update_params=True):
        """
        Convolutional backward operation of 2 slices.

        Parameters
        ----------
        if update_params is true:
            input_slice: Input slice, shape = (m, fW, fH, in_filters)
        else:
            input_slice: Kernel, shape = (fW, fH, in_filters, out_filters)
        d_prev_slice: Derivative slice of previous layer. shape = (m, out_filters)

        Returns
        -------
        if update_params is true:
            Derivative with respect to W shape = (fW, fH, in_filters, out_filters)
        else:
            Derivative with respect to X shape = (m, fW, fH, in_filters)
        """
        if update_params:
            return np.einsum("mijk,md->ijkd", input_slice, d_prev_slice)
        else:
            return np.einsum("ijkl,ml->mijk", input_slice, d_prev_slice)

    def _pad_input(self, inp):
        """
        Pad the input when using padding mode 'SAME'.
        """
        m, iW, iH, iC = inp.shape
        fW, fH = self.filter_size
        oW, oH = iW, iH
        pW = int(((oW - 1)*self.stride + fW - iW)/2)
        pH = int(((oH - 1)*self.stride + fH - iH)/2)
        X = np.pad(inp, ((0, 0), (pW, pH), (pW, pH), (0, 0)), 'constant')
        return X

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
            self.W = initialization_mapping[self.weight_init](weight_shape=self.filter_size + (X.shape[-1], self.filters))
        m, iW, iH, iC = X.shape
        fW, fH = self.filter_size
        oW = int((iW - fW)/self.stride + 1)
        oH = int((iH - fH)/self.stride + 1)
        if self.padding == "SAME":
            X = self._pad_input(X)
            m, iW, iH, iC = X.shape
        self.output = np.zeros(shape=(m, oW, oH, self.filters))
        for w in range(oW):
            for h in range(oH):
                w_step = w*self.stride
                h_step = h*self.stride
                self.output[:, w, h, :] = self._conv_op(X[:, w_step:w_step+fW, h_step:h_step+fH, :], self.W)
        return self.output

    def backward(self, d_prev, prev_layer, optimizer):
        """
        Backward propagation of the conv layer.
        
        Parameters
        ----------
        d_prev: gradient of J respect to A[l+1] of the previous layer according the backward direction.
        prev_layer: previous layer according the forward direction.
        
        """
        m, oW, oH, num_filts = self.output.shape
        fW, fH, fC, _ = self.W.shape
        dW = np.zeros(shape=(fW, fH, fC, num_filts))
        dA_temp = None
        if type(prev_layer) is not np.ndarray:
            dA_temp = np.zeros(shape=prev_layer.output.shape)
            X = prev_layer.output.copy()
        else:
            X = prev_layer.copy()
        if self.padding == "SAME":
            X = self._pad_input(X)
        for w in range(oW):
            for h in range(oH):
                w_step = w*self.stride
                h_step = h*self.stride
                dW = np.add(dW, self._conv_backward_op(X[:, w_step:w_step+fW, h_step:h_step+fH, :], d_prev[:, w, h, :])) 
                if dA_temp is None:
                    continue
                dA_temp[:, w_step:w_step+fW, h_step:h_step+fH, :] = np.add(dA_temp[:, w_step:w_step+fW, h_step:h_step+fH, :],
                                                                           self._conv_backward_op(self.W, d_prev[:, w, h, :], update_params=False)) 
        self.update_params(dW, optimizer)
        return dA_temp

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
        assert len(filter_size) == 2, "Pooling filter size must be a 2-elements tuple (width, height)."
        assert mode in ["max", "avg"], "Mode of pooling is either max pooling or average pooling."
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode

    def _pool_op(self, slice_a):
        """
        Pooling operation, either max pooling or average pooling.
        """
        if self.mode == "max":
            return np.max(slice_a, axis=(1, 2))
        else:
            return np.average(slice_a, axis=(1, 2))

    def forward(self, X):
        """
        Pooling layer forward propagation. Through this layer, the input dimension will reduce:
            oW = floor((iW - fW)/stride + 1)
            oH = floor((iH - fH)/stride + 1)

        Paramters
        ---------
        X: input tensor to this pooling layer. shape=(m, iW, iH, iC)

        Returns
        -------
        Output tensor that has shape = (m, oW, oH, iC)
        """
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

    def _mask_op(self, slice_a, slice_b, d_prev):
        """
        Compute mask for backpropgation that have the same dimension as previous layer in forward direction.
        """
        m, fW, fH, iC = slice_a.shape
        slice_temp = np.zeros(shape=slice_a.shape)
        for i in range(m):
            for c in range(iC):
                if self.mode == "max":
                    slice_temp[i, :, :, c] = d_prev[i, c]*(slice_a[i, :, :, c] == slice_b[i, c])
                else:
                    slice_temp[i, :, :, c] = d_prev[i, c]/(fW*fH)
        return slice_temp

    def backward(self, d_prev, prev_layer):
        """
        Pooling layer backward propagation.

        Parameters
        ----------
        d_prev: gradient of J respect to A[l+1] of the previous layer according the backward direction.
        prev_layer: previous layer according the forward direction `l-1`.
        
        Returns
        -------
        Gradient of J respect to this pooling layer `l`. The shape out this gradient will equal the shape of prev_layer output
                                                            with corresponding pooling type (max or avg).
        E.g:
            prev_layer output: [[1, 2],         then max: [[0, 0],      or avg: [[1/4, 2/4],
                                [3, 4]]                    [0, 4]]               [3/4, 4/4]]
        """
        m, oW, oH, oC = self.output.shape
        fW, fH = self.filter_size
        dA_temp = np.zeros(shape=prev_layer.output.shape)
        for w in range(oW):
            for h in range(oH):
                w_step = w*self.stride
                h_step = h*self.stride
                dA_temp[:, w_step:w_step+fW, h_step:h_step+fH, :] = self._mask_op(prev_layer.output[:, w_step:w_step+fW,h_step:h_step+fH, :], 
                                                                                    self.output[:, w, h, :], d_prev[:, w, h, :])
        return dA_temp


class FlattenLayer(Layer):

    def __init__(self):
        pass

    def forward(self, X):
        """
        Flatten tensor `X` to a vector.
        """
        m, iW, iH, iC = X.shape
        self.output = np.reshape(X, (m, iW*iH*iC))
        return self.output

    def backward(self, d_prev, prev_layer):
        """
        Reshape d_prev shape to prev_layer output shape in the backpropagation.
        """
        m, iW, iH, iC = prev_layer.output.shape
        d_prev = np.reshape(d_prev, (m, iW, iH, iC))
        return d_prev


class ActivationLayer(Layer):

    def __init__(self, activation):
        """
        activation: (string) available activation functions. Must be in [sigmoid, tanh,
                                relu, softmax]. Softmax activation must be at the last layer.
        
        """
        assert activation in ["sigmoid", "tanh", "relu", "softmax"], "Unknown activation function: " + str(activation)
        self.activation = activation

    def forward(self, X):
        """
        Activation layer forward propgation.
        """
        self.output = eval(self.activation)(X)
        return self.output

    def backward(self, d_prev, _):
        """
        Activation layer backward propagation.

        Parameters
        ---------- 
        d_prev: gradient of J respect to A[l+1] of the previous layer according the backward direction.
        prev_layer: previous layer according the forward direction.
        
        Returns
        -------
        Gradient of J respect to type of activations (sigmoid, tanh, relu) in this layer `l`.
        """
        d_prev = d_prev * eval(self.activation + "_grad")(self.output)
        return d_prev


class DropoutLayer(Layer):

    def __init__(self, keep_prob):
        """
        keep_prob: (float) probability to keep neurons in network, use for dropout technique.
        """
        self.keep_prob = keep_prob


class BatchNormLayer(Layer):

    def __init__(self, momentum=0.99, epsilon=1e-9):
        self.momentum = momentum
        self.epsilon = epsilon

    def forward(self, X, prediction=False):
        """
        Compute batch norm forward.
        LINEAR -> BATCH NORM -> ACTIVATION.

        Returns
        -------
        Output values of batch normalization.
        """
        if not hasattr(self, "gamma") and not hasattr(self, "beta"):
            self.gamma = np.ones(((1,) + X.shape[1:]))
            self.beta = np.zeros(((1,) + X.shape[1:]))
            self.mu_moving_average = np.zeros(shape=self.beta.shape)
            self.sigma_moving_average = np.zeros(shape=self.gamma.shape)
        if not prediction:
            self.mu = np.mean(X, axis=0, keepdims=True)
            self.sigma = np.std(X, axis=0, keepdims=True)
            self.mu_moving_average = self.momentum*(self.mu_moving_average) + (1-self.momentum)*self.mu
            self.sigma_moving_average = self.momentum*(self.sigma_moving_average) + (1-self.momentum)*self.sigma
        else:
            self.mu = self.mu_moving_average
            self.sigma = self.sigma_moving_average    
        self.Xnorm = (X - self.mu)/np.sqrt(self.sigma + self.epsilon)
        self.output = self.gamma*self.Xnorm + self.beta
        return self.output

    def backward(self, d_prev, prev_layer, optimizer):
        """
        Computex batch norm backward.
        LINEAR <- BATCH NORM <- ACTIVATION.
        https://giangtranml.github.io/ml/machine-learning/batch-normalization

        Parameters
        ---------- 
        d_prev: gradient of J respect to A[l+1] of the previous layer according the backward direction.
        prev_layer: previous layer according the forward direction.
        
        Returns
        -------
        dZ: Gradient w.r.t LINEAR function Z.
        """
        m = prev_layer.output.shape[0]
        dXnorm = d_prev * self.gamma
        gamma_grad = np.sum(d_prev * self.Xnorm, axis=0, keepdims=True)
        beta_grad = np.sum(d_prev, axis=0, keepdims=True)
        self.update_params(gamma_grad, beta_grad, optimizer)
        dSigma = np.sum(dXnorm * (-((prev_layer.output - self.mu)*(self.sigma+self.epsilon)**(-3/2))/2),
                       axis=0, keepdims=True)
        dMu = np.sum(dXnorm*(-1/np.sqrt(self.sigma+self.epsilon)), axis=0, keepdims=True) +\
                dSigma*((-2/m)*np.sum(prev_layer.output - self.mu, axis=0, keepdims=True))
        d_prev = dXnorm*(1/np.sqrt(self.sigma+self.epsilon)) + dMu/m +\
                dSigma*((2/m)*np.sum(prev_layer.output - self.mu, axis=0, keepdims=True))
        return d_prev

    def update_params(self, gamma_grad, beta_grad, optimizer):
        updated_gamma_grad = optimizer.minimize(gamma_grad)
        updated_beta_grad = optimizer.minimize(beta_grad)
        self.gamma -= updated_gamma_grad
        self.beta -= updated_beta_grad