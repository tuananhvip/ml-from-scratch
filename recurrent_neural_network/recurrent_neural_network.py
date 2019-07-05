import sys
# sys.path.append("/home/james/Desktop/ml-from-scratch")
from neural_network.neural_network import NeuralNetwork
from nn_components.activations import softmax, tanh, sigmoid, tanh_grad
import numpy as np


class RNN(NeuralNetwork):

    def __init__(self, hidden_units, epochs, optimizer):
        self.hidden_units = hidden_units
        self.optimizer = optimizer
        self.epochs = epochs

    def _initialize_params(self, train_X, train_Y):
        m, time_steps, vector_len = train_X.shape
        _, _, vocab_len = train_Y.shape
        self.Wax = np.random.normal(size=(vector_len, self.hidden_units))
        self.Waa = np.random.normal(size=(self.hidden_units, self.hidden_units))
        self.Wy = np.random.normal(size=(self.hidden_units, vocab_len))
        self.ba = np.zeros(shape=(1, self.hidden_units))
        self.by = np.zeros(shape=(1, vocab_len))
        self.A_state = np.zeros(shape=(m, time_steps, self.hidden_units))
        self.a_state = np.zeros(shape=(m, self.hidden_units))
    
    def _forward(self, X_timestamp, Y_timestamp):
        """
        RNN forward propagation.
        """
        self.a_state = tanh(X_timestamp.dot(self.Wax) + self.a_state.dot(self.Waa) + self.ba)
        Y_hat_timestamp = sigmoid(self.a_state.dot(self.Wy) + self.by)
        loss = self._loss(Y_timestamp, Y_hat_timestamp)
        return loss, Y_hat_timestamp, self.a_state

    def _backward(self, train_X, train_Y, Y_hat):
        """
        train_X: shape=(m, time_steps, vector_length)
        train_Y: shape=(m, time_steps, vocab_length)
        Y_hat: shape=(m, time_steps, vocab_length)
        """
        m, time_steps, vector_len = train_X.shape
        dWy = np.zeros(shape=self.Wy.shape)
        dby = np.zeros(shape=self.by.shape)
        dWaa = np.zeros(shape=self.Waa.shape)
        dWax = np.zeros(shape=self.Wax.shape)
        dba = np.zeros(shape=self.ba.shape)
        dA_chain = np.zeros(shape=self.A_state.shape)
        da_state = np.zeros(shape=self.a_state.shape)
        for t in range(time_steps):
            dA_chain[:, t, :] = tanh_grad(self.A_state[:, t, :]).dot(self.Waa) # shape = (m, hidden_units)
        for t in reversed(range(time_steps)):
            delta = (Y_hat[:, t, :] - train_Y[:, t, :]) # shape = (m, vocab_len)
            dWy += self.A_state[:, t, :].T.dot(delta)
            dby += np.sum(delta, axis=0)
            for k in range(1, t):
                same_chain = (delta.dot(self.Wy.T) # shape=(m, hidden_units)
                                * np.prod(dA_chain[:, k+1:t, :], axis=1) # shape=(m, hidden_units)
                                * tanh_grad(self.A_state[:, k, :]) # shape=(m, hidden_units)
                                )
                dWaa += np.dot(self.A_state[:, k-1, :].T, same_chain)

                dWax += np.dot(train_X[:, k, :].T, same_chain)
                dba += np.sum(same_chain, axis=0)
            da_state = np.dot(np.prod(dA_chain[:, :t, :], axis=1)*tanh_grad(self.a_state), self.Waa)
        self.update_params(dWy, dby, dWaa, dWax, dba, da_state)

    def train(self, train_X, train_Y):
        """
        train_X: shape=(m, time_steps, vector_length)
        train_Y: shape=(m, time_steps, vocab_length)
        """
        m, time_steps, vector_len = train_X.shape
        self._initialize_params(train_X, train_Y)
        Y_hat = np.zeros(shape=train_Y.shape)
        for _ in range(self.epochs):
            total_loss = 0
            for t in range(time_steps):
                loss_t, Y_hat_t, A_state_t = self._forward(train_X[:, t, :], train_Y[:, t, :])
                total_loss += loss_t
                Y_hat[:, t, :] = Y_hat_t
                self.A_state[:, t, :] = A_state_t
            print(total_loss)
            self._backward(train_X, train_Y, Y_hat)

    def update_params(self, dWy, dby, dWaa, dWax, dba, da_state):
        dWy = self.optimizer.minimize(dWy)
        dby = self.optimizer.minimize(dby)
        dWaa = self.optimizer.minimize(dWaa)
        dWax = self.optimizer.minimize(dWax)
        dba = self.optimizer.minimize(dba)
        da_state = self.optimizer.minimize(da_state)

        self.Wy -= dWy
        self.by -= dby
        self.Waa -= dWaa
        self.Wax -= dWax
        self.ba -= dba
        self.a_state -= da_state