"""
Author: Giang Tran
Email: giangtran240896@gmail.com
Docs: https://giangtranml.github.io/ml/machine-learning/recurrent-neural-network
"""

import sys
from neural_network.neural_network import NeuralNetwork
from nn_components.activations import softmax, tanh, sigmoid, tanh_grad
import numpy as np
from tqdm import tqdm


class RNN(NeuralNetwork):

    def __init__(self, hidden_units, epochs, optimizer, batch_size):
        """
        Constructor for Recurrent Neural Network. 
        """
        self.hidden_units = hidden_units
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    def _initialize_params(self, X_train, Y_train):
        """
        Initialize all neccessary parameters once at the start.
        """
        m, time_steps, vector_len = X_train.shape
        _, _, vocab_len = Y_train.shape
        self.Wax = np.random.normal(size=(vector_len, self.hidden_units))
        self.Waa = np.random.normal(size=(self.hidden_units, self.hidden_units))
        self.Wy = np.random.normal(size=(self.hidden_units, vocab_len))
        self.ba = np.zeros(shape=(1, self.hidden_units))
        self.by = np.zeros(shape=(1, vocab_len))
        self.A_state = np.zeros(shape=(m, time_steps, self.hidden_units))
        self.a_state = np.zeros(shape=(m, self.hidden_units))
    
    def _forward(self, X_timestamp, Y_timestamp, state):
        """
        RNN forward propagation.
        """
        state = tanh(X_timestamp.dot(self.Wax) + state.dot(self.Waa) + self.ba)
        Y_hat_timestamp = sigmoid(state.dot(self.Wy) + self.by)
        loss = self._loss(Y_timestamp, Y_hat_timestamp)
        return loss, Y_hat_timestamp, state

    def _backward(self, X_train, Y_train, Y_hat):
        """
        X_train: shape=(m, time_steps, vector_length)
        Y_train: shape=(m, time_steps, vocab_length)
        Y_hat: shape=(m, time_steps, vocab_length)
        """
        m, time_steps, vector_len = X_train.shape
        dWy = np.zeros(shape=self.Wy.shape)
        dby = np.zeros(shape=self.by.shape)
        dWaa = np.zeros(shape=self.Waa.shape)
        dWax = np.zeros(shape=self.Wax.shape)
        dba = np.zeros(shape=self.ba.shape)
        dA_chain = np.zeros(shape=self.A_state.shape)
        da_state = np.zeros(shape=self.a_state.shape)
        for t in range(time_steps):
            # Compute dA[t] respect to dA[t-1]
            dA_chain[:, t, :] = tanh_grad(self.A_state[:, t, :]).dot(self.Waa.T)
        for t in reversed(range(time_steps)):
            delta = (Y_hat[:, t, :] - Y_train[:, t, :])/m # shape = (m, vocab_len)
            dWy += self.A_state[:, t, :].T.dot(delta)
            dby += np.sum(delta, axis=0)
            for k in range(0, t):
                same_chain = (delta.dot(self.Wy.T) # shape=(m, hidden_units)
                                * np.prod(dA_chain[:, k+1:t, :], axis=1) # shape=(m, hidden_units)
                                * tanh_grad(self.A_state[:, k, :]) # shape=(m, hidden_units)
                                )
                if k == 0:
                    dWaa += np.dot(self.a_state.T, same_chain)
                else:
                    dWaa += np.dot(self.A_state[:, k-1, :].T, same_chain)

                dWax += np.dot(X_train[:, k, :].T, same_chain)
                dba += np.sum(same_chain, axis=0)
            da_state = np.dot(np.prod(dA_chain[:, 1:t, :], axis=1)*tanh_grad(self.A_state[:, 0, :]), self.Waa)
        self.update_params(dWy, dby, dWaa, dWax, dba, da_state)

    def train(self, X_train, Y_train):
        """
        X_train: shape=(m, time_steps, vector_length)
        Y_train: shape=(m, time_steps, vocab_length)
        """
        m, time_steps, vector_len = X_train.shape
        self._initialize_params(X_train, Y_train)
        Y_hat = np.zeros(shape=Y_train.shape)
        for e in range(self.epochs):
            batch_loss = 0
            num_batches = 0
            pbar = tqdm(range(0, X_train.shape[0], self.batch_size), desc="Epoch " + str(e+1))
            for it in pbar:
                total_timesteps_loss = 0
                for t in range(time_steps):
                    state = self.a_state if t == 0 else self.A_state[:, t-1, :]
                    loss_t, Y_hat_t, A_state_t = self._forward(X_train[:, t, :], Y_train[:, t, :], state)
                    total_timesteps_loss += loss_t
                    Y_hat[:, t, :] = Y_hat_t
                    self.A_state[:, t, :] = A_state_t
                self._backward(X_train, Y_train, Y_hat)
                batch_loss += total_timesteps_loss
                num_batches += 1
                pbar.set_description("Epoch " +str(e+1) + " - Loss: %.4f" % (batch_loss/num_batches))
            print("Loss at epoch %s: %f" % (e + 1 , batch_loss / num_batches))

    def update_params(self, dWy, dby, dWaa, dWax, dba, da_state):
        """
        Update parameters of RNN by its gradient. 
        """
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