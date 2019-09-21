"""
Author: Giang Tran
Email: giangtran240896@gmail.com
Docs: https://giangtranml.github.io/ml/machine-learning/recurrent-neural-network
Note: not correctly implemented yet!
"""
from nn_components.activations import softmax, tanh, tanh_grad
from neural_network.neural_network import NeuralNetwork
import numpy as np
from tqdm import tqdm


class RecurrentNeuralNetwork:

    def __init__(self, hidden_units, epochs, optimizer, batch_size):
        """
        Constructor for Recurrent Neural Network. 
        """
        self.hidden_units = hidden_units
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    def _loss(self, Y, Y_hat):
        """
        Loss function for many to many RNN.

        Parameters
        ----------
        Y: one-hot encoding label tensor. shape = (N, T, C)
        Y_hat: output at each time step. shape = (N, T, C)
        """
        return -np.mean(np.sum(Y*np.log(Y_hat), axis=(1, 2)))

    def _forward(self, X):
        """
        RNN forward propagation.

        Parameters
        ----------
        X: time series input, shape = (N, T, D)
        """
        m, timesteps, _ = X.shape
        self.h0 = np.zeros(shape=(m, self.hidden_units))
        self.states = np.zeros(shape=(m, timesteps, self.hidden_units))
        self.states[:, 0, :] = tanh(np.dot(X[:, 0, :], self.Wax) + np.dot(self.h0, self.Waa) + self.ba)
        for t in range(1, timesteps):
            self.states[:, t, :] = tanh(np.dot(X[:, t, :], self.Wax) + np.dot(self.states[:, t-1, :], self.Waa) + self.ba)
        Y_hat = np.einsum("nth,hc->ntc", self.states, self.Wy)
        Y_hat = softmax(Y_hat + self.by)
        return Y_hat

    def _backward(self, X_train, Y_train, Y_hat):
        """
        X_train: shape=(m, time_steps, vector_length)
        Y_train: shape=(m, time_steps, vocab_length)
        Y_hat: shape=(m, time_steps, vocab_length)
        """
        m, time_steps, vector_len = X_train.shape
        dWaa = np.zeros(shape=self.Waa.shape)
        dWax = np.zeros(shape=self.Wax.shape)
        dba = np.zeros(shape=self.ba.shape)
        dA_chain = np.zeros(shape=self.A_state.shape)
        da_state = np.zeros(shape=self.a_state.shape)

        delta = (Y_hat - Y_train)/m
        dWy = np.einsum("ntc,nth->hc", delta, self.states)
        dby = np.sum(delta, axis=(0, 1))

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
        _, _, vocab_len = Y_train.shape
        self.Wax = np.random.normal(size=(vector_len, self.hidden_units))
        self.Waa = np.random.normal(size=(self.hidden_units, self.hidden_units))
        self.Wy = np.random.normal(size=(self.hidden_units, vocab_len))
        self.ba = np.zeros(shape=(1, self.hidden_units))
        self.by = np.zeros(shape=(1, vocab_len))
        Y_hat = np.zeros(shape=Y_train.shape)
        for e in range(self.epochs):
            batch_loss = 0
            num_batches = 0
            pbar = tqdm(range(0, X_train.shape[0], self.batch_size), desc="Epoch " + str(e+1))
            for it in pbar:
                Y_hat = self._forward(X_train[it:it+self.batch_size])
                loss = self._loss(Y_train[it:it+self.batch_size], Y_hat)
                self._backward(X_train[it:it+self.batch_size], Y_train[it:it+self.batch_size], Y_hat)
                batch_loss += loss
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
