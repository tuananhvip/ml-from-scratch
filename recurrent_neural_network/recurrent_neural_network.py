import sys
sys.path.append("..")
from neural_network.neural_network import NeuralNetwork
from nn_components.activations import softmax, tanh
import numpy as np


class RNN(NeuralNetwork):

    def __init__(self, hidden_units):
        self.hidden_units = hidden_units
        self.a_state = None 
        self.Wa = None
        self.Wy = None

    def _initialize_params(self, train_X, train_Y):
        m, time_steps, vector_len = train_X.shape
        m, vocab_len = train_Y.shape
        self.Wax = np.random.normal(size=(vector_len, self.hidden_units))
        self.Waa = np.random.normal(size=(self.hidden_units, self.hidden_units))
        self.Wy = np.random.normal(size=(self.hidden_units, vocab_len))
        self.ba = np.zeros(shape=(1, self.hidden_units))
        self.by = np.zeros(shape=(1, vocab_len))
        self.a_state = np.zeros(shape=(m, self.hidden_units))
    
    def _forward(self, X_timestamp):
        """
        RNN forward propagation.
        """
        self.a_state = np.tanh(X_timestamp.dot(self.Wax) + self.a_state.dot(self.Waa) + self.ba)
        y_hat = tanh(self.a_state.dot(self.Wy) + self.by)

    def _backward(self, train_X, train_Y, Y_hat):
        m, time_steps, vector_len = train_X.shape
        pass

    def train(self, train_X, train_Y):
        """
        train_X: shape=(m, time_steps, vector_length)
        train_Y: shape=(m, vocab_length)
        """
        m, time_steps, vector_len = train_X.shape
        self._initialize_params(train_X, train_Y)
        for t in range(time_steps):
            self._forward(train_X[:, t, :])
        