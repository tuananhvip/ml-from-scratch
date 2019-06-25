import numpy as np

def he_initialization(weight_shape):
    """
    Initialize weights according `He initialization` = sqrt(2 / num_input)
    """
    num_input, num_output = weight_shape
    return np.random.normal(0, np.sqrt(2 / num_input), (num_input, num_output))

def xavier_initialization(weight_shape):
    """
    Initialize weights according `Xavier initialization` = sqrt(1 / (num_input + num_output))
    """
    num_input, num_output = weight_shape
    return np.random.normal(0, np.sqrt(1 / (num_input + num_output)), (num_input, num_output))

def standard_normal_initialization(weight_shape):
    """
    Initialize weights according standard normal distribution with mean 0 variance 1.
    """
    num_input, num_output = weight_shape
    return np.random.normal(size=(num_input, num_output))
