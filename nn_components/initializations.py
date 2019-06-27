import numpy as np

def he_initialization(weight_shape):
    """
    Initialize weights according `He initialization` = sqrt(2 / num_input)
    """
    if len(weight_shape) == 4:
        fW, fH, fC, num_fitls = weight_shape
        return np.random.normal(0, np.sqrt(2 / (fW*fH*fC*num_fitls)), weight_shape)
    num_input, num_output = weight_shape
    return np.random.normal(0, np.sqrt(2 / num_input), weight_shape)

def xavier_initialization(weight_shape):
    """
    Initialize weights according `Xavier initialization` = sqrt(1 / (num_input + num_output))
    """
    if len(weight_shape) == 4:
        fW, fH, fC, num_fitls = weight_shape
        return np.random.normal(0, np.sqrt(1 / (fW*fH*fC*num_fitls)), weight_shape)
    num_input, num_output = weight_shape
    return np.random.normal(0, np.sqrt(1 / (num_input + num_output)), weight_shape)

def standard_normal_initialization(weight_shape):
    """
    Initialize weights according standard normal distribution with mean 0 variance 1.
    """
    return np.random.normal(size=weight_shape)
