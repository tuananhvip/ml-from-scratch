import os
import sys
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import imageio

sess = tf.InteractiveSession()


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_C_unrolled = tf.reshape(a_C, shape=(m, n_H * n_W, n_C))
    a_G_unrolled = tf.reshape(a_G, shape=(m, n_H * n_W, n_C))
    
    J_content = tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled))/(4*n_H*n_W*n_C)
    
    return J_content

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    GA = tf.matmul(A, tf.transpose(A))
    
    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.transpose(tf.reshape(a_S, shape=(n_H*n_W, n_C)))
    a_G = tf.transpose(tf.reshape(a_G, shape=(n_H*n_W, n_C)))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum(tf.square(GS - GG))/(4 * n_C**2 * (n_H*n_W)**2)
        
    return J_style_layer

STYLE_LAYERS = [
    ('conv1_2', 0.2),
    ('conv2_2', 0.2),
    ('conv3_3', 0.2),
    ('conv4_3', 0.2),
    ('conv5_3', 0.2)]

def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    
    J = alpha*J_content + beta*J_style
    
    return J


def main(image_content_file, image_style_file, iterations):
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    if iterations is None:
        iterations = 60

    def model_nn(sess, input_image, num_iterations = iterations):
    
        sess.run(tf.global_variables_initializer())

        sess.run(model['input'].assign(input_image))
        
        for i in range(num_iterations):
        
            sess.run(train_step)

            generated_image = sess.run(model['input'])
        
        save_image('output/generated_image.jpg', generated_image)
        
        return generated_image

    content_image = imageio.imread(image_content_file)
    style_image = imageio.imread(image_style_file)

    content_image = sess.run(tf.image.resize_images(content_image, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH), 1))
    content_image = normalize_image(content_image)

    style_image = sess.run(tf.image.resize_images(style_image, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH), 1))
    style_image = normalize_image(style_image)

    generated_image = generate_noise_image(content_image)

    # Assign the content image to be the input of the VGG model.  
    sess.run(model['input'].assign(content_image))

    # Select the output tensor of layer conv4_2
    out = model['conv2_2']

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)

    # Assign the input of the model to be the "style" image 
    sess.run(model['input'].assign(style_image))

    # Compute the style cost
    J_style = compute_style_cost(model, STYLE_LAYERS)

    J = total_cost(J_content, J_style, alpha=10, beta=40)

    optimizer = tf.train.AdamOptimizer(2.0)

    train_step = optimizer.minimize(J)

    generated_image = model_nn(sess, generated_image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple art generation.")
    parser.add_argument("-c", dest="content")
    parser.add_argument("-s", dest="style")
    parser.add_argument("--iters", dest="iterations")

    arg = parser.parse_args()

    main(arg.content, arg.style, arg.iterations)