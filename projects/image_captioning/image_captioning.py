#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append("..")

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
L = keras.layers
K = keras.backend
import utils
import os

def reset_tf_session():
    curr_session = tf.get_default_session()
    # close current session
    if curr_session is not None:
        curr_session.close()
    # reset graph
    K.clear_session()
    # create new session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.InteractiveSession(config=config)
    K.set_session(s)
    return s

CHECKPOINT_ROOT = "saved_model/"
def get_checkpoint_path():
    return os.path.abspath(CHECKPOINT_ROOT + "weights")

# we take the last hidden layer of IncetionV3 as an image embedding
def get_cnn_encoder():
    K.set_learning_phase(False)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input

    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model

# special tokens  
UNK = "#UNK#"
START = "#START#"
END = "#END#"
PAD = "#PAD#"

# prepare vocabulary
vocab = utils.read_pickle("vocab.pickle")
vocab_inverse = utils.read_pickle("vocab_inverse.pickle")
      
IMG_SIZE = 300
IMG_EMBED_SIZE = 2048
IMG_EMBED_BOTTLENECK = 128
WORD_EMBED_SIZE = 100
LSTM_UNITS = 300
LOGIT_BOTTLENECK = 120
pad_idx = vocab[PAD]

s = reset_tf_session()

class decoder:
    # [batch_size, IMG_EMBED_SIZE] of CNN image features
    img_embeds = tf.placeholder('float32', [None, IMG_EMBED_SIZE])
    # [batch_size, time steps] of word ids
    sentences = tf.placeholder('int32', [None, None])
    
    # we use bottleneck here to reduce the number of parameters
    # image embedding -> bottleneck
    img_embed_to_bottleneck = L.Dense(IMG_EMBED_BOTTLENECK, 
                                      input_shape=(None, IMG_EMBED_SIZE), 
                                      activation='elu')
    # image embedding bottleneck -> lstm initial state
    img_embed_bottleneck_to_h0 = L.Dense(LSTM_UNITS,
                                         input_shape=(None, IMG_EMBED_BOTTLENECK),
                                         activation='elu')
    # word -> embedding
    word_embed = L.Embedding(len(vocab), WORD_EMBED_SIZE)
    # lstm cell 
    lstm_cell = L.LSTMCell(LSTM_UNITS)
    
    # we use bottleneck here to reduce model complexity
    # lstm output -> logits bottleneck
    token_logits_bottleneck = L.Dense(LOGIT_BOTTLENECK, 
                                      input_shape=(None, LSTM_UNITS),
                                      activation="elu")
    # logits bottleneck -> logits for next token prediction
    token_logits = L.Dense(len(vocab),
                           input_shape=(None, LOGIT_BOTTLENECK))
    
    # initial lstm cell state of shape (None, LSTM_UNITS),
    # we need to condition it on `img_embeds` placeholder.
    c0 = h0 = img_embed_bottleneck_to_h0(img_embed_to_bottleneck(img_embeds))

    # embed all tokens but the last for lstm input,
    # remember that L.Embedding is callable,
    # use `sentences` placeholder as input.
    word_embeds = word_embed(sentences[:, :-1])

    # during training we use ground truth tokens `word_embeds` as context for next token prediction.
    # `out_hidden_states` has a shape of [batch_size, time steps, LSTM_UNITS].
    lstm = L.RNN(lstm_cell, return_sequences=True, return_state=True)
    out_hidden_states, _, _ = lstm(word_embeds, initial_state=[c0, h0])
    
    # now we need to calculate token logits for all the hidden states
    # first, we reshape `hidden_states` to [-1, LSTM_UNITS]
    flat_hidden_states = tf.reshape(out_hidden_states, shape=(-1, LSTM_UNITS))

    # then, we calculate logits for next tokens using `token_logits_bottleneck` and `token_logits` layers
    flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states))

    # then, we flatten the ground truth token ids.
    # remember, that we predict next tokens for each time step,
    # use `sentences` placeholder.
    flat_ground_truth = tf.reshape(sentences[:, 1:], shape=(-1,))

    # compute cross-entropy between `flat_ground_truth` and `flat_token_logits` predicted by lstm
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=flat_ground_truth, 
        logits=flat_token_logits
    )

    loss = tf.reduce_mean(xent) 

saver = tf.train.Saver()

class final_model:
    # CNN encoder
    encoder, preprocess_for_model = get_cnn_encoder()
    saver.restore(s, get_checkpoint_path())
    
    # containers for current lstm state
    lstm_c = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="cell")
    lstm_h = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="hidden")

    # input images
    input_images = tf.placeholder('float32', [1, IMG_SIZE, IMG_SIZE, 3], name='images')

    # get image embeddings
    img_embeds = encoder(input_images)

    # initialize lstm state conditioned on image
    init_c = init_h = decoder.img_embed_bottleneck_to_h0(decoder.img_embed_to_bottleneck(img_embeds))
    init_lstm = tf.assign(lstm_c, init_c), tf.assign(lstm_h, init_h)
    
    # current word index
    current_word = tf.placeholder('int32', [1, 1], name='current_input')

    # embedding for current word
    word_embed = decoder.word_embed(current_word)
    
    # compute lstm 
    out_hidden, new_h, new_c = decoder.lstm(word_embed, initial_state=(lstm_h, lstm_c))

    # compute logits for next token
    new_logits = decoder.token_logits(decoder.token_logits_bottleneck(out_hidden))
    
    # compute probabilities for next token
    new_probs = tf.nn.softmax(new_logits)

    # assign state_h and state_c for next prediction
    assign_c = tf.assign(lstm_c, tf.reshape(new_c, shape=(1, LSTM_UNITS)))
    assign_h = tf.assign(lstm_h, tf.reshape(new_h, shape=(1, LSTM_UNITS)))

# this is an actual prediction loop
def generate_caption(image, max_len=20):
    """
    Generate caption for given image.
    """
    s.run(final_model.init_lstm, 
          {final_model.input_images: [image]})
    
    # current caption
    # start with only START token
    caption = [vocab[START]]
   
    for _ in range(max_len):
        next_word_probs, _, _ = s.run([final_model.new_probs, final_model.assign_c, final_model.assign_h], 
                                {final_model.current_word: [[caption[-1]]]})
        next_word_probs = next_word_probs.ravel()
        
        next_word = np.argmax(next_word_probs)

        caption.append(next_word)
        if next_word == vocab[END]:
            break
       
    return list(map(vocab_inverse.get, caption))

# look at validation prediction example
def apply_model_to_image_raw_bytes(raw):
    img = utils.decode_image_from_buf(raw)
    img = utils.crop_and_preprocess(img, (IMG_SIZE, IMG_SIZE), final_model.preprocess_for_model)
    print(' '.join(generate_caption(img)[1:-1]))

apply_model_to_image_raw_bytes(open("mia.jpg", "rb").read())