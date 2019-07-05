import sys
sys.path.append("..")
sys.path.append("/home/james/Desktop/ml-from-scratch")
from recurrent_neural_network import RNN
import numpy as np
from keras.utils.np_utils import to_categorical
from optimizations_algorithms.optimizers import SGD

def main():
    start_token = " "
    pad_token = "#"

    with open("/home/james/Desktop/ml-from-scratch/recurrent_neural_network/names") as f:
        names = f.read()[:-1].split('\n')
        names = [start_token + name for name in names]
    print('number of samples:', len(names))
    MAX_LENGTH = max(map(len, names))
    print("max length:", MAX_LENGTH)

    tokens = set()
    for name in names:
        temp_name = set(list(name))
        for t_n in temp_name:
            tokens.add(t_n)
    tokens.add("#")
        
    tokens = list(tokens)
    n_tokens = len(tokens)
    print ('n_tokens:', n_tokens)

    token_to_id = dict() 
    for ind, token in enumerate(tokens):
        token_to_id[token] = ind

    def to_matrix(names, max_len=None, pad=token_to_id[pad_token], dtype=np.int32):
        """Casts a list of names into rnn-digestable padded matrix"""
    
        max_len = max_len or max(map(len, names))
        names_ix = np.zeros([len(names), max_len], dtype) + pad
        for i in range(len(names)):
            name_ix = list(map(token_to_id.get, names[i]))
            names_ix[i, :len(name_ix)] = name_ix

        return names_ix
        
    matrix_sequences = to_matrix(names)
    m, length = matrix_sequences.shape
    input_sequences = np.zeros(shape=(m, length, n_tokens))
    for i in range(m):
        input_sequences[i] = to_categorical(matrix_sequences[i], n_tokens, dtype='int32')
    del matrix_sequences
    train_X = input_sequences[:, :-1, :]
    train_Y = input_sequences[:, 1:, :]

    optimizer = SGD()
    epochs = 2

    rnn = RNN(hidden_units=64, epochs=epochs, optimizer=optimizer)
    rnn.train(train_X, train_Y)


if __name__ == "__main__":
    main()