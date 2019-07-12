import sys
sys.path.append("..")
import os
from libs.utils import load_dataset_mnist, preprocess_data
from libs.mnist_lib import MNIST
from optimizations_algorithms.optimizers import SGD, SGDMomentum, RMSProp, Adam
from convolutional_neural_network import CNN
import pickle

def main():
    load_dataset_mnist("../libs")
    mndata = MNIST('../libs/data_mnist')
    lenet_arch = [{"type": "conv", "filter_size": (5, 5), "filters": 6, "padding": "SAME", "stride": 1, "activation": "relu", "weight_init": "he"},
                {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
                {"type": "conv", "filter_size": (5, 5), "filters": 16, "padding": "SAME", "stride": 1, "activation": "relu", "weight_init": "he"},
                {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
                "flatten",
                {"type": "fc", "num_neurons": 120, "weight_init": "std", "activation": "tanh"},
                {"type": "fc", "num_neurons": 84, "weight_init": "std", "activation": "tanh"},
                {"type": "fc", "num_neurons": 10, "weight_init": "std", "activation": "softmax"}
                ]
    epochs = 20
    batch_size = 32
    learning_rate = 0.1
    optimizer = SGD(alpha=learning_rate)
    cnn = CNN(epochs=epochs, batch_size=batch_size, optimizer=optimizer, cnn_structure=lenet_arch)
    weight_path = "cnn_weights.pickle"
    training_phase = weight_path not in os.listdir(".")
    if training_phase:
        images, labels = mndata.load_training()
        images, labels = preprocess_data(images, labels, nn=True)
        cnn.train(images[:10000], labels[:10000])
    else:
        images_test, labels_test = mndata.load_testing()
        images_test, labels_test = preprocess_data(images_test, labels_test, test=True)

        with open(weight_path, "rb") as f:
            cnn = pickle.load(f) 
        pred = cnn.predict(images_test[:2000])

        print("Accuracy:", len(pred[labels_test[:2000] == pred]) / len(pred))
        from sklearn.metrics.classification import confusion_matrix

        print("Confusion matrix: ")
        print(confusion_matrix(labels_test[:2000], pred))

if __name__ == "__main__":
    main()