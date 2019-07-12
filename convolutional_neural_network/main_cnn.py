import sys
sys.path.append("..")
import os
from libs.utils import load_dataset_mnist, preprocess_data
from libs.mnist_lib import MNIST
from optimizations_algorithms.optimizers import SGD, SGDMomentum, RMSProp, Adam
from convolutional_neural_network import CNN

def main():
    load_dataset_mnist("../libs")
    mndata = MNIST('../libs/data_mnist')
    arch = [{"type": "conv", "filter_size": (3, 3), "filters": 6, "padding": "SAME", "stride": 1, "activation": "relu", "weight_init": "he"},
            {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
            {"type": "conv", "filter_size": (3, 3), "filters": 16, "padding": "SAME", "stride": 1, "activation": "relu", "weight_init": "he"},
            {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
            {"type": "conv", "filter_size": (3, 3), "filters": 32, "padding": "SAME", "stride": 1, "activation": "relu", "weight_init": "he"},
            {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
            "flatten",
            {"type": "fc", "num_neurons": 128, "weight_init": "he", "activation": "relu"},
            {"type": "fc", "num_neurons": 64, "weight_init": "he", "activation": "relu"},
            {"type": "fc", "num_neurons": 10, "weight_init": "he", "activation": "softmax"}
            ]
    epochs = 5
    batch_size = 32
    learning_rate = 0.1
    optimizer = SGD(alpha=learning_rate)
    cnn = CNN(epochs=epochs, batch_size=batch_size, optimizer=optimizer, cnn_structure=arch)
    weight_path = "cnn_weights.pickle"
    training_phase = weight_path not in os.listdir(".")
    if training_phase:
        images, labels = mndata.load_training()
        images, labels = preprocess_data(images, labels, nn=True)
        cnn.train(images[:10000], labels[:10000])
    else:
        import pickle
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