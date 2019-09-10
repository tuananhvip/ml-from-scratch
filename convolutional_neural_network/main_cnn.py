import sys
sys.path.append("..")
sys.path.append("D:\ml_from_scratch")
import os
from libs.utils import load_dataset_mnist, preprocess_data
from libs.mnist_lib import MNIST
from optimizations_algorithms.optimizers import SGD, SGDMomentum, RMSProp, Adam
from convolutional_neural_network import CNN
from cnn_keras import CNNKeras


def main(use_keras=False):
    load_dataset_mnist("D:/ml_from_scratch/libs")
    mndata = MNIST('D:/ml_from_scratch/libs/data_mnist')
    arch = [{"type": "conv", "filter_size": (3, 3), "filters": 6, "padding": "SAME", "stride": 1, "activation": "relu", "weight_init": "he"},
            {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
            {"type": "conv", "filter_size": (3, 3), "filters": 16, "padding": "SAME", "stride": 1, "activation": "relu", "weight_init": "he"},
            {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
            {"type": "conv", "filter_size": (3, 3), "filters": 32, "padding": "SAME", "stride": 1, "activation": "relu", "weight_init": "he"},
            {"type": "pool", "filter_size": (2, 2), "stride": 2, "mode": "max"},
            "flatten",
            {"type": "fc", "num_neurons": 128, "weight_init": "he", "activation": "relu"}, # use "batch_norm": None
            {"type": "fc", "num_neurons": 64, "weight_init": "he", "activation": "relu"},
            {"type": "fc", "num_neurons": 10, "weight_init": "he", "activation": "softmax"}
            ]
    epochs = 5
    batch_size = 64
    learning_rate = 0.01
    if use_keras:
        print("Train MNIST dataset by CNN with Keras/Tensorflow.")
        from keras.optimizers import SGD as SGDKeras
        training_phase = True
        optimizer = SGDKeras(lr=learning_rate)
        cnn = CNNKeras(epochs=epochs, batch_size=batch_size, optimizer=optimizer, cnn_structure=arch)
    else:
        print("Train MNIST dataset by CNN with pure Python: Numpy.")
        optimizer = SGD(alpha=learning_rate)
        cnn = CNN(epochs=epochs, batch_size=batch_size, optimizer=optimizer, cnn_structure=arch)
        weight_path = "cnn_weights.pickle"
        training_phase = weight_path not in os.listdir(".")
    if training_phase:
        images, labels = mndata.load_training()
        images, labels = preprocess_data(images, labels, nn=True)
        
        if not use_keras:
            cnn.train(images, labels)
            cnn.save(weight_path)
        else:
            cnn.train(images, labels)
            training_phase = False
    if not training_phase:
        import pickle
        images_test, labels_test = mndata.load_testing()
        images_test, labels_test = preprocess_data(images_test, labels_test, nn=True, test=True)
        if not use_keras:
            with open(weight_path, "rb") as f:
                cnn = pickle.load(f)
        pred = cnn.predict(images_test)

        print("Accuracy:", len(pred[labels_test == pred]) / len(pred))
        from sklearn.metrics.classification import confusion_matrix

        print("Confusion matrix: ")
        print(confusion_matrix(labels_test, pred))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A CNN program.")
    parser.add_argument("--keras", action="store_true", help="Whether use keras or not.")

    args = parser.parse_args()
    main(use_keras=args.keras)