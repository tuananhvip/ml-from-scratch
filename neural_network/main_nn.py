import sys
sys.path.append("..")
from optimizations_algorithms.optimizers import SGD, SGDMomentum, RMSProp, Adam
from neural_network import NeuralNetwork
from libs.utils import load_dataset_mnist, preprocess_data
from libs.mnist_lib import MNIST

load_dataset_mnist("../libs")
mndata = MNIST('../libs/data_mnist')
training_phase = True
if training_phase:
    images, labels = mndata.load_training()
    images, labels = preprocess_data(images, labels)
    epochs = 20
    batch_size = 64
    learning_rate = 0.1

    sgd = SGD(learning_rate)
    archs = [
        {"num_neurons": 100, "weight_init": "he", "activation": "sigmoid"},
        {"num_neurons": 125, "weight_init": "he", "activation": "sigmoid"},
        {"num_neurons": 50, "weight_init": "he", "activation": "sigmoid"},
        {"num_neurons": labels.shape[1], "weight_init": "he", "activation": "softmax"}]
    nn = NeuralNetwork(epochs, batch_size, sgd, archs, False)
    nn.train(images, labels)
else:
    images_test, labels_test = mndata.load_testing()
    images_test, labels_test = preprocess_data(images_test, labels_test, test=True)

    pred = nn.predict(images_test)

    print("Accuracy:", len(pred[labels_test == pred]) / len(pred))
    from sklearn.metrics.classification import confusion_matrix

    print("Confusion matrix: ")
    print(confusion_matrix(labels_test, pred))