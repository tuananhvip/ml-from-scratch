import tensorflow as tf
from utils import load_dataset_mnist, preprocess_data
from mnist import MNIST
from sklearn.preprocessing import StandardScaler
from numpy import array
import os
tf.enable_eager_execution()


class Lenet:

    def __init__(self, epoch, batch_size, optimizer, loss_function):
        self.model = tf.keras.Sequential()
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.conv_layers()
        self.fc_layers()

    def train(self, X_train, y_train):
        X_train = tf.cast(X_train, tf.float32)

        y_train = tf.one_hot(y_train, 10)
        for e in range(self.epoch):
            batch_loss = 0
            num_batches = 0
            it = 0
            while it < X_train.shape[0]:
                with tf.GradientTape() as tape:
                    logits = self.model(inputs=X_train[it:it+self.batch_size, ], training=True)
                    loss_value = self.loss_function(y_train[it:it+self.batch_size, ], logits)
                    grads = tape.gradient(loss_value, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    batch_loss += loss_value
                it += self.batch_size
                num_batches += 1
            print(batch_loss / num_batches)

            if (e + 1) % 5 == 0:
                self.model.save_weights("./saved_model/")

    def conv_layers(self):
        self.model.add(tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=6, kernel_size=(5, 5), activation='tanh', padding='same')) # (n, 24, 24, 6)
        self.model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')) # (n, 12, 12, 6)
        self.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='tanh', padding='same')) # (n, 8, 8, 16)
        self.model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')) # (n, 4, 4, 16)
        self.model.add(tf.keras.layers.Flatten())

    def fc_layers(self):
        self.model.add(tf.keras.layers.Dense(units=120, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(units=84, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    def predict(self, X_test):
        # self.model.load_weights(tf.train.latest_checkpoint("./saved_model/"))
        X_test = tf.cast(X_test, tf.float32)
        self.model.load_weights("./saved_model/")
        predictions = self.model(inputs=X_test)
        return tf.argmax(predictions, axis=1).numpy()


def mnist_classification():
    training_phase = "saved_model" not in os.listdir()
    load_dataset_mnist()
    mndata = MNIST('data_mnist')

    lenet = Lenet(20, 64, tf.train.AdamOptimizer(learning_rate=0.001), tf.losses.softmax_cross_entropy)

    if training_phase:
        images, labels = mndata.load_training()
        images, labels = preprocess_data(images, labels, True)
        lenet.train(images, labels)
    else:
        images_test, labels_test = mndata.load_testing()
        images_test, labels_test = preprocess_data(images_test, labels_test, True)

        pred = lenet.predict(images_test)
        print("Accuracy:", len(labels_test[pred == labels_test]) / len(labels_test))  # 98%

        from sklearn.metrics.classification import confusion_matrix
        print("Confusion matrix: ")
        print(confusion_matrix(labels_test, pred))


if __name__ == '__main__':
    mnist_classification()