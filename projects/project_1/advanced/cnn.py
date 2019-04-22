import tensorflow as tf
from utils import load_dataset_mnist
from mnist import MNIST
from sklearn.preprocessing import StandardScaler
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
        print(tf.shape(X_train))
        for e in range(self.epoch):
            with tf.GradientTape() as tape:
                logits = self.model(inputs=X_train, training=True)
                loss_value = self.loss_function(y_train, logits)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            print(loss_value)

    def conv_layers(self):
        self.model.add(tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=6, kernel_size=(5, 5), activation='tanh', padding='same'))
        self.model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))
        self.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='tanh', padding='same'))
        self.model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))

    def fc_layers(self):
        self.model.add(tf.keras.layers.Dense(units=120, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(units=84, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(units=10, activation='softmax'))


if __name__ == '__main__':
    load_dataset_mnist()

    mndata = MNIST('data_mnist')

    images, labels = mndata.load_training()

    scaler = StandardScaler()
    scaler.fit(images)
    images = scaler.transform(images)

    images = images.reshape((-1, 28, 28, 1))

    lenet = Lenet(100, 64, tf.keras.optimizers.SGD, tf.keras.losses.categorical_crossentropy)
    lenet.train(images, labels)