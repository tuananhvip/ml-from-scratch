import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
tf.enable_eager_execution()


class CNN:

    def __init__(self, epochs, batch_size, optimizer, loss_function):
        self.model = tf.keras.Sequential()
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.hidden_layers()

    def hidden_layers(self):
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Dropout(0.2))

        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Dropout(0.3))

        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Dropout(0.4))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(10, activation='softmax'))
        self.model.summary()

    def data_augmentation(self, X_train):
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90,
                                                                  width_shift_range=0.1,
                                                                  height_shift_range=0.1,
                                                                  horizontal_flip=True)
        self.datagen.fit(X_train)

    def _train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
        X_train = tf.cast(X, tf.float32)
        X_val = tf.cast(X_val, tf.float32)
        y_train = tf.one_hot(y, 10)
        y_val = tf.one_hot(y_val, 10)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=['accuracy'])
        self.model.fit_generator(self.datagen.flow(X_train, y_train, batch_size=self.batch_size),
                            steps_per_epoch=X_train.shape[0] / self.batch_size, epochs=self.epochs,
                            verbose=1, validation_data=(X_val, y_val))
        model_json = self.model.to_json()
        with open('model.json', 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights('model.h5')

    def load_model(self):
        self.model.load_weights("./saved_model/cifar/cnn")
        print("----> LOADED WEIGHTS")

    def train(self, X_train, y_train):
        self.data_augmentation(X_train)
        self._train(X_train, y_train)

    def predict(self, X_test):
        predictions = self.model(inputs=X_test)
        return tf.argmax(predictions, axis=1).numpy()


def main():
    training_phase = None
    learning_rate = 0.01
    epochs = 125
    batch_size = 32
    optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate, decay=1e-6)
    loss_function = tf.keras.losses.categorical_crossentropy
    cnn = CNN(epochs, batch_size, optimizer, loss_function)

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train / 255
    X_test = X_test / 255
    cnn.train(X_train, y_train)

    from keras.preprocessing.sequence import TimeseriesGenerator


if __name__ == '__main__':
    main()
