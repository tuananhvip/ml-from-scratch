import tensorflow as tf
import numpy as np
tf.enable_eager_execution()


class CNN:

    def __init__(self, epochs, batch_size, optimizer, loss_function):
        self.model = tf.keras.Sequential()
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_function = loss_function

    def hidden_layers(self, X_train, y_train):
        num_classes = np.unique(y_train)

        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
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
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    def data_augmentation(self, image):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90,
                                                                  width_shift_range=0.1,
                                                                  height_shift_range=0.1,
                                                                  horizontal_flip=True)
        datagen.fit(image)

    def _train(self):
        pass

    def train(self, X_train, y_train):
        self.hidden_layers(X_train, y_train)

def main():
    learning_rate = 0.01
    epochs = 100
    batch_size = 32
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss_function = tf.keras.losses.categorical_crossentropy

    cnn = CNN(epochs, batch_size, optimizer, loss_function)

if __name__ == '__main__':
    main()
