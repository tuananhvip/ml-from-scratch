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
        num_classes = len(np.unique(y_train))

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

    def data_augmentation(self, X_train):
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90,
                                                                  width_shift_range=0.1,
                                                                  height_shift_range=0.1,
                                                                  horizontal_flip=True)
        self.datagen.fit(X_train)

    def _train(self, X_train, y_train):
        test_x = X_train[0].reshape((1, 32, 32, 3))
        abc = []
        for e in range(self.epochs):
            batches = 0
            for x_batch, y_batch in self.datagen.flow(test_x, y_train[0], batch_size=1):
                batches += 1
                abc.append(x_batch)
                # if batches >= test_x.shape[0] / self.batch_size:
                #     break

    def train(self, X_train, y_train):
        self.hidden_layers(X_train, y_train)
        self.data_augmentation(X_train)
        self._train(X_train, y_train)
        # self.model.fit_generator(self.datagen.flow(X_train, y_train, batch_size=self.batch_size),
        #                          steps_per_epoch=X_train.shape[0]/self.batch_size,
        #                          epochs=self.epochs)

def main():
    learning_rate = 0.01
    epochs = 1
    batch_size = 32
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss_function = tf.keras.losses.categorical_crossentropy

    cnn = CNN(epochs, batch_size, optimizer, loss_function)

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    cnn.train(X_train, y_train)

if __name__ == '__main__':
    main()
