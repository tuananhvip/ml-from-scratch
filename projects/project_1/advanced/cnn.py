import tensorflow as tf
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
        for e in range(self.epoch):
            with tf.GradientTape() as tape:
                logits = self.model(inputs=X_train, training=True)
                loss_value = self.loss_function(y_train, logits)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def conv_layers(self):
        self.model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5)))
        self.model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2))
        self.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5)))
        self.model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2))

    def fc_layers(self):
        self.model.add(tf.keras.layers.Dense(units=120, activation=tf.keras.activations.tanh))
        self.model.add(tf.keras.layers.Dense(units=84))


tf.keras.optimizers.SGD
tf.keras.losses.categorical_crossentropy