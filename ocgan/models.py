from tensorflow import keras
import tensorflow as tf


def get_encoder(input_shape=(28, 28, 1)):
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.tanh, input_shape=input_shape),
        keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.tanh),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),

        keras.layers.Conv2D(64, 3, padding='same', activation=tf.nn.tanh),
        keras.layers.Conv2D(64, 3, padding='same', activation=tf.nn.tanh),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),

        keras.layers.Conv2D(32, 3, padding='valid', activation=tf.nn.tanh),
        keras.layers.Conv2D(32, 3, padding='valid', activation=tf.nn.tanh),
        keras.layers.Flatten()
    ], name='encoder')
    return model


def get_decoder(input_shape=(288,)):
    model = keras.Sequential([
        keras.layers.Reshape((3, 3, 32), input_shape=input_shape),
        keras.layers.UpSampling2D(),
        keras.layers.Conv2DTranspose(32, 3, padding='same', activation=tf.nn.tanh),
        keras.layers.Conv2DTranspose(32, 3, padding='same', activation=tf.nn.tanh),
        keras.layers.BatchNormalization(),

        keras.layers.UpSampling2D(),
        keras.layers.Conv2DTranspose(64, 3, padding='same', activation=tf.nn.tanh),
        keras.layers.Conv2DTranspose(64, 3, padding='same', activation=tf.nn.tanh),
        keras.layers.BatchNormalization(),

        keras.layers.UpSampling2D(),
        keras.layers.Conv2DTranspose(64, 3, padding='valid', activation=tf.nn.tanh),
        keras.layers.Conv2DTranspose(64, 3, padding='valid', activation=tf.nn.tanh),
        keras.layers.BatchNormalization(),

        keras.layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid')
    ], name='decoder')
    return model


def get_disc_latent(input_shape=(288,)):
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),

        keras.layers.Dense(64),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),

        keras.layers.Dense(32),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),

        keras.layers.Dense(16),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),

        keras.layers.Dense(1)
    ], name='discriminator_latent')
    return model


def get_disc_visual(input_shape=(28, 28, 1)):
    model = keras.Sequential([
        keras.layers.Conv2D(16, 5, (2, 2), padding='same', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Conv2D(16, 5, (2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Conv2D(16, 5, (2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Conv2D(1, 5, (2, 2), padding='same'),
        keras.layers.GlobalAveragePooling2D()
    ], name='discriminator_visual')
    return model


def get_classifier(input_shape=(28, 28, 1)):
    model = keras.Sequential([
        keras.layers.Conv2D(32, 5, (2, 2), padding='same', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Conv2D(64, 5, (2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Conv2D(64, 5, (2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),

        keras.layers.Conv2D(1, 5, (2, 2), padding='same'),
        keras.layers.GlobalAveragePooling2D()
    ], name='classifier')
    return model

