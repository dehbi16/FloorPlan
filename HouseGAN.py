import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


def define_generator(latent_dim=128, nb_classes=10):
    in_shape = (8, 8, 16)
    n_nodes = in_shape[0] * in_shape[1] * in_shape[2]

    in_label = layers.Input(shape=(nb_classes,))
    in_lat = layers.Input(shape=(latent_dim,))

    merge = layers.Concatenate()([in_lat, in_label])

    x = layers.Dense(n_nodes)(merge)
    x = layers.Reshape((in_shape[0], in_shape[1], in_shape[2]))(x)

    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(256, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(128, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    output = layers.Conv2D(1, (7, 7), activation="tanh", padding="same")(x)

    model = keras.Model([in_lat, in_label], output)
    return model


def define_discriminator(in_shape = (32, 32, 8), nb_classes=10):
    n_nodes = in_shape[0] * in_shape[1] * in_shape[2]

    in_label = layers.Input(shape=(nb_classes,))
    li = layers.Dense(n_nodes)(in_label)
    li = layers.Reshape((in_shape[0], in_shape[1], in_shape[2]))(li)

    in_image = layers.Input(shape=(32, 32, 1))

    merge = layers.Concatenate()([in_image, li])

    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(merge)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(merge)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(merge)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(16, (3, 3), strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)

    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model([in_image, in_label], output)
    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def define_gan(g_model, d_model):
    d_model.trainable = False
    gen_noise, gen_label = g_model.input
    gen_output = g_model.output

    gan_output = d_model([gen_output, gen_label])

    model = keras.Model([gen_noise, gen_label], gan_output)
    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    return model


nb_classes = 10
z = np.random.normal(size=128)
t = np.zeros(nb_classes)
t[np.random.randint(0, nb_classes, 1)] = 1
input = np.concatenate((z, t), axis=0)

in_shape = (32, 32, 8)
model = define_discriminator()
print(model.summary())
# print(input)

