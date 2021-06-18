import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import utils
import os
from tensorflow import keras
from keras import layers


def write2pickle(image_array):
    boundary_mask = image_array[:, :, 0]
    category_mask = image_array[:, :, 1]
    index_mask = image_array[:, :, 2]
    inside_mask = image_array[:, :, 3]
    shape_array = image_array.shape
    index_category = []
    room_node = []
    # print(index_mask[60,:])
    interiorWall_mask = np.zeros(category_mask.shape, dtype=np.uint8)
    interiorWall_mask[category_mask == 16] = 1
    interiordoor_mask = np.zeros(category_mask.shape, dtype=np.uint8)
    interiordoor_mask[category_mask == 17] = 1

    for h in range(shape_array[0]):
        for w in range(shape_array[1]):
            index = index_mask[h, w]
            category = category_mask[h, w]
            if index > 0 and category <= 12:
                if len(index_category):
                    flag = True
                    for i in index_category:
                        if i[0] == index:
                            flag = False
                    if flag:
                        index_category.append((index, category))
                else:
                    index_category.append((index, category))

    for (index, category) in index_category:
        node = {}
        node['category'] = int(category)
        mask = np.zeros(index_mask.shape, dtype=np.uint8)
        mask[index_mask == index] = 1
        node['centroid'] = utils.compute_centroid(mask)
        room_node.append(node)
    return room_node


def load_data(train_dir):
    train_data_path = [os.path.join(train_dir, path) for path in os.listdir(train_dir)]
    print(f'Number of dataset: {len(train_data_path)}')
    data = []
    labels = []
    for path in train_data_path:
        # print(path)
        with Image.open(path) as temp:
            image_array = np.asarray(temp, dtype=np.uint8)
        data.append(image_array)
        category_mask = image_array[:, :, 1]
        y = np.unique(category_mask)
        label = np.zeros(13)
        for elem in y:
            if elem <= 12:
                label[elem] = 1
        labels.append(label)
    return [np.array(data), np.array(labels)]


def define_discriminator(in_shape=(256, 256, 4), nb_classes=13):
    in_label = layers.Input(shape=(nb_classes,))
    n_nodes = in_shape[0] * in_shape[1] * in_shape[2]
    li = layers.Dense(n_nodes)(in_label)
    li = layers.Reshape((in_shape[0], in_shape[1], in_shape[2]))(li)
    in_image = layers.Input(shape=in_shape)
    merge = layers.Concatenate()([in_image, li])

    fe = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(merge)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.Flatten()(fe)
    fe = layers.Dropout(0.4)(fe)

    output = layers.Dense(1, activation="sigmoid")(fe)

    model = keras.Model([in_image, in_label], output)
    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def define_generator(latent_dim, nb_classes=13):
    in_label = layers.Input(shape=(nb_classes, ))
    n_nodes = 8 * 8 * 4
    li = layers.Dense(n_nodes)(in_label)
    li = layers.Reshape((8, 8, 4))(li)

    in_lat = layers.Input(shape=(latent_dim, ))
    n_nodes = 128*8*8
    gen = layers.Dense(n_nodes)(in_lat)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Reshape((8,8,128))(gen)

    merge = layers.Concatenate()([gen, li])
    gen = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(merge)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    output = layers.Conv2D(4, (7, 7), activation="tanh", padding="same")(gen)
    model = keras.Model([in_lat, in_label], output)
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


def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = np.random.randint(0, images.shape[0], n_samples)
    x, label = images[ix], labels[ix]
    y = np.ones((n_samples, 1))
    return [x, label], y


def generate_latent_points(latent_dim, n_samples, nb_classes=13):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    labels = np.random.randint(0, nb_classes, size=(n_samples, nb_classes))
    return [x_input, labels]


def generate_fake_samples(generator, latent_dim, n_samples):
    x_input, labels = generate_latent_points(latent_dim, n_samples)
    x = generator.predict([x_input, labels])
    y = np.zeros((n_samples, 1))
    return [x, labels], y


def train(g_model, d_model, gan_model, dataset, latent_dim, epochs=100, n_batch=128):
    bat_per_epoch = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch/2)

    for i in range(epochs):
        for j in range(bat_per_epoch):
            [x_real, label_real], y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch([x_real, label_real], y_real)

            [x_fake, label_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch([x_fake, label_fake], y_fake)

            [x_gan, labels_input] = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch([x_gan, labels_input], y_gan)
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epoch, d_loss1, d_loss2, g_loss))
            g_model.save('cgan_generator.h5')


if __name__ == "__main__":
    latent_dim = 100
    # create the discriminator
    discriminator = define_discriminator()
    # create the generator
    generator = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(generator, discriminator)
    # load image data
    dataset = load_data('dataset/floorplan')
    # train model
    train(generator, discriminator, gan_model, dataset, latent_dim, epochs=500)
