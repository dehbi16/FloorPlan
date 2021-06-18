import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import utils
import os
from tensorflow import keras
from keras import layers
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop


def write2pickle(train_dir):
    train_data_path = [os.path.join(train_dir, path) for path in os.listdir(train_dir)]
    print(f'Number of dataset: {len(train_data_path)}')
    for path in train_data_path:
        print(path)
        with Image.open(path) as temp:
            image_array = np.asarray(temp, dtype=np.uint8)
        boundary_mask = image_array[:, :, 0]
        category_mask = image_array[:, :, 1]
        index_mask = image_array[:, :, 2]
        inside_mask = image_array[:, :, 3]
        shape_array = image_array.shape
        index_category = []
        room_node = []

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


def plot_image(image):
    ax = plt.subplot(1, 5, 1)
    plt.imshow(image, cmap="Greys")
    plt.axis("off")
    for i in range(image.shape[-1]):
        ax = plt.subplot(1, 5, i+2)
        plt.imshow(image[:, :, i], cmap="Greys")
        plt.axis("off")
    plt.show()


def load_data(train_dir):
    train_data_path = [os.path.join(train_dir, path) for path in os.listdir(train_dir)]
    print(f'Number of dataset: {len(train_data_path)}')
    data = []
    for path in train_data_path:
        # print(path)
        with Image.open(path) as temp:
            image_array = np.asarray(temp, dtype=np.uint8)
        data.append(image_array)
    return np.array(data)


def define_discriminator(in_shape=(256, 256, 4)):
    model = keras.Sequential()
    model.add(layers.Input(shape=in_shape))
    # model.add(CenterCrop(height=256, width=256))
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation="sigmoid"))

    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def define_generator(latent_dim):
    model = keras.Sequential()
    n_node = 128 * 8 * 8
    model.add(layers.Dense(n_node, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((8, 8, 128)))
    model.add(layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(4, (7, 7), activation="tanh", padding="same"))
    return model


def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    return model


def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    x = dataset[ix]
    y = np.ones((n_samples, 1))
    return x, y


def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim*n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    x = generator.predict(x_input)
    y = np.zeros((n_samples, 1))
    return x, y


def train(g_model, d_model, gan_model, dataset, latent_dim, epochs=100, n_batch=128):
    print("start")
    bat_per_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch/2)
    print("start")
    for i in range(epochs):
        for j in range(bat_per_epoch):
            x_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(x_real, y_real)

            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch(x_fake, y_fake)

            x_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(x_gan, y_gan)
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epoch, d_loss1, d_loss2, g_loss))
    g_model.save('generator.h5')
"""
image_size = (256, 256)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"floorplan_dataset", label_mode=None
)
"""
"""plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()


dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset", label_mode=None
    )
"""

if __name__ == "__main__":
    print("*******************************************")
    latent_dim = 100
    # create the discriminator
    discriminator = define_discriminator()
    print(discriminator.summary())
    # create the generator
    # print("*******************************************")
    generator = define_generator(latent_dim)
    # print(generator.summary())
    # create the gan
    gan_model = define_gan(generator, discriminator)

    # load image data
    # dataset = load_data("dataset/floorplan")

    print("*******************************************")
    # train model
    # train(generator, discriminator, gan_model, dataset, latent_dim, epochs=500)
