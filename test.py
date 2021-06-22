# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import utils


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


def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def plot_image(data):
    for k in range(data.shape[0]):
        image = data[k]
        ax = plt.subplot(data.shape[0], 5, k*5+1)
        plt.imshow(image, cmap="Greys")
        plt.axis("off")
        for i in range(image.shape[-1]):
            ax = plt.subplot(data.shape[0], 5, k*5+i+2)
            plt.imshow(image[:, :, i], cmap="Greys")
            plt.axis("off")
    plt.show()

"""
# load model
model = load_model('generator.h5')
# generate images
latent_points = generate_latent_points(100, 5)
# generate images
X = model.predict(latent_points)
print(X.shape)
# plot the result
plot_image(X)

"""

# dataset = load_data("dataset/floorplan")
print("********************************")
# plot_image(dataset[:5])
path = "dataset/floorplan/0.png"
with Image.open(path) as temp:
    image_array = np.asarray(temp, dtype=np.uint8)
room_node = write2pickle(image_array)
for i in range(len(room_node)):
    print(room_node[i])

print(image_array[50, :, 1])

plt.subplot(1, 5, 1)
plt.imshow(image_array, cmap="Greys")
plt.axis("off")
for i in range(image_array.shape[-1]):
    ax = plt.subplot(1, 5, i+2)
    plt.imshow(image_array[:, :, i], cmap="Greys")
    plt.axis("off")
plt.show()

