from keras.models import load_model
from numpy import load
from numpy import vstack
import matplotlib.pyplot as plt
from numpy.random import randint


# load and prepare training images
def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    for i in range(len(X1)):
        X1[i] = (X1[i] - 127.5) / 127.5
        X2[i] = (X2[i] - 127.5) / 127.5
    return [X1, X2]


# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
    images = vstack((src_img, gen_img, tar_img))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Source', 'Generated', 'Expected']
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        plt.subplot(1, 3, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[i], cmap="Greys")
        # show title
        plt.title(titles[i])
    plt.show()


[X1, X2] = load_real_samples('floor_plan.npz')
print('Loaded', X1.shape, X2.shape)
model = load_model('model_000200.h5')
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
gen_image = model.predict(src_image)
print(tar_image[0, 10, :])
print(gen_image[0, 10, :])
plot_images(src_image, gen_image, tar_image)