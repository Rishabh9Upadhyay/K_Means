# One imp app of clustring is in color compression with images.
# Imagine u have an image with millions of colors in it.
# In most images, a large number of colors will be unused,
# and many of the pics int the image wil, have similar or even identical colors.
# Having to many color in image makes it very hard for image processing an
# image analysis----->This is one area where k-means applied
# It's applied in image segmentation, image analysis, image compression ans so on.

# Now will use an inbuilt image defined in pillow package
from sklearn.datasets import load_sample_image
from matplotlib import pyplot as plt
import numpy as np
china = load_sample_image('china.jpg')

ax = plt.axes(xticks=[],yticks=[])
ax.imshow(china)

print(china.shape)

data = china/255.0      #use 0......1 scale
data = data.reshape(427*640,3)
print(data.shape)


def plot_pixel(data, title, colors=None, N=10000):
    if colors is None:
        colors=data

    # chose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B =data[i].T

    fig, ax = plt.subplots(1,2, figsize=(16,6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0,1), ylim=(0,1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0,1), ylim=(0,1))

    fig.suptitle(title, size=20)

plot_pixel(data, title='Input color space: 16 millions possible colors')
plt.show()

import warnings
warnings.simplefilter('ignore')         #fix numpy issues

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(16)
# Because we r dealing with very large dataset we will use minibatchkmeans.
# This operates on subsets of the data to compute the result more quickly
# and more accuratily.

kmeans.fit(data)
new_color = kmeans.cluster_centers_[kmeans.predict(data)]
plot_pixel(data, colors=new_color, title='Reduced color space to 16 colors')
plt.show()

china_recolored = new_color.reshape(china.shape)
fig, ax = plt.subplots(1, 2, figsize=(16,6),subplot_kw=dict(xticks=[],yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Orginal image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-colored image',size=16)
plt.show()

print(data)