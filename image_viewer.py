"""
.. module:: image_viewer
   :platform: Linux, Windows
   :synopsis: Example code for viewing the dataset in Assignment 3

.. moduleauthor:: Fredrik Nysjo
"""

from load_mnist import load_mnist_words, load_mnist_labels

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


# Load the image stack
images = load_mnist_words("Esposalles-bgr-images")
assert images is not None, "Could not load image stack"
n_images = images.shape[0]

# Load the label data
labels = load_mnist_labels("Esposalles-bgr-labels")
assert labels is not None, "Could not load labels"
n_labels = labels.shape[0]
assert n_images == n_labels

# Generate subsampled smaller version of the dataset
n_skip = 20
images_small = images[::n_skip]
labels_small = labels[::n_skip]

# Create lookup table with randomized colors for the unique labels. Feel free to
# change this if you prefer other colors!
np.random.seed(2345)
n_unique_labels = len(np.unique(labels))
color_lut = np.random.rand(n_unique_labels, 3) * 0.8 + 0.1

# Create a figure for inspecting the image and label data
plt.figure()
image_plot = plt.imshow(np.transpose(images[0,:,:]), cmap="gray", vmin=0, vmax=255)
label_text = plt.title("Label: " + str(int(labels[0])))

# Add a slider for scrolling through the dataset
slider_rect = plt.axes([0.20, 0.1, 0.60, 0.03])
slider = widgets.Slider(slider_rect, "Image", 0, n_images-1, valinit=0, valfmt="%d")
def slider_callback(val):
    label_text.set_text("Label: " + str(labels[int(val)]))
    image_plot.set_data(np.transpose(images[int(val),:,:]))
slider.on_changed(slider_callback)

plt.show()