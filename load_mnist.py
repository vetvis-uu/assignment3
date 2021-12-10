"""
.. module:: load_mnist
   :platform: Linux, Windows
   :synopsis: I/O utils for loading MNIST-like word images and label data

.. moduleauthor:: Fredrik Nysjo (after original Matlab code by Anders Hast)
"""

import logging
logging.getLogger().setLevel(logging.INFO)

import numpy as np


def load_mnist_words(filename):
    """Load stack of MNIST-like word images and return in NumPy format"""
    images = None
    with open(filename, "rb") as stream:
        # Read header that is stored in big-endian format
        dt_int32 = np.dtype(np.int32).newbyteorder(">")
        magic = np.frombuffer(stream.read(4), dtype=dt_int32)[0]
        n_images = np.frombuffer(stream.read(4), dtype=dt_int32)[0]
        n_rows = np.frombuffer(stream.read(4), dtype=dt_int32)[0]
        n_cols = np.frombuffer(stream.read(4), dtype=dt_int32)[0]

        logging.debug((magic, n_images, n_rows, n_cols))
        if not (magic in (0x3840, 0x5eec, 0x7080)):
            logging.error("Not an image stack")
            return images

        # Read image stack
        n_bytes = n_rows * n_cols * n_images
        images = np.frombuffer(stream.read(n_bytes), dtype=np.uint8)
        images = images.astype(dtype=np.uint8)  # Make writeable copy
        images = images.reshape(n_images, n_cols, n_rows)
    return images


def load_mnist_labels(filename):
    """Load MNIST-like label data and return in NumPy format"""
    labels = None
    with open(filename, "rb") as stream:
        # Read header that is stored in big-endian format
        dt_int32 = np.dtype(np.int32).newbyteorder(">")
        magic = np.frombuffer(stream.read(4), dtype=dt_int32)[0]
        n_labels = np.frombuffer(stream.read(4), dtype=dt_int32)[0]

        logging.debug((magic, n_labels))
        if magic != 0x801:
            logging.error("Not a label file")
            return labels

        # Read label data
        n_bytes = n_labels
        labels = np.frombuffer(stream.read(n_bytes), dtype=np.uint8)
        labels = labels.astype(dtype=np.uint8)  # Make writeable copy
    return labels
