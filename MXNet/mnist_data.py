"""
MNIST data loader

@author socurites@aidentify.io
"""
import numpy as np
import gzip
import struct


def read_data(label_path, image_path):
    with gzip.open(label_path) as flbl:
        _, _ = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image_path, 'rb') as fimg:
        _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)