import h5py
import numpy as np
from .datasets import ConditionalDataset


def load_data():
    # Load the image and label data from an HDF5 file
    with h5py.File('data.hdf5', 'r') as f:
        x_train = np.array(f['images'])
        y_train = np.array(f['labels'])

    # Preprocess the dataset
    x_train = (x_train[:, :, :, np.newaxis] / 255.0).astype('float32')
    y_train = y_train.astype('float32')

    # Create a ConditionalDataset object and return it
    datasets = ConditionalDataset()
    datasets.images = x_train
    datasets.attrs = y_train
    datasets.attr_names = ['valence', 'arousal']

    return datasets