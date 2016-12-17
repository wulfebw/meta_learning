import numpy as np
from sklearn.datasets import fetch_mldata

# data utils
def load_mnist(train_split=.8, normalize=True, shuffle=True, debug_size=None):
    # load
    mnist = fetch_mldata('MNIST original')

    # convert targets to one-hot
    targets = np.zeros((len(mnist['target']), 10))
    hot_idxs = mnist['target'].astype(int)
    targets[np.arange(len(targets)), hot_idxs] = 1
    mnist['target'] = targets

    # shuffle, do this before subsetting the data because it is ordered
    if shuffle:
        idxs = np.random.permutation(len(mnist['data']))
        mnist['data'] = mnist['data'][idxs]
        mnist['target'] = mnist['target'][idxs]

    # if debugging, use fewer samples
    if debug_size is not None:
        mnist['data'] = mnist['data'][:debug_size]
        mnist['target'] = mnist['target'][:debug_size]

    # load into format
    num_train = int(len(mnist['data']) * train_split)
    data = {}
    data['x_train'] = mnist['data'][:num_train]
    data['y_train'] = mnist['target'][:num_train]
    data['x_val'] = mnist['data'][num_train:]
    data['y_val'] = mnist['target'][num_train:]

    # normalize
    if normalize:
        # compute stats for the training set only
        mean = np.mean(data['x_train'])
        data['x_train'] = (data['x_train'] - mean) / 255.
        data['x_val'] = (data['x_val'] - mean) / 255.

    return data