import os
import h5py
import numpy as np
import tensorflow as tf


def load_data(foldername, num_points):
    def load_h5(h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        return (data, label)
    # load points and labels
    path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(path, foldername)
    filenames = [d for d in os.listdir(data_path)]
    print(data_path)
    print(filenames)
    points = None
    labels = None
    for d in filenames:
        cur_points, cur_labels = load_h5(os.path.join(data_path, d))
        cur_points = cur_points.reshape(1, -1, 3)
        cur_labels = cur_labels.reshape(1, -1)
        if labels is None or points is None:
            labels = cur_labels
            points = cur_points
        else:
            labels = np.hstack((labels, cur_labels))
            points = np.hstack((points, cur_points))
    points_r = points.reshape(-1, num_points, 3)
    labels_r = labels.reshape(-1, 1)

    return points_r, labels_r


class mat_mul(tf.keras.layers.Layer):
    def call(self, A, B):
        return tf.matmul(A, B)

def sim_func_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, C, 1)
    # v shape: (N, 1, 1)
    v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
    return v


def sim_func_dim2(x, y):
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v

def get_negative_mask(batch_size):
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)