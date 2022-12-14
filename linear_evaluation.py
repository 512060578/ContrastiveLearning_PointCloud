from sklearn.manifold import TSNE
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import seaborn as sns
import numpy as np
import h5py
from utils import *


# name of the trained model
model_name = 'model_name'


# number of points in each sample
num_points = 2048

# number of categories
k = 40

# def batch size
BATCH_SIZE = 128

# Load Training Data
train_points_r, train_labels_r = load_data("Data_Train", num_points)

# load Testing Data
test_points_r, test_labels_r = load_data("Data_Test", num_points)


def get_pointnet_simclr(hidden_1, hidden_2, hidden_3):
    input_points = tf.keras.Input(shape=(num_points, 3))

    # input_Transformation_net
    x = tf.keras.layers.Conv1D(64, 1, activation='relu',
                  input_shape=(num_points, 3))(input_points)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 1, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(1024, 1, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=num_points)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = tf.keras.layers.Reshape((3, 3))(x)

    # forward net
    g = g = mat_mul()(input_points,input_T)
    g = tf.keras.layers.Conv1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = tf.keras.layers.BatchNormalization()(g)
    g = tf.keras.layers.Conv1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = tf.keras.layers.BatchNormalization()(g)

    # feature transform net
    f = tf.keras.layers.Conv1D(64, 1, activation='relu')(g)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Conv1D(128, 1, activation='relu')(f)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Conv1D(1024, 1, activation='relu')(f)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.MaxPool1D(pool_size=num_points)(f)
    f = tf.keras.layers.Dense(512, activation='relu')(f)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dense(256, activation='relu')(f)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)

    feature_T = tf.keras.layers.Reshape((64, 64))(f)

    # forward net
    g = mat_mul()(g,feature_T)
    g = tf.keras.layers.Conv1D(64, 1, activation='relu')(g)
    g = tf.keras.layers.BatchNormalization()(g)
    g = tf.keras.layers.Conv1D(128, 1, activation='relu')(g)
    g = tf.keras.layers.BatchNormalization()(g)
    g = tf.keras.layers.Conv1D(1024, 1, activation='relu')(g)
    g = tf.keras.layers.BatchNormalization()(g)

    # global_feature
    h = tf.keras.layers.GlobalAveragePooling1D()(g)

    # projection head
    projection_1 = tf.keras.layers.Dense(hidden_1)(h)
    projection_1 = tf.keras.layers.Activation("relu")(projection_1)
    projection_2 = tf.keras.layers.Dense(hidden_2)(projection_1)
    projection_2 = tf.keras.layers.Activation("relu")(projection_2)
    projection_3 = tf.keras.layers.Dense(hidden_3)(projection_2)

    pointnet_simclr = tf.keras.Model(input_points, projection_3)

    return pointnet_simclr

pointnet_simclr = get_pointnet_simclr(256, 128, 50)
pointnet_simclr.load_weights(model_name)
pointnet_simclr.summary()

def get_linear_model(features):
    linear_model = tf.keras.Sequential([tf.keras.layers.Dense(64, input_shape=(features, ), activation='relu'),
                                        tf.keras.layers.Dense(40, activation="softmax")
                                                    ])
    return linear_model


# Encoder model with projections
projection = tf.keras.Model(pointnet_simclr.input, pointnet_simclr.layers[-4].output)

# Extract train and test features
train_features = projection.predict(train_points_r)
test_features = projection.predict(test_points_r)

print(train_features.shape, test_features.shape)

# Early Stopping to prevent overfitting
es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=2, restore_best_weights=True)

linear_model = get_linear_model(256)
linear_model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"],
                     optimizer="adam")
history = linear_model.fit(train_features, train_labels_r,
                 validation_data=(test_features, test_labels_r),
                 batch_size=64,
                 epochs=1000,
                 callbacks=[es])

# Plot the training history
def plot_training(H):
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("history.jpg")

plot_training(history)

# Visualization of the representations
def plot_vecs_n_labels(v, labels):
    fig = plt.figure(figsize = (10, 10))
    sns.set_style("darkgrid")
    sns.scatterplot(v[:,0], v[:,1], hue=labels, legend='full', palette=sns.color_palette("bright", k))
    plt.savefig("TSNE.jpg")

    return fig

tsne = TSNE()
low_vectors = tsne.fit_transform(train_features)
fig = plot_vecs_n_labels(low_vectors, train_labels_r.reshape(-1))