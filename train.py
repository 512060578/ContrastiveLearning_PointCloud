import os
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import numpy as np
from tqdm import tqdm
from utils import *

# number of points in each sample
num_points = 2048

# number of categories
k = 40

# def batch size
BATCH_SIZE = 128


# Load Data
train_points_r, _ = load_data("Data_Train", num_points)

train_ds = tf.data.Dataset.from_tensor_slices(train_points_r)

train_ds = (
    train_ds
    .shuffle(1024)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)


# The augmentation pipeline
class CustomAugment(object):
    def __call__(self, sample):        
        # sample = self._random_apply(self.flip_point_cloud, sample, p=0.1)

        sample = self._random_apply(self.jitter_point_cloud, sample, p=1)
        
        sample = self._random_apply(self.rotate_point_cloud, sample, p=1)

        # sample = self._random_apply(self.transition_point_cloud, sample, p=1)

        # sample = self._random_apply(self.drop_point_cloud, sample, p=0.1)

        return sample

    def jitter_point_cloud(self, batch_data, sigma=0.05, clip=0.05):
    # Applies a jittering toward each points
      B, N, C = batch_data.shape
      assert(clip > 0)
      jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
      jittered_data += batch_data
      return jittered_data

    def rotate_point_cloud(self, batch_data):
    #Rotate the object with a randomly generated angle
      rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
      for k in range(batch_data.shape[0]):
          rotation_angle = np.random.uniform() * 2 * np.pi
          cosval = np.cos(rotation_angle)
          sinval = np.sin(rotation_angle)
          rotation_matrix = np.array([[cosval, 0, sinval],
                                      [0, 1, 0],
                                      [-sinval, 0, cosval]])
          shape_pc = batch_data[k, ...]
          rotated_data[k, ...] = np.dot(tf.reshape(shape_pc,(-1, 3)), rotation_matrix)
      return rotated_data


    def transition_point_cloud(self, batch_data, clip=0.1):
    # Apply a transition for the whole object
        B, N, C = batch_data.shape
        assert(clip > 0)
        transition = np.clip(np.random.randn(1,1,C), -1 * clip, clip)
        transition_data = batch_data + transition

        return transition_data
    

    def flip_point_cloud(self, batch_data):
    # Flip the whole object in the batch
        flipped_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            shape_pc = batch_data[k, ...]
            flipped_data[k, ...] = -1 * shape_pc

        return flipped_data

    def drop_point_cloud(batch_data, rate=0.93):
    # The rate of points in the object is remained and others are dropped
        dropped_data = np.zeros(batch_data.shape, dtype=np.float32)
        for b in range(batch_data.shape[0]):
            drop_idx = np.where(np.random.random((batch_data.shape[1]))<=rate)[0]
            if len(drop_idx)>0:
                dropped_data[b,drop_idx,:] = batch_data[b,drop_idx,:] # set to the first point
        return dropped_data
    
    def _random_apply(self, func, x, p):
        return tf.cond(
          tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                  tf.cast(p, tf.float32)),
          lambda: func(x),
          lambda: x)

# Build the augmentation pipeline
data_augmentation = tf.keras.Sequential([tf.keras.layers.Lambda(CustomAugment())])

negative_mask = get_negative_mask(BATCH_SIZE)
            

@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature):
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        l_pos = sim_func_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (BATCH_SIZE, 1))
        l_pos /= temperature

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(BATCH_SIZE, dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (BATCH_SIZE, -1))
            l_neg /= temperature

            logits = tf.concat([l_pos, l_neg], axis=1) 
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * BATCH_SIZE)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

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

    print(pointnet_simclr.summary())

    return pointnet_simclr

def train_pointnet(model, dataset, optimizer, criterion,
                 temperature=0.1, epochs=100):
    step_wise_loss = []
    epoch_wise_loss = []

    for epoch in tqdm(range(epochs)):
        for image_batch in dataset:

            a = data_augmentation(image_batch)
            b = data_augmentation(image_batch)

            loss = train_step(a, b, model, optimizer, criterion, temperature)
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))

        if epoch % 10 == 0:
            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

    return epoch_wise_loss, model

criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                          reduction=tf.keras.losses.Reduction.SUM)
decay_steps = 1000
lr_decayed_fn = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.1, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)

pointnet_simclr_2 = get_pointnet_simclr(256, 128, 50)

epoch_wise_loss, pointnet_simclr  = train_pointnet(pointnet_simclr_2, train_ds, optimizer, criterion,
                 temperature=0.1, epochs=200)


# Plot the training loss and save the weights

import datetime

figurename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "losses.jpg"
print(figurename)

plt.plot(epoch_wise_loss)
plt.title("Training Loss")
plt.savefig(figurename)
plt.show()

filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "pointnet_simclr.h5"


pointnet_simclr.save_weights(filename)
print(filename)