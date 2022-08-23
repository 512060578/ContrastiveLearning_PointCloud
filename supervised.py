import os
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import numpy as np
from tqdm import tqdm 

# number of points in each sample
num_points = 2048

# number of categories
k = 40

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

# load train points and labels
path = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(path, "Data_Train")
filenames = [d for d in os.listdir(train_path)]
print(train_path)
print(filenames)
train_points = None
train_labels = None
for d in filenames:
    cur_points, cur_labels = load_h5(os.path.join(train_path, d))
    cur_points = cur_points.reshape(1, -1, 3)
    cur_labels = cur_labels.reshape(1, -1)
    if train_labels is None or train_points is None:
        train_labels = cur_labels
        train_points = cur_points
    else:
        train_labels = np.hstack((train_labels, cur_labels))
        train_points = np.hstack((train_points, cur_points))
train_points_r = train_points.reshape(-1, num_points, 3)
train_labels_r = train_labels.reshape(-1, 1)

# load test points and labels
test_path = os.path.join(path, "Data_Test")
filenames = [d for d in os.listdir(test_path)]
print(test_path)
print(filenames)
test_points = None
test_labels = None
for d in filenames:
    cur_points, cur_labels = load_h5(os.path.join(test_path, d))
    cur_points = cur_points.reshape(1, -1, 3)
    cur_labels = cur_labels.reshape(1, -1)
    if test_labels is None or test_points is None:
        test_labels = cur_labels
        test_points = cur_points
    else:
        test_labels = np.hstack((test_labels, cur_labels))
        test_points = np.hstack((test_points, cur_points))
test_points_r = test_points.reshape(-1, num_points, 3)
test_labels_r = test_labels.reshape(-1, 1)

Y_train = tf.keras.utils.to_categorical(train_labels_r, k)
Y_test = tf.keras.utils.to_categorical(test_labels_r, k)

class mat_mul(tf.keras.layers.Layer):
    def call(self, A, B):
        return tf.matmul(A, B)

input_points = tf.keras.Input(shape=(num_points, 3))

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

g = g = mat_mul()(input_points,input_T)
g = tf.keras.layers.Conv1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
g = tf.keras.layers.BatchNormalization()(g)
g = tf.keras.layers.Conv1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
g = tf.keras.layers.BatchNormalization()(g)

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

g = mat_mul()(g,feature_T)
g = tf.keras.layers.Conv1D(64, 1, activation='relu')(g)
g = tf.keras.layers.BatchNormalization()(g)
g = tf.keras.layers.Conv1D(128, 1, activation='relu')(g)
g = tf.keras.layers.BatchNormalization()(g)
g = tf.keras.layers.Conv1D(1024, 1, activation='relu')(g)
g = tf.keras.layers.BatchNormalization()(g)

h = tf.keras.layers.GlobalAveragePooling1D()(g)

projection_1 = tf.keras.layers.Dense(256)(h)
projection_1 = tf.keras.layers.Activation("relu")(projection_1)
projection_2 = tf.keras.layers.Dense(128)(projection_1)
projection_2 = tf.keras.layers.Activation("relu")(projection_2)
projection_3 = tf.keras.layers.Dense(k, activation='softmax')(projection_2)
prediction = tf.keras.layers.Flatten()(projection_3)

model = tf.keras.Model(inputs=input_points, outputs=prediction)

print(model.summary())

adam = tf.keras.optimizers.Adam(lr=0.001, decay=0.7)

# compile classification model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

for i in range(1,50):
    model.fit(train_points_r, Y_train, batch_size=32, epochs=1, shuffle=True, verbose=1)
    s = "Current epoch is:" + str(i)
    print(s)
    if i % 5 == 0:
        score = model.evaluate(test_points_r, Y_test, verbose=1)
        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])

# score the model
score = model.evaluate(test_points_r, Y_test, verbose=1)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])