import random
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from keras.utils import np_utils
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

data = np.load('./dataset_v2/data.npy', encoding='bytes') 
labels = np.load('./dataset_v2/labels.npy', encoding='bytes') 
class_idx = np.load('./dataset_v2/label_classes.npy', encoding='bytes')

print(data.shape)
print(labels.shape)

number_classes = len(class_idx)
print('number of classes', number_classes)

x_train, x_test, y_train, y_test = train_test_split(
    data, labels,
    test_size=0.3,
    random_state=42,
    shuffle=True,
    stratify=labels
    )

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

x_train /= 255
x_test /= 255

(img_shape_w, img_shape_h) = x_train[0].shape
x_train = x_train.reshape(x_train.shape[0], img_shape_w, img_shape_h, 1)
x_test = x_test.reshape(x_test.shape[0], img_shape_w, img_shape_h, 1)

print('x_train', x_train.shape)
print('y_train', y_train.shape)
print('x_test', x_test.shape)
print('y_test', y_test.shape)

ly1_neurons = 416
ly2_neurons = 272
ly3_neurons = 240

kernel_size_1 = (3, 3)
kernel_size_2 = (5, 5)
kernel_size_3 = (5, 5)

dropout_1 = 0.2
dropout_2 = 0.1
dropout_3 = 0.0

pool_size_1 = (2, 2)
pool_size_2 = (2, 2)
pool_size_3 = (2, 2)

categories=number_classes

model = Sequential()

model.add(Conv2D(ly1_neurons, kernel_size_1, input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=pool_size_1))
model.add(Dropout(dropout_1))

model.add(Conv2D(ly2_neurons, kernel_size_2))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=pool_size_2))
model.add(Dropout(dropout_2))

model.add(Conv2D(ly3_neurons, kernel_size_3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=pool_size_3))
model.add(Dropout(dropout_3))

model.add(Flatten())
model.add(Dense( 512 ))
model.add(Activation('relu'))
model.add(Dense( categories ))
model.add(Activation('softmax'))

learning_rate = 0.0001
model.compile(optimizer=optimizers.Adam(
    learning_rate=learning_rate
), loss='categorical_crossentropy', metrics=['accuracy'])

gen = ImageDataGenerator(rotation_range=12, width_shift_range=0.1, shear_range=0.3,
                        height_shift_range=0.1, zoom_range=0.1, data_format='channels_first')

batch_size = 512
batches = gen.flow(x_train, y_train, batch_size=batch_size)
test_batches = gen.flow(x_test, y_test, batch_size=batch_size)
steps_per_epoch = int(np.ceil(batches.n/batch_size))
validation_steps = int(np.ceil(test_batches.n/batch_size))

history = model.fit(
    x_train,
    y_train,
    batch_size=120,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=1
    )