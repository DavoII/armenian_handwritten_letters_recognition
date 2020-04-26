import random
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from kerastuner import RandomSearch
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from kerastuner.engine.hyperparameters import HyperParameters
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

x_train = x_train.reshape(x_train.shape[0], 64, 64, 1)
x_test = x_test.reshape(x_test.shape[0], 64, 64, 1)

print('x_train', x_train.shape)
print('y_train', y_train.shape)
print('x_test', x_test.shape)
print('y_test', y_test.shape)

ly1_neurons = 128
ly2_neurons = 256
ly3_neurons = 512

categories=number_classes

model = Sequential()

model.add(Conv2D(ly1_neurons, (3,3), input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(ly2_neurons, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(ly3_neurons, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense( 1024 ))
model.add(Activation('relu'))
model.add(Dense( 512 ))
model.add(Activation('relu'))
model.add(Dense( categories ))
model.add(Activation('softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

model.summary()

history = model.fit(
    x_train,
    y_train,
    batch_size=220,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=1
    )

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

test_predictions = model.predict_classes(x_test)

y_test_asCategories = np.argmax(y_test, axis=1)

confusion = confusion_matrix(y_test_asCategories, test_predictions)

df_cm = pd.DataFrame(confusion,
                     index = [i for i in class_idx],
                     columns = [i for i in class_idx])
plt.figure(figsize = (40,40))
sn.heatmap(df_cm, annot=True)


image_index = random.randint(1,len(x_test))

image = x_test[image_index]
plt.imshow(image[:,:,0], cmap='gray', interpolation='none')

image = np.expand_dims(image, axis=0)
scores = model.predict(image)

index = np.argmax(scores)
plt.title("Letter: " + str( class_idx[ index ] ))

def build_model(hp):

  categories=number_classes

  ly1_neurons = hp.Int('conv_1_filter', min_value=64, max_value=128, step=16)
  ly2_neurons = hp.Int('conv_2_filter', min_value=128, max_value=256, step=16)
  ly3_neurons = hp.Int('conv_3_filter', min_value=256, max_value=512, step=16)

  kernel_size_1 = hp.Choice('conv_1_kernel', values=[3,5])
  kernel_size_2 = hp.Choice('conv_2_kernel', values=[3,5])
  kernel_size_3 = hp.Choice('conv_3_kernel', values=[3,5])

  model = Sequential()
  
  model.add(Conv2D(filters=ly1_neurons, kernel_size=kernel_size_1, input_shape=x_train.shape[1:]))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.2))

  model.add(Conv2D(filters=ly2_neurons, kernel_size=kernel_size_2))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.2))

  model.add(Conv2D(filters=ly3_neurons, kernel_size=kernel_size_3))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.2))

  model.add(Flatten())
  model.add(Dense( 1024 ))
  model.add(Activation('relu'))
  model.add(Dense( 512 ))
  model.add(Activation('relu'))
  model.add(Dense( categories ))
  model.add(Activation('softmax'))

  model.compile(
      optimizer=optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
      loss='categorical_crossentropy',
      metrics=['accuracy']
      )

  return model

tuner_search = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    )

tuner_search.search(x_train, y_train, epochs=5, validation_split=0.2)
model = tuner_search.get_best_models(num_models=1)[0]
model.summary()

history = model.fit(
    x_train,
    y_train,
    batch_size=320,
    epochs=20,
    validation_data=(x_test, y_test),
    verbose=1)