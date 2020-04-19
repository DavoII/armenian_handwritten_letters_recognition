import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
import seaborn as sn

upper_case_class_idx = [
             'Ա', 'Բ', 'Գ', 'Դ', 'Ե', 'Զ', 'Է', 'Ը', 'Թ', 'Ժ', 'Ի', 'Լ', 'Խ',
             'Ծ', 'Կ', 'Հ', 'Ձ', 'Ղ', 'Ճ', 'Մ', 'Յ', 'Ն', 'Շ', 'Ո', 'Չ', 'Պ', 
             'Ջ', 'Ռ', 'Ս', 'Վ', 'Տ', 'Ր', 'Ց', 'Ու', 'Պ', 'Ք', 'ԵՎ', 'Օ', 'Ֆ'
]
lower_case_class_idx = []

number_classes = len(upper_case_class_idx) + len(lower_case_class_idx)

data = np.load('./dataset_v2/data.npy', encoding='bytes') 
labels = np.load('./dataset_v2/labels.npy', encoding='bytes') 

print(data.shape)
print(labels.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.67,
                                                    test_size=0.33, random_state=42, shuffle=True, stratify=labels)

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

ly1_neurons = 512
ly2_neurons = 256
ly3_neurons = 128

categories=number_classes

model = Sequential()

model.add(Conv2D(ly1_neurons, (8,8), input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(ly2_neurons, (8,8)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(ly3_neurons, (8,8)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense( categories ))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

batch_size = 120
epochs = 10

history = model.fit(x_train, y_train, batch_size, epochs, verbose=1)

test_predictions = model.predict_classes(x_test)

y_test_asCategories = np.argmax(y_test, axis=1)

confusion = confusion_matrix(y_test_asCategories, test_predictions)

df_cm = pd.DataFrame(confusion, index = [i for i in upper_case_class_idx], columns = [i for i in upper_case_class_idx])
plt.figure(figsize = (40,40))
sn.heatmap(df_cm, annot=True)

import random
image_index = random.randint(1,len(x_test))

image = x_test[image_index]
plt.imshow(image[:,:,0], cmap='gray', interpolation='none')

image = np.expand_dims(image, axis=0)
scores = model.predict(image)

index = np.argmax(scores)
plt.title("Letter: " + str( upper_case_class_idx[ index ] ))