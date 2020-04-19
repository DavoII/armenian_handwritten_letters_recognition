import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator(rotation_range=12, width_shift_range=0.1, shear_range=0.3,
                        height_shift_range=0.1, zoom_range=0.1, data_format='channels_last')

data = np.load('./dataset_v2/data.npy', encoding='bytes') 
labels = np.load('./dataset_v2/labels.npy', encoding='bytes')

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

x_train /= 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 64, 64, 1)
x_test = x_test.reshape(x_test.shape[0], 64, 64, 1)

img = x_train[1]
plt.imshow(img[:,:,0], cmap='gray')
print(img.shape)

img = np.expand_dims(img, axis=0)
aug_iter = gen.flow(img)

f = plt.figure(figsize=(12,6))
for i in range(8):
    sp = f.add_subplot(2, 26//3, i+1)
    sp.axis('Off')
    aug_img = next(aug_iter)[0].astype(np.float32)
    plt.imshow(aug_img[:,:,0], cmap='gray')

