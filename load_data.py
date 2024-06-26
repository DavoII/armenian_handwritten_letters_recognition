# -*- coding: utf-8 -*-
import sys
import glob
import cv2
import numpy as np
from PIL import Image
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.1,
    shear_range=0.3,
    height_shift_range=0.1,
    zoom_range=0.1,
    data_format='channels_last'
    )

upper_case_class_idx = [
    'Ա', 'Բ', 'Գ', 'Դ', 'Ե', 'Զ', 'Է', 'Ը', 'Թ', 'Ժ', 'Ի', 'Լ', 'Խ',
    'Ծ', 'Կ', 'Հ', 'Ձ', 'Ղ', 'Ճ', 'Մ', 'Յ', 'Ն', 'Շ', 'Ո', 'Չ', 'Պ', 
    'Ջ', 'Ռ', 'Ս', 'Վ', 'Տ', 'Ր', 'Ց', ' Ու', 'Փ', 'Ք', 'ԵՎ', 'Օ', 'Ֆ'
]

lower_case_class_idx = [
    'ա', 'բ', 'գ', 'դ', 'ե', 'զ', 'է', 'ը', 'թ', 'ժ', 'ի', 'լ', 'խ',
    'ծ', 'կ', 'հ', 'ձ', 'ղ', 'ճ', 'մ', 'յ', 'ն', 'շ', 'ո', 'չ', 'պ', 
    'ջ', 'ռ', 'ս', 'վ', 'տ', 'ր', 'ց', 'ու', 'փ', 'ք', 'և', 'օ', 'ֆ'
]

data = []
labels = []
label_classes = []

img_size = 64

def getFIlePath(folderName, className):
    return './' + folderName +  '/' + className + '/*.png'

def get_agumented_data(img):
    
    result = []
    img = np.reshape(img, (img_size, img_size, 1))
    img = np.expand_dims(img, axis=0)
    aug_iter = gen.flow(img)
    for i in range(3):
        aug_img = next(aug_iter)[0].astype(np.float32)
        aug_img = np.reshape(aug_img, (img_size, img_size))
        result.append(aug_img)
    
    return result

def append_img_to_data(index, fileName, args, firstIndex = 0):
    image = Image.open(fileName)
    # imgResized = image.resize((img_size, img_size), Image.ANTIALIAS)
    imgInit = np.array(image)
    
    img = []
    img = cv2.fastNlMeansDenoising(imgInit)
    index_x = firstIndex + index
    if '-a' in args:
        data.append(img)
        labels.append(index_x)
        for single_img in get_agumented_data(img):
            data.append(single_img)
            labels.append(index_x)
    else:
        data.append(img)
        labels.append(index)

def load_data(args):
    print('loading data')
    firstIndex = 0
    if '-u' in args:
        firstIndex += 1
        folderName = 'Upper cleaned/Upper cleaned'
        for index, className in enumerate(upper_case_class_idx):
            for idx, fileName in enumerate(glob.glob(getFIlePath(folderName, className))):
                print(className, idx)
                append_img_to_data(index, fileName, args)
        for num in upper_case_class_idx:
            label_classes.append(num)
    
    if '-l' in args:
        folderName = 'Lower cleaned/Lower cleaned'
        for index, className in enumerate(lower_case_class_idx):
            for idx, fileName in enumerate(glob.glob(getFIlePath(folderName, className))):
                print(className, idx)
                append_img_to_data(index, fileName, args, firstIndex)
        for num in lower_case_class_idx:
            label_classes.append(num)


system_arg = sys.argv
system_arg.pop(0)


if not system_arg:
    load_data(['-u', '-l'])    
elif len(system_arg) == 1 and '-a'in system_arg:
    load_data(['-l', '-u', '-a'])
else: 
    load_data(system_arg)
        
labels = np_utils.to_categorical(labels, len(label_classes))

np.save('./dataset_v2/data.npy', data)
np.save('./dataset_v2/labels.npy', labels)
np.save('./dataset_v2/label_classes.npy', label_classes)