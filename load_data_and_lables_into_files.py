import glob
import PIL
import cv2
import numpy as np

from keras.utils import np_utils

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

def getFIlePath(className):
    return './dataset/Upper cleaned/Upper cleaned/' + className + '/*.png'

for index, className in enumerate(upper_case_class_idx):
    for idx, fileName in enumerate(glob.glob(getFIlePath(className))):
        print(className, idx)
        img = np.array(PIL.Image.open(fileName))
        median = []
        median = cv2.fastNlMeansDenoising(img)
        data.append(median)
        labels.append(index)
        
data = np.array(data)
labels = np_utils.to_categorical(labels, len(upper_case_class_idx))

np.save('./dataset_v2/data.npy', data)
np.save('./dataset_v2/labels.npy', labels)
