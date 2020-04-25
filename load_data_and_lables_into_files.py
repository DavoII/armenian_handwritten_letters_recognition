import sys
import glob
from PIL import Image
#import cv2
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
uppercase_data = []
uppercase_labels = []
lowercase_data = []
lowercase_labels = []

def getFIlePath(folderName, className):
    return './' + folderName +  '/' + className + '/*.png'


def load_data(args):
        if '-u' in args:
            folderName = 'Upper cleaned'
            for index, className in enumerate(upper_case_class_idx):
                for idx, fileName in enumerate(glob.glob(getFIlePath(folderName, className))):
                    print(className, idx)
                    img = np.array(Image.open(fileName))
                    #median = []
                    #median = cv2.fastNlMeansDenoising(img)
                    uppercase_data.append(img)
                    uppercase_labels.append(index)
        
        if '-l' in args:
            folderName = 'Lower cleaned'
            for index, className in enumerate(lower_case_class_idx):
                for idx, fileName in enumerate(glob.glob(getFIlePath(folderName, className))):
                    print(className, idx)
                    img = np.array(Image.open(fileName))
                    #median = []
                    #median = cv2.fastNlMeansDenoising(img)
                    lowercase_data.append(img)
                    lowercase_labels.append(index)

system_arg = sys.argv
system_arg.pop(0)

if not system_arg:
    load_data(['-l', '-u'])
elif len(system_arg) == 1 and '-a'in system_arg:
    load_data(['-l', '-u', '-a'])
else: 
    load_data(system_arg)
        
uppercase_data = np.array(uppercase_data)
lowercase_data = np.array(lowercase_data)
uppercase_labels = np_utils.to_categorical(uppercase_labels, len(upper_case_class_idx))
lowercase_labels = np_utils.to_categorical(lowercase_labels, len(lower_case_class_idx))
        
data = np.concatenate(uppercase_data, lowercase_data)
labels = np.concatenate(uppercase_labels, lowercase_labels)


np.save('./dataset_v2/data.npy', data)
np.save('./dataset_v2/labels.npy', labels)