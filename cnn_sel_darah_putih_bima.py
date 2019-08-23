# =====================================PART 1================================
# IMPORT LIBRARIES
import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os

# DEFINISI FUNGSI

def buat_file(path):
  files = []
  # r=root, d=directories, f = files
  for r, d, f in os.walk(path):
      for file in f:
          if '.jpg' in file:
              files.append(os.path.join(r, file))
          elif '.JPG' in file:
              files.append(os.path.join(r, file))

  return files


# define location of dataset
def leukemia_dataset(files)
  photos, labels = list(), list()
  # enumerate files in the directory
  for file in files:
    # load image
    photo = load_img(file, target_size=(200, 200))
    # convert to numpy array
    photo = img_to_array(photo)
    # store
    photos.append(photo)
  # convert to a numpy arrays
  photos = asarray(photos)
  print(photos.shape)
  # save the reshaped photos
  save('citra_sel_darah_putih.npy', photos)
  return photos


def pre_processing_labels(labels):

  label_encoder = LabelEncoder()
  integer_encoded = label_encoder.fit_transform(labels)
  print(integer_encoded)

  # binary encode
  onehot_encoder = OneHotEncoder(sparse=False)
  integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
  y_label = onehot_encoder.fit_transform(integer_encoded)
  return y_label

classes = [
    'Limfoblas',
    'Limfosit',
    'Metamielosit',
    'Monosit',
    'Segment',
    'Stab'
]







# ====================================================================================PROSES===========================================================
# CNN Architecture
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import preprocess_input
from skimage.transform import resize   # for resizing images



path = 'E:\\Bimantara\\LOMBA\\IMERI FKUI\\Source\\Raw_copy'
labels = pd.read_excel('label_darah.xlsx')
# SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(leukemia_dataset(buat_file(path)), pre_processing_labels(labels), test_size=0.3, random_state=0)



base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (200, 200, 3))
X_train = base_model.predict(X_train)
X_test = base_model.predict(X_test)
X_train.shape, X_test.shape
X_train = X_train.reshape(1730, 6*6*512)
X_test = X_test.reshape(742, 6*6*512)
train = X_train/X_train.max()
X_test = X_test/X_train.max()

model = Sequential()
model.add(InputLayer((6*6*512,)))
model.add(Dense(units=1024, activation='sigmoid'))
model.add(Dense(6, activation='softmax'))

# ii. Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# iii. Training the model
model.fit(train, y_train, epochs=100, validation_data=(X_test, y_test))

# BUAT TAMPILAN HASIL

img = plt.imread('Limfoblas.jpg')
test_img = np.array(img)
a = resize(test_img, preserve_range=True, output_shape=(200,200)).astype(int)
test_image = np.array(a)
test_image.shape

# preprocessing the images
test_image = preprocess_input(test_image)
test_image.shape
# extracting features from the images using pretrained model
test_image = base_model.predict(test_image.reshape(1,200,200,3))
test_image.shape
# zero centered images
test_image = test_image/test_image.max()
predictions = model.predict(test_image.reshape(1,6*6*512))

labels_dict
top_3 = np.argsort(predictions[0])[:-4:-1]
plt.imshow(img)
for i in range(3):
    print("{}".format(classes[top_3[i]])+" ({:.3})".format(predictions[0][top_3[i]]))