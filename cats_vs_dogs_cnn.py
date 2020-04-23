# -*- coding: utf-8 -*-
"""cats_vs_dogs_CNN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AeZcPgTWky7KsLlBGZ0YngUiFY5gy7Mx
"""

from tensorflow import keras

import os
import zipfile

local_zip = '/content/drive/My Drive/ML_Dataset/cats_dogs_dataset.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

import tensorflow as tf

model = keras.Sequential([
                          keras.layers.Conv2D(32,(3,3),activation = 'relu',input_shape=(150,150,3)),
                          keras.layers.MaxPool2D(2,2),
                          keras.layers.Conv2D(64,(3,3),activation='relu'),
                          keras.layers.MaxPool2D(2,2),
                          keras.layers.Conv2D(128,(3,3),activation = 'relu'),
                          keras.layers.MaxPool2D(2,2),
                          keras.layers.Conv2D(128,(3,3),activation = 'relu'),
                          keras.layers.MaxPool2D(2,2),
                          keras.layers.Flatten(),
                          keras.layers.Dense(512,activation = 'relu'),
                          keras.layers.Dense(1,activation = 'sigmoid')

                          
])

model.summary()

model.compile(optimizer= RMSprop(1e-4),loss='binary_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
    )
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = '/tmp/cats_dogs_dataset/train'
validation_dir = '/tmp/cats_dogs_dataset/validation'

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    class_mode='binary',
    batch_size = 20
)
test_data = test_datagen.flow_from_directory(
     validation_dir,
     target_size=(150,150),
     class_mode='binary',
     batch_size=20
 )

history = model.fit_generator(
     train_data,
     steps_per_epoch =100,
     epochs= 100,
     validation_data=test_data,
     validation_steps = 50,
     verbose = 2

 )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt

epochs = range(len(acc))
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.figure()

plt.plot(epochs,loss,'bo',label='Training acc')
plt.plot(epochs,val_loss,'b',label='Validation acc')

plt.legend()
plt.show()

model.save('cats_dogs_model.h5')

