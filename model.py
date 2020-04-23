import tensorflow as tf
import argparse
from tensorflow import keras
from PIL import Image
from keras.preprocessing import image
import numpy as np


arg = argparse.ArgumentParser(description="Image  classifier")
arg.add_argument('-i','--image')

image_path = vars(arg.parse_args())

model = tf.keras.models.load_model('/Users/user/Desktop/LearningTensorflow/Cats_dogs/cats_dogs_model.h5',compile=False)



img = image.load_img(image_path['image'],target_size=(150,150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)




classes = model.predict(x)

if classes[0]==0:
    print('This is a cat')
else:
    print('This is a dog')
