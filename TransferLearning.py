import numpy as np
import cv2 as cv
import matplotlib.pylab as plt

from PIL import Image
import os
import pathlib

import keras
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers as l
from keras.models import Sequential
from sklearn.model_selection import train_test_split

IMAGE_SHAPE = (224 , 224)

classifer = Sequential([
    hub.KerasLayer('https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4')
])

goldfish = Image.open('data/goldfish.jpg').resize(IMAGE_SHAPE)
goldfish = np.array(goldfish)/255

result = classifer.predict(goldfish[np.newaxis , ...])
print(result)

# flower_data = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
# data_dir = tf.keras.utils.get_file('flower_photos' , origin = flower_data , cache_dir = '.' , untar = True)
# print(data_dir)

data_dir = pathlib.Path('data/datasets/flower_photos')
image_count = len(list(data_dir.glob('*/*.jpg'))[:5])
print(image_count)

flower_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*'))
}

flower_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4
}

X , y = [] , []

for flower_name , images in flower_images_dict.items():
    for image in images:
        img = cv.imread(str(image))
        resized = cv.resize(img , IMAGE_SHAPE)
        X.append(resized)
        y.append(flower_labels_dict[flower_name])
print('done')

X = np.array(X)
y = np.array(y)
X_train , X_test , y_train , y_test = train_test_split(X , y , random_state = 0)

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

feature_extraction_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
pretrained_model_minustop = hub.KerasLayer(feature_extraction_model,
                                           input_shape = (224 , 224 , 3),
                                           trainable = False)

model = Sequential([
    pretrained_model_minustop,
    l.Dense(5)
])
print(model.summary())

model.compile(optimizer = 'adam',
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])
model.fit(X_train_scaled , y_train , epochs = 5)
print(model.evaluate(X_test_scaled , y_test))
