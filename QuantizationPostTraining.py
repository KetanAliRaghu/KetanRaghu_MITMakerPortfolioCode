"""
Quantization decreases the values of the stored weights and converts to a
smaller datatype to converse memory and storage so the AI can be deployed on
edge devices (small devices with less power than PCs, ex: smartwatches, Arduino)
"""

import tensorflow as tf
import keras
from keras.layers import Flatten , Dense
import numpy as np

(X_train , y_train) , (X_test , y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

"""
model = keras.Sequential([
    Flatten(input_shape = (28 , 28)),
    Dense(100 , activation = 'relu'),
    Dense(10 , activation = 'sigmoid')
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train , y_train , epochs = 5)

print(model.evaluate(X_test , y_test))
model.save('models/QuantizationTestingModel')
"""

# model = tf.keras.models.load_model('models/QuantizationTestingModel')

converter = tf.lite.TFLiteConverter.from_saved_model('models/QuantizationTestingModel')
tfLite_model = converter.convert()

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tfLite_quantized_model = converter.convert()

print('tfLite_model:' , f'{len(tfLite_model)} bytes')
print('tfLite_quantized_model:' , f'{len(tfLite_quantized_model)} bytes')
