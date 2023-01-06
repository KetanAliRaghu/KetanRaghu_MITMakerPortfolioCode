"""
Quantization decreases the values of the stored weights and converts to a
smaller datatype to converse memory and storage so the AI can be deployed on
edge devices (small devices with less power than PCs, ex: smartwatches, Arduino)
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import keras
from keras.layers import Flatten , Dense
import numpy as np
from sklearn.metrics import classification_report

(X_train , y_train) , (X_test , y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

model = tf.keras.models.load_model('models/QuantizationTestingModel')

quantization_aware_model = tfmot.quantization.keras.quantize_model(model)

quantization_aware_model.compile(optimizer = 'adam',
                                 loss = 'sparse_categorical_crossentropy',
                                 metrics = ['accuracy'])
quantization_aware_model.fit(X_train , y_train , epochs = 3)
print(quantization_aware_model.evaluate(X_test , y_test))
print(quantization_aware_model.summary())

y_pred = quantization_aware_model.predict(X_test)
y_pred_collapsed = []

for vals in y_pred:
    max_val = 0
    for i in range(len(vals)):
        if vals[i] > vals[max_val]:
            max_val = i
    y_pred_collapsed.append(max_val)
y_pred_collapsed = np.array(y_pred_collapsed)

print(y_test)
print(y_pred_collapsed)
print(classification_report(y_test , y_pred_collapsed))

print('=======================================\n=======================================')

converter = tf.lite.TFLiteConverter.from_keras_model(quantization_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tfLite_quantized_model = converter.convert()
