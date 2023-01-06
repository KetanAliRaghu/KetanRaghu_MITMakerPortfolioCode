import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense , Conv2D , Flatten , MaxPool2D
import matplotlib.pyplot as plt
import numpy as np
from neuralplot import ModelPlot

print(tf.config.experimental.list_physical_devices())
print(tf.test.is_built_with_cuda())

(X_train , y_train) , (X_test , y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

# X_train_flat = X_train.reshape(len(X_train), X_train.shape[1] * X_train.shape[2])
# X_test_flat = X_test.reshape(len(X_test), X_test.shape[1] * X_test.shape[2])

model = keras.Sequential([
    Flatten(input_shape = (28,28)),
    Dense(128 , activation = 'relu'),
    Dense(10 , activation = 'sigmoid')
])

tf_callback = tf.keras.callbacks.TensorBoard(log_dir ='../logs/', histogram_freq = 1)

model.compile(optimizer = 'SGD',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train , y_train , epochs = 5 , callbacks = [tf_callback])
model.evaluate(X_test , y_test)

y_pred = model.predict(X_test)
# print(np.argmax(y_pred[0]))
# plt.matshow(X_test[0])
# plt.show()

y_pred_labels = [np.argmax(i) for i in y_pred]
print(tf.math.confusion_matrix(labels = y_test , predictions = y_pred_labels))
