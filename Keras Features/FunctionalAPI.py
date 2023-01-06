import tensorflow as tf
import keras
from keras import losses , optimizers , metrics
from keras.activations import relu , sigmoid
from keras import layers as l, regularizers as r
from keras.datasets import mnist


(X_train , y_train) , (X_test , y_test) = mnist.load_data()

inputs = keras.Input(shape = (28 , 28))
x = l.Flatten(input_shape = (28 , 28))(inputs)
x = l.Dense(512 , activation = relu , name = 'test_extraction')(x)
x = l.Dense(256 , activation = relu,
            kernel_regularizer = r.l2(0.01))(x)
outputs = l.Dense(10 , activation = sigmoid)(x)

model = keras.Model(inputs = inputs , outputs = outputs)
model.compile(optimizer = optimizers.Adam(),
              loss = losses.SparseCategoricalCrossentropy(),
              metrics = ['accuracy'])
model.fit(X_train , y_train , epochs = 5)

print(model.summary())
print(model.evaluate(X_test , y_test))
# Layer Extraction
print(model.layers[-2].output)
print(model.get_layer('test_extraction').output)
print([layer.output for layer in model.layers])


