import tensorflow as tf
import keras
from keras import layers as l
from keras.datasets import mnist
from keras import losses , optimizers

(X_train , y_train) , (X_test , y_test) = mnist.load_data()
X_train = X_train.reshape(-1 , 28 * 28).astype('float32') / 255
X_test = X_test.reshape(-1 , 28 * 28 , 1).astype('float32') / 255

class Dense(l.Layer):
    def __init__(self , units):
        super(Dense , self).__init__()
        self.units = units

    def call(self , inputs , **kwargs):
        return tf.matmul(inputs , self.w) + self.b

    # Allows the removal of mandatory input_dim parameter
    def build(self, input_shape):
        self.w = self.add_weight(
            name = 'w',
            shape = (input_shape[-1] , self.units),
            initializer = 'random_normal',
            trainable = True
        )
        self.b = self.add_weight(
            name = 'b',
            shape = (self.units,),
            initializer = 'zeros',
            trainable = True
        )

class MyReLU(l.Layer):
    def __init__(self):
        super(MyReLU , self).__init__()

    def call(self, x, **kwargs):
        return tf.math.maximum(0.0 , x)

class CustomModel(keras.Model):
    def __init__(self , num_classes = 10):
        super(CustomModel , self).__init__()
        self.dense1 = Dense(64)
        self.dense2 = Dense(10)
        # self.dense1 = Dense(64 , 784)
        # self.dense2 = Dense(10 , 64)
        self.relu = MyReLU()

    def call(self , input_tensor , **kwargs):
        x = self.relu(self.dense1(input_tensor))
        return self.dense2(x)

model = keras.Sequential([
    l.Dense(64 , activation = 'relu'),
    l.Dense(10)
])
model.compile(
    optimizer = optimizers.Adam(),
    loss = losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)

model.fit(X_train , y_train , batch_size = 32 , epochs = 2)
model.evaluate(X_test , y_test , batch_size = 32)
