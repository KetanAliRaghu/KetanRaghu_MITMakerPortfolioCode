import tensorflow as tf
import keras
from keras import layers as l
from keras.datasets import mnist
from keras import optimizers , losses

(X_train , y_train) , (X_test , y_test) = mnist.load_data()
X_train = X_train.reshape(-1 , 28 , 28 , 1).astype('float32') / 255
X_test = X_test.reshape(-1 , 28 , 28 , 1).astype('float32') / 255

class CNNBlock(l.Layer):
    def __init__(self , out_channels , kernel_size = 3):
        super(CNNBlock , self).__init__()
        self.conv = l.Conv2D(out_channels , kernel_size , padding = 'same')
        self.bn = l.BatchNormalization()

    def call(self, input_tensor, training=False, **kwargs):
        x = self.conv(input_tensor)
        x = self.bn(x , training = training)
        x = tf.nn.relu(x)
        return x

class ResBlock(l.Layer):
    def __init__(self , channels):
        super(ResBlock , self).__init__()
        self.cnn1 = CNNBlock(channels[0])
        self.cnn2 = CNNBlock(channels[1])
        self.cnn3 = CNNBlock(channels[2])
        self.pooling = l.MaxPooling2D()
        self.identity_mapping = l.Conv2D(channels[1] , 1 , padding = 'same')

    def call(self , input_tensor , training = False , **kwargs):
        x = self.cnn1(input_tensor , training = training)
        x = self.cnn2(x , training = training)
        x = self.cnn3(x + self.identity_mapping(input_tensor) , training = training)
        return self.pooling(x)

"""
model = keras.Sequential([
    ResBlock([32, 32, 64]),
    ResBlock([128, 128, 256]),
    ResBlock([128, 256, 512]),
    l.GlobalAveragePooling2D(),
    l.Dense(10)
])

model.compile(
    optimizer = optimizers.Adam(),
    loss = losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)

model.fit(X_train , y_train , batch_size = 64 , epochs = 3)
print(model.evaluate(X_test , y_test , batch_size = 64))
"""

class ResNet_Like(keras.Model):
    def __init__(self , num_classes = 10):
        super(ResNet_Like , self).__init__()
        self.block1 = ResBlock([32 , 32 , 64])
        self.block2 = ResBlock([128, 128, 256])
        self.block3 = ResBlock([128, 256, 512])
        self.pool = l.GlobalAveragePooling2D()
        self.classifier = l.Dense(num_classes)

    def call(self , input_tensor , training = False , **kwargs):
        x = self.block1(input_tensor , training = training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.pool(x)
        return self.classifier(x)

    def model(self):
        x = keras.Input(shape = (28 , 28 , 1))
        return keras.Model(inputs = [x] , outputs = self.call(x))


model = ResNet_Like(num_classes = 10)
model.compile(
    optimizer = optimizers.Adam(),
    loss = losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)

model.fit(X_train , y_train , batch_size = 64 , epochs = 3)
print(model.summary())
print(model.model().summary())
print(model.evaluate(X_test , y_test , batch_size = 64))

