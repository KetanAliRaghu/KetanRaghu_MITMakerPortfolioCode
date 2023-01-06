import tensorflow as tf
import keras
from keras import layers as l , optimizers , losses , metrics
from keras.datasets import mnist

(X_train , y_train) , (X_test , y_test) = mnist.load_data()
X_train = X_train.reshape(-1 , 28 , 28 , 1).astype('float32') / 255.0
X_test = X_test.reshape(-1 , 28 , 28 , 1).astype('float32') / 255.0

model = keras.Sequential([
    l.Input(shape = (28 , 28 , 1)),
    l.Conv2D(64 , 3 , padding = 'same'),
    l.ReLU(),
    l.Conv2D(128 , 3 , padding = 'same'),
    l.ReLU(),
    l.Flatten(),
    l.Dense(10)
], name = 'model')


class CustomFit(keras.Model):
    def __init__(self , model):
        super(CustomFit , self).__init__()
        self.model = model

    def compile(self,
              optimizer = optimizers.Adam(),
              loss = None,
              metrics = None,
              loss_weights = None,
              weighted_metrics = None,
              run_eagerly = None,
              steps_per_execution = None,
              jit_compile = None,
              **kwargs):
        super(CustomFit, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, data):
        X , y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(X , training = True)
            loss = self.loss(y , y_pred)

        training_vars = self.trainable_variables
        gradients = tape.gradient(loss , training_vars)

        self.optimizer.apply_gradients(zip(gradients , training_vars))
        acc_metric.update_state(y , y_pred)

        return {'loss': loss , 'accuracy': acc_metric.result()}

    def test_step(self, data):
        X , y = data

        y_pred = self.model(X , training = False)
        loss = self.loss(y , y_pred)
        acc_metric.update_state(y , y_pred)

        return {'loss': loss , 'accuracy': acc_metric.result()}


training = CustomFit(model)

acc_metric = metrics.SparseCategoricalAccuracy(name = 'accuracy')
training.compile(
    optimizer = optimizers.Adam(),
    loss = losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)

training.fit(X_train , y_train , batch_size = 32 , epochs = 2)
training.evaluate(X_test , y_test , batch_size = 32)