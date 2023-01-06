import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers as l , losses , optimizers , callbacks
import tensorflow_datasets as tfds
from neuralplot import ModelPlot

(ds_train , ds_test) , ds_info = tfds.load(
    name = 'mnist',
    split = ['train' , 'test'],
    shuffle_files = True,
    as_supervised = True, # Will return tuple (img , label) otherwise dict
    with_info = True # returns the information about the dataset
)

def normalize_img(image , label):
    return tf.cast(image , tf.float32) / 255.0 , label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128

ds_train = ds_train.map(normalize_img , num_parallel_calls = AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

model = keras.Sequential([
    keras.Input((28 , 28 , 1)),
    l.Conv2D(32 , 3 , activation = 'relu'),
    l.Flatten(),
    l.Dense(10)
])

# Full Custom Callback (check link for more info)
# https://www.tensorflow.org/guide/keras/custom_callback
class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.90:
            print('Accuracy over 90%, Training ending')
            self.model.stop_training = True

save_callback = callbacks.ModelCheckpoint(
    filepath ='../models/CallbackCheckpoints/',
    save_weights_only = True,
    monitor = 'accuracy', # If using both training and validation set, need to specify val_ or train_
    save_best_only = False
)

# Custom Callback Function (lr = learn rate)
def scheduler(epoch , lr):
    if epoch < 2:
        return lr
    else:
        return lr * 0.99

lr_scheduler = callbacks.LearningRateScheduler(scheduler , verbose = 1)

model.compile(
    optimizer = optimizers.Adam(0.01),
    loss = losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)
model.fit(ds_train , epochs = 10,
          callbacks = [save_callback,
                       lr_scheduler,
                       CustomCallback()])

modelplot = ModelPlot(model = model , grid = True , connection = True , linewidth = 0.1)
modelplot.show()
