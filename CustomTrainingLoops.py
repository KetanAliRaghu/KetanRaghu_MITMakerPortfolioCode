import tensorflow as tf
import keras
from keras import layers as l , losses , optimizers , metrics
import tensorflow_datasets as tfds

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

EPOCHS = 5
OPTIMIZER = optimizers.Adam()
LOSS_FN = losses.SparseCategoricalCrossentropy(from_logits = True)
ACC_METRIC = metrics.SparseCategoricalAccuracy()

# Training Loop
for epoch in range(EPOCHS):
    print(f'\nStart of Training Epoch {epoch}')
    for batch_idx , (X_batch , y_batch) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            y_pred = model(X_batch , training = True)
            loss = LOSS_FN(y_batch , y_pred)

        gradients = tape.gradient(loss , model.trainable_weights)
        OPTIMIZER.apply_gradients(zip(gradients , model.trainable_weights))
        ACC_METRIC.update_state(y_batch , y_pred)

    train_acc = ACC_METRIC.result()
    print(f'Accuracy over epoch: {train_acc}')
    ACC_METRIC.reset_states()

# Testing Loop
for batch_idx , (X_batch , y_batch) in enumerate(ds_train):
    y_pred = model(X_batch , training = True)
    ACC_METRIC.update_state(y_batch , y_pred)

test_acc = ACC_METRIC.result()
print(f'Accuracy over Test Set: {test_acc}')
ACC_METRIC.reset_states()
