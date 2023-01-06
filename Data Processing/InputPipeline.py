import tensorflow as tf
import keras

daily_sales_numbers = [21 , 22 , -108 , 31 , -1 , 32 , 34 , 31 , 68 , -23]

tf_dataset = tf.data.Dataset.from_tensor_slices(daily_sales_numbers)
# for sales in tf_dataset.as_numpy_iterator():
    # print(sales) # prints each value

# for sales in tf_dataset.take(3): # gets only the first 3 values
   # print(sales.numpy())

# tf_dataset = tf_dataset.filter(lambda x : abs(x))
tf_dataset = tf_dataset.map(lambda x : abs(x))
tf_dataset = tf_dataset.map(lambda x : x * 72)
tf_dataset = tf_dataset.shuffle(3)
# for sale in tf_dataset.as_numpy_iterator():
    # print(sale)
# for sale in tf_dataset.batch(2):
    # print(sale.numpy()) # prints in batches of 2 (lists each with 2 values)

"""The above code can be chained to:
tf_dataset.filter(lambda x: x > 0).map(lambda y: y * 72).shuffle(3).batch(2)
"""

train_size = int(len(list(tf_dataset)) * 0.8)
# TensorFlow data splits
train_ds = tf_dataset.take(train_size)
test_ds = tf_dataset.skip(train_size)

for val in train_ds.as_numpy_iterator():
    print(val / 72)
for val in test_ds.as_numpy_iterator():
    print(val / 72)
