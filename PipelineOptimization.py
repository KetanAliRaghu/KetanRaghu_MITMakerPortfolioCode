"""
MIMICS FILE OPEN/READ/MAP(TRANSFORM)/TRAIN LATENCIES
Using prefetch and cache
"""
import timeit

import tensorflow as tf
import time

"""
PREFETCH
"""

class FileDataset(tf.data.Dataset):
    @staticmethod
    def read_files_in_batches(num_samples):
        # open file
        time.sleep(0.003)
        for sample_idx in range(num_samples):
            time.sleep(0.015)
            yield sample_idx,

    def __new__(cls , num_samples = 3):
        return tf.data.Dataset.from_generator(
            cls.read_files_in_batches,
            output_signature = tf.TensorSpec(shape = (1,) , dtype = tf.int64),
            args = (num_samples,)
        )

def benchmark(dataset , num_epochs = 2):
    for epoch_num in range(num_epochs):
        for sample in dataset:
            time.sleep(0.01)

# timed
benchmark(FileDataset().prefetch(tf.data.AUTOTUNE))

"""
CACHE
"""

dataset = tf.data.Dataset.range(5)
dataset = dataset.map(lambda x : x**2)
# Caches the dataset so the map function isn't repeated every time
dataset = dataset.cache()

# retrieves data from cache, not recomputed
print(list(dataset.as_numpy_iterator()))
