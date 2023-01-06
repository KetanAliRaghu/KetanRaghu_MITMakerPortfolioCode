"""
Free Code Camp RNN model
"""

import keras
from keras import layers as l
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import pad_sequences
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584
MAX_LEN = 250
BATCH_SIZE = 64

(X_train , y_train) , (X_test , y_test) = imdb.load_data(num_words = VOCAB_SIZE)

X_train = pad_sequences(X_train , MAX_LEN)
X_train = pad_sequences(X_train , MAX_LEN)

"""
model = keras.Sequential([
    l.Embedding(VOCAB_SIZE , 32),
    l.LSTM(32),
    l.Dense(1 , activation = 'sigmoid')
])
print(model.summary())

model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

history = model.fit(X_train , y_train , epochs = 10 , validation_split = 0.2)
print(history)
model.save('models/FCC_RNN_Model')
"""

model = keras.models.load_model('../models/FCC_RNN_Model')

word_index = imdb.get_word_index()

def encode_text(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return pad_sequences([tokens] , MAX_LEN)[0]

text = 'that movie was just amazing, so amazing'
encoded = encode_text(text)
print(encoded)

reverse_word_index = {value: key for (key , value) in word_index.items()}

def decode_integers(integers):
    PAD = 0
    text = ''
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + ' '
    return text[:-1]

print(decode_integers(encoded))

def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1 , 250))
    pred[0] = encoded_text
    result = model.predict(pred)
    return result[0]

positive_review = 'That movie was so awesome! I really loved it and would watch it again because it was amazingly great'
print(predict(positive_review))
negative_review = 'that movie sucked. I hated it and wouldn\'t watch it again. Was one of the worst things I\'ve ever watched'
print(predict(negative_review))
