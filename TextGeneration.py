import tensorflow as tf
import string
import requests
import numpy as np

import keras
from keras_preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras import Sequential
from keras.utils import pad_sequences
from keras.layers import Dense , CuDNNLSTM , Embedding

response = requests.get('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt')
data = response.text.split('\n')
print(data[253])

data = data[253:]
print(len(data))
data = ' '.join(data)
# print(data)

def clean_text(doc):
    tokens = doc.split()
    table = str.maketrans('' , '' , string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]

    return tokens

tokens = clean_text(data)
print(tokens[:50])
print(len(tokens))
print(len(set(tokens)))

length = 51
lines = []

for i in range(length , len(tokens)):
    seq = tokens[i - length:i]
    line = ' '.join(seq)
    lines.append(line)

    if i > 200000:
        break

print(len(lines))

""" PREPARING THE DATA """
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

sequences = np.array(sequences)
X , y = sequences[: , :-1] , sequences[: , -1]

vocab_size = len(tokenizer.word_index) + 1

y = to_categorical(y , num_classes = vocab_size)

seq_length = X.shape[1]

""" LSTM MODEL """
model = Sequential([
    Embedding(vocab_size , 50 , input_length = seq_length),
    CuDNNLSTM(100 , return_sequences = True),
    CuDNNLSTM(100),
    Dense(100 , activation = 'relu'),
    Dense(vocab_size , activation = 'softmax')
])

print(model.summary())

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(X , y , batch_size = 256 , epochs = 10)
