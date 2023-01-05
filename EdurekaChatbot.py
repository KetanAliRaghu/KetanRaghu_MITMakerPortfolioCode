import keras.saving.save
import tensorflow as tf
import  numpy as np
import pandas as pd
import json
import nltk
import math

from keras.preprocessing.text import Tokenizer
from keras import Model
from keras.layers import Input , Embedding , LSTM,\
    Dense , GlobalMaxPooling1D , Flatten
import matplotlib.pyplot as plt
from keras.utils import pad_sequences
from keras import losses , optimizers

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/emotion-emotion_69k.csv')

for i in range(len(df['empathetic_dialogues'])):
    df['empathetic_dialogues'][i] = df['empathetic_dialogues'][i].replace('Customer :' , '').replace('Agent :' , '')

tokenizer = Tokenizer(num_words = 18000)
tokenizer.fit_on_texts(df['empathetic_dialogues'])

train = tokenizer.texts_to_sequences(df['empathetic_dialogues'])
X = pad_sequences(train)

labEnc = LabelEncoder()
y = labEnc.fit_transform(df['labels'])

input_shape = X.shape[0]
vocab = len(tokenizer.word_index)
output_len = labEnc.classes_.shape[0]

# Model
input_layer = Input(shape = (input_shape,))
x = Embedding(vocab + 1 , 10)(input_layer)
x = LSTM(10 , return_sequences = True)(x)
x = Flatten()(x)
x = Dense(output_len , activation = 'sigmoid')(x)

model = Model(inputs = [input_layer] , outputs = x)
model.compile(
    loss = losses.SparseCategoricalCrossentropy(),
    optimizer = optimizers.Adam(),
    metrics = ['accuracy']
)

print(model.summary())

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 2)

report = model.fit(X_train , y_train , epochs = 20)

model.save('models/EdurekaChatbot2_0Acc_20Epoch')

print(report)
print(model.evaluate(X_test , y_test))

print(X , y)
print(X_train , X_test)
print(y_train , y_test)


model = keras.saving.save.load_model('models/EdurekaChatbot_0Acc_20Epoch')

def pad_seq(texts , length):
    returned = []

    if len(texts) <= length:
        for k in range(length - len(text)):
            returned.append(0)

        for l in range(len(texts)):
            returned.append(texts[l])

        return np.array(returned)

    for k in reversed(range(int(len(texts) / length))):
        returned.append(texts[len(texts) - (length * k) - length : len(texts) - (length * k)])
    if len(texts) % length != 0:
        padded = texts[:len(texts) - (length * int(len(texts) / length))]
        for j in range(length - len(padded)):
            padded.insert(0 , 0)
        returned.insert(padded , 0)
    return np.array(returned)

text = 'what new home'
text = tokenizer.texts_to_sequences([text])
text = np.array(text).reshape(-1)
text = pad_sequences([text] , 107)
text = model.predict(text)
text = text.argmax()
print(f'BOT: {labEnc.inverse_transform([text])}')
