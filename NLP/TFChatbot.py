import keras.saving.save
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from nltk.tokenize import word_tokenize

from keras.preprocessing.text import Tokenizer
from keras import  callbacks
from keras.utils import pad_sequences
from keras.layers import Input , Embedding , CuDNNLSTM , Dense , GlobalMaxPooling1D , Flatten,\
    Bidirectional , SimpleRNN , LSTM
from keras.models import Model

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('../data/emotion-emotion_69k.csv')
data['empathetic_dialogues'] = data['empathetic_dialogues'].map(lambda x: x.replace('Customer :' , '').replace('Agent :' , '').replace('#' , '').lower())
data['empathetic_dialogues'] = data['empathetic_dialogues'].map(lambda x: x.replace('.' , '').replace(',' , '').replace('!' , '').replace('?' , ''))
data['empathetic_dialogues'] = data['empathetic_dialogues'].map(lambda x: x.replace('-' , '').replace('*' , '').replace('$' , '').replace('%' , ''))

vocab_list = []
for sent in data['empathetic_dialogues']:
    tokens = word_tokenize(sent)
    for token in tokens:
        if token not in vocab_list:
            vocab_list.append(token)

print(len(vocab_list))
# print(sorted(vocab_list))

tokenizer = Tokenizer(num_words = 18000)
tokenizer.fit_on_texts(data['empathetic_dialogues'])
train = tokenizer.texts_to_sequences(data['empathetic_dialogues'])

#apply padding
X_train = pad_sequences(train)

#encoding the outputs
le = LabelEncoder()
y_train = le.fit_transform(data['labels'])

#input length
input_shape = X_train.shape[1]
print(input_shape)
#define vocabulary
vocabulary = len(tokenizer.word_index)
print("number of unique words : ",vocabulary)
#output length
output_length = le.classes_.shape[0]
print("output length: ",output_length)

print(input_shape)
print(output_length)

class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.96:
            print('Accuracy over 96%, Training ending')
            self.model.stop_training = True

def scheduler(epoch , lr):
    if epoch < 2:
        return lr
    else:
        return lr * 0.99

lr_scheduler = callbacks.LearningRateScheduler(scheduler , verbose = 1)

i = Input(shape = (input_shape,))
# i2 = Input(shape = (input_shape2,))
x = Embedding(vocabulary + 1 , 10)(i)
x = LSTM(20 , return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length , activation = "softmax")(x)
model = Model(i , x)

#compiling the model
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

#training the model
train = model.fit(X_train , y_train , epochs = 200 , callbacks = [lr_scheduler , CustomCallback()])

# np.save('model_logs/TFChatbot_96Acc_history' , train.history)
# model.save('models/TFChatbot_96Acc')

model = keras.saving.save.load_model('../models/TFChatbot_96Acc')

text = 'hello'
text = tokenizer.texts_to_sequences([text])
text = np.array(text).reshape(-1)
text = pad_sequences([text] , 110)
text = model.predict(text)
text = text.argmax()
print(f'BOT: {le.inverse_transform([text])}')
