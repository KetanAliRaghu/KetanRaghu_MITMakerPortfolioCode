import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from keras.layers import Input , Dropout , Dense
from keras.models import Model
from keras.metrics import BinaryAccuracy , Precision , Recall

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report

preprocess_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
encoder_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'

BERT_preprocess_model = hub.KerasLayer(preprocess_url)
BERT_encoder_model = hub.KerasLayer(encoder_url)

df = pd.read_csv('data/spam.csv' , encoding_errors = 'replace')
# print(df.groupby('Category').describe())

df_spam = df[df['Category'] == 'spam']
df_ham = df[df['Category'] == 'ham'].sample(df_spam.shape[0])

df_balanced = pd.concat([df_spam , df_ham])
print(df_balanced['Category'].value_counts())

df_balanced['spam'] = df_balanced['Category'].apply(lambda x: 1 if x == 'spam' else 0)

X_train , X_test , y_train , y_test = train_test_split(df_balanced['Message'],
                                                       df_balanced['spam'],
                                                       stratify = df_balanced['spam'])


# BERT Layers
input_layer = Input(shape = () , dtype = tf.string , name = 'text')
preprocessed_text = BERT_preprocess_model(input_layer)
outputs = BERT_encoder_model(preprocessed_text)

# NN Layers
dropout_1 = Dropout(0.1)(outputs['pooled_output'])
dense_1 = Dense(1 , activation = 'sigmoid' , name = 'output')(dropout_1)

"""
# Final Model
model = Model(inputs = [input_layer] , outputs = dense_1)
print(model.summary())

METRICS = [BinaryAccuracy(),
           Precision(),
           Recall()]

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train , y_train , epochs = 10)

model.save('models/FunctionalAPI_BERT-Classifier')
"""
model = tf.keras.models.load_model('models/FunctionalAPI_BERT-Classifier')

print(model.evaluate(X_test , y_test))
y_pred = model.predict(X_test).flatten()
# if the value < 0.5 it is set to 0 else it is 1
y_pred = np.where(y_pred < 0.5 , 0 , 1)
print(y_pred)
print(classification_report(y_test , y_pred))
