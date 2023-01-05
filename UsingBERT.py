import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import keras

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

preprocess_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
encoder_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'

BERT_preprocess_model = hub.KerasLayer(preprocess_url)
BERT_encoder_model = hub.KerasLayer(encoder_url)

text_test = ['nice movie indeed' , 'I don\'t like python programming']
# text_preprocessed = BERT_preprocess_model(text_test)


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

def get_sentence_embedding(vals):
    return BERT_encoder_model(BERT_preprocess_model(vals))['pooled_output']

print(get_sentence_embedding([
    '$500 discount. hurry up!',
    'Volleyball game today at 7'
]))

test_e = get_sentence_embedding([
    'banana',
    'grape',
    'mango',
    'jeff bezos',
    'elon musk',
    'bill gates'
])

print(cosine_similarity([test_e[0]] , [test_e[1]]))
print(cosine_similarity([test_e[1]] , [test_e[3]]))
