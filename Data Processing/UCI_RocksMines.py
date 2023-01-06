"""
DEMONSTRATING AND ADDRESSING OVER-FITTING
"""

import tensorflow as tf
import keras
import keras.layers as l
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/sonar_all-data.csv', header = None)
print(df.shape)
# Checking for null values
# print(df.isna().sum())

# Checking the number of each value in a column
# print(df[60].value_counts())
df[60].replace({'M': 0 , 'R': 1} , inplace = True)

y = df[60]
X = df.drop(60 , axis = 'columns')

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.25 , random_state = 1)

model = keras.models.Sequential([
    l.Dense(60 , input_dim = 60 , activation = 'relu'),
    l.Dropout(0.5),
    l.Dense(30 , activation = 'relu'),
    l.Dropout(0.5),
    l.Dense(15 , activation = 'relu'),
    l.Dropout(0.5),
    l.Dense(1 , activation = 'sigmoid')
])

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
model.fit(X_train , y_train , epochs = 100 , batch_size = 8)