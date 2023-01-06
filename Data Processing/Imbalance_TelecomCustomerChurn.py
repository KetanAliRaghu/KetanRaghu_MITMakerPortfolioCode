"""
HANDLING PREPROCESSING AND RUNNING A MODEL WITH PROCESSED DATA
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix , classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import layers as l

df = pd.read_csv('../data/Telco-Customer-Churn.csv')

# Removes customerID column from Dataframe (doesn't help this model)
df.drop('customerID' , axis = 'columns' , inplace = True)
print(df.dtypes)

# Converts strings to numbers + ignores columns with errors
pd.to_numeric(df.TotalCharges , errors = 'coerce')
# Returns all rows where TotalCharges is null
print(df[pd.to_numeric(df.TotalCharges , errors = 'coerce').isnull()])

# Storing everything in a new DataFrame without the null values
df1 = df[df.TotalCharges != ' ']

df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
print(df1.TotalCharges)

tenure_churn_no = df1[df1.Churn == 'No'].tenure
tenure_churn_yes = df1[df1.Churn == 'Yes'].tenure

plt.hist([tenure_churn_yes , tenure_churn_no],
         label = ['Churn=Yes' , 'Churn=No'])
plt.legend()
# plt.show()

# Lists the different possible values in object columns
def print_unique_colvals(daf):
    for col in daf:
        print(f'{col}: {daf[col].unique()}')


df1.replace('No phone service' , 'No' , inplace = True)
df1.replace('No internet service' , 'No' , inplace = True)

yes_no_cols = ['Partner' , 'Dependents' , 'PhoneService' , 'MultipleLines',
               'InternetService' , 'OnlineSecurity' , 'OnlineBackup' , 'DeviceProtection',
               'TechSupport' , 'StreamingTV' , 'StreamingMovies' , 'PaperlessBilling' , 'Churn']
for col in yes_no_cols:
    df1.replace({'Yes': 1 , 'No': 0} , inplace = True)
df1['gender'].replace({'Female': 1 , 'Male': 0} , inplace = True)

df2 = pd.get_dummies(data = df1 , columns = ['InternetService' , 'Contract' , 'PaymentMethod'])

cols_to_scale = ['tenure' , 'MonthlyCharges' , 'TotalCharges']
scaler = MinMaxScaler()

df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
# print_unique_colvals(df2)

X = df2.drop('Churn' , axis = 'columns')
y = df2['Churn']

count_class_0 , count_class_1 = df2.Churn.value_counts()
df_class_0 = df2[df2['Churn'] == 0]
df_class_1 = df2[df2['Churn'] == 1]

""" ---------------------- UNDER-SAMPLING ----------------------
print(count_class_0 , count_class_1)
print(df_class_0.shape , df_class_1.shape)

# Randomly takes the specified number of samples from the dataset
df_class_0_under = df_class_0.sample(count_class_1)

df_test_under = pd.concat([df_class_0_under , df_class_1] , axis = 0)

print('Random Under-Sampling')
print(df_test_under.Churn.value_counts())

y = df_test_under['Churn']
X = df_test_under.drop('Churn' , axis = 'columns')
"""

""" ---------------------- OVER SAMPLING ----------------------
# replace = True (duplicates data)
df_class_1_over = df_class_1.sample(count_class_0 , replace = True)
df_test_over = pd.concat([df_class_0 , df_class_1_over] , axis = 0)
print('Random Over-Sampling')
print(df_test_over.Churn.value_counts())

y = df_test_over['Churn']
X = df_test_over.drop('Churn' , axis = 'columns')
"""

""" ---------------------- Synthetic Minority Oversampling Technique (SMOTE) ----------------------
smote = SMOTE(sampling_strategy = 'minority')
X , y = smote.fit_resample(X , y)
"""

""" ---------------------- Ensemble With Under-Sampling ----------------------
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 5 , stratify = y)
df3 = X_train.copy()
df3['Churn'] = y_train

df3_class_0 = df3[df3.Churn == 0]
df3_class_1 = df3[df3.Churn == 1]

def get_train_batch(df_major , df_minor , start , end):
    # size of class0/3 is roughly the size of class1 (1495)
    df_train = pd.concat([df3_class_0[:1495] , df3_class_1] , axis = 0)

    y_train = df_train['Churn']
    X_train = df_train.drop('Churn' , axis = 'columns')

    return X_train , y_train

X_train , y_train = get_train_batch(df3_class_0 , df3_class_1 , 0 , 1495)

# RUN 3 MODELS (or until all data is used) AND TAKE THE MAJORITY PREDICTION
# IS 2 MODELS PREDICT 0 AND ONE PREDICTS 1 THE RESULT IS 0 (similar to kNN)
"""


# stratify = y (ensures the split is equal to the y split)
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 5 , stratify = y)


model = keras.Sequential([
    # Input layer should have equal number of neurons as columns/features
    l.Dense(20 , input_shape = (26,) , activation = 'relu'),
    l.Dense(128 , activation = 'relu'),
    l.Dense(1 , activation = 'sigmoid')
])

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train , y_train , epochs = 100)
model.evaluate(X_test , y_test)

# model.save('models/Imbalance_TCC1_TFsave')


# tf.save(path) + tf.keras.models.load_model(path) WORK SUPER WELL
# model = tf.keras.models.load_model('models/model_TFsave')

y_pred = model.predict(X_test)

collapsed_y_pred = []
for val in y_pred:
    if val < 0.5:
        collapsed_y_pred.append(0)
    else:
        collapsed_y_pred.append(1)

print(y_test[:10])
print(collapsed_y_pred[:10])

print(classification_report(y_test , collapsed_y_pred))
print(model.summary())
