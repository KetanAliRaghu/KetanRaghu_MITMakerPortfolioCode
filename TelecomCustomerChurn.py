"""
HANDLING PREPROCESSING AND RUNNING A MODEL WITH PROCESSED DATA
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow import keras
from keras import layers as l

df = pd.read_csv('data/Telco-Customer-Churn.csv')

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

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 5)

"""
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

model.save('models/model_TFsave')
"""

# tf.save(path) + tf.keras.models.load_model(path) WORK SUPER WELL
model = tf.keras.models.load_model('models/model_TFsave')

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
model.summary()


