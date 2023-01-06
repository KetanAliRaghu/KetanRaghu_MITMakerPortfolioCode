import numpy as np
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Flatten , Embedding

reviews = [
    'nice food',
    'amazing restaurant',
    'too good',
    'just loved it!',
    'will go again',
    'horrible food',
    'never go there',
    'poor service',
    'poor quality',
    'needs improvement'
]

reviews_sentiment = np.array([1 , 1 , 1 , 1 , 1,
                              0 , 0 , 0 , 0 , 0])

# Encodes each word to a number between the vocab size [1 , VOCAB_SIZE]
VOCAB_SIZE = 50
encoded_reviews = [one_hot(x , VOCAB_SIZE) for x in reviews]
print(encoded_reviews)

max_length = 4
padded_reviews = pad_sequences(encoded_reviews , maxlen = max_length , padding = 'post')
print(padded_reviews)

embedded_vector_size = 4

model = Sequential([
    Embedding(VOCAB_SIZE,
              embedded_vector_size,
              input_length = max_length),
    Flatten(),
    Dense(1 , activation = 'sigmoid')
])

X = padded_reviews
y = reviews_sentiment

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = 'accuracy')
print(model.summary())

model.fit(X , y , epochs = 50)
loss , accuracy = model.evaluate(X , y)
print(loss , accuracy)
