

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
import collections
import yaml
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
train = pd.read_pickle('objects/sentiment.pkl')
print(f'Training data: {train.head()}')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
X_train, X_val, y_train, y_val = train_test_split(train['Processed Inbound'], train['Real Outbound'], test_size = 0.3, 
                                                   shuffle = True, stratify = train['Real Outbound'], random_state = 7)
print(f'\nShape checks:\nX_train: {X_train.shape} X_val: {X_val.shape}\ny_train: {y_train.shape} y_val: {y_val.shape}')

le = LabelEncoder()
le.fit(y_train)

y_train = le.transform(y_train)
y_val = le.transform(y_val)



t = Tokenizer()
t.fit_on_texts(X_train)

print("Document Count: \n{}\n".format(t.document_count))
def convert_to_padded(tokenizer, docs):

    embedded = t.texts_to_sequences(docs)

    padded = pad_sequences(embedded, maxlen = max_length, padding = 'post')
    return padded


vocab_size = len(t.word_counts) + 1
print(f'Vocab size:\n{vocab_size}')


max_length = len(max(X_train, key = len))

print(f'Max length:\n{max_length}')

padded_X_train = convert_to_padded(tokenizer = t, docs = X_train)
padded_X_val = convert_to_padded(tokenizer = t, docs = X_val)

print(f'padded_X_train\n{padded_X_train}')
print(f'padded_X_val\n{padded_X_val}')

embeddings_index = {}
f = open('objects/glove.twitter.27B.50d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
# Initializing required objects
word_index = t.word_index
EMBEDDING_DIM = 50 


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
import keras.backend as K
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
def make_model(vocab_size, max_token_length):

    model = Sequential()

    model.add(Embedding(vocab_size, embedding_matrix.shape[1], input_length = 50, 
                        trainable = False, weights = [embedding_matrix]))
    
    model.add(Bidirectional(LSTM(128)))

    model.add(Dense(600, activation = "relu",kernel_regularizer ='l2'))
    

    model.add(Dense(600, activation = "relu",kernel_regularizer ='l2'))
    # model.add(Dense(600, activation = "relu",kernel_regularizer ='l2'))
    model.add(Dense(300, activation = "relu",kernel_regularizer ='l2'))
    # model.add(Dense(300, activation = "relu",kernel_regularizer ='l2'))
    model.add(Dense(150, activation = "relu",kernel_regularizer ='l2'))
    # model.add(Dense(150, activation = "relu",kernel_regularizer ='l2'))
    model.add(Dense(60, activation = "relu",kernel_regularizer ='l2'))

    model.add(Dropout(0.5))
    

    model.add(Dense(50, activation = "softmax"))
    
    return model


model = make_model(vocab_size, 50)
model.compile(loss = "sparse_categorical_crossentropy", 
              optimizer = "adam", metrics = ["accuracy"])
model.summary()

filename = 'models/sentiment_classification.h5'

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_sched_checkpoint = tf.keras.callbacks.LearningRateScheduler(scheduler)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)



checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min')




hist = model.fit(padded_X_train, y_train, epochs = 20, batch_size = 32, 
                 validation_data = (padded_X_val, y_val), 
                 callbacks = [checkpoint, lr_sched_checkpoint, early_stopping])

plt.figure(figsize=(10,7))
plt.plot(hist.history['val_loss'], label = 'Validation Loss', color = 'cyan')
plt.plot(hist.history['loss'], label = 'Training Loss', color = 'purple')
plt.title('Training Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.figure(figsize=(10,7))
plt.plot(hist.history['val_accuracy'], label = 'Validation Accuracy', color = 'cyan')
plt.plot(hist.history['accuracy'], label = 'Training Accuracy', color = 'purple')
plt.title('Training Accuracy vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
