from __future__ import division, print_function
from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
import os
import collections
import re
import string
import itertools

from keras.layers import LeakyReLU

# print(df['Sent'])


# 1
data = pd.read_csv('maindataz1.csv', header = None,encoding='latin1')
data.columns = ['Text', 'Label']
data.Text=data.Text.astype(str)
# print(data.head())
# print(data.Label.value_counts())

# 2
pos = []
neg = []
for l in data.Label:
    if l == 0:
        pos.append(0)
        neg.append(1)
    elif l == 1:
        pos.append(1)
        neg.append(0)
data['Pos']= pos
data['Neg']= neg


# data cleaning:
def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct
data['Text_Clean'] = data['Text'].apply(lambda x: remove_punct(x))

import nltk
from nltk import word_tokenize, WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('stopwords')
tokens = [word_tokenize(sen) for sen in data.Text_Clean]

def lower_token(tokens): 
    return [w.lower() for w in tokens]        
lower_tokens = [lower_token(token) for token in tokens]

from nltk.corpus import stopwords
stoplist = stopwords.words('english')

def removeStopWords(tokens): 
    return [word for word in tokens if word not in stoplist]
filtered_words = [removeStopWords(sen) for sen in lower_tokens]
data['Text_Final'] = [' '.join(sen) for sen in filtered_words]
data['tokens'] = filtered_words
data = data[['Text_Final', 'tokens', 'Label', 'Pos', 'Neg']]
# print(data)




# splitting data
data_train, data_test = train_test_split(data, test_size=0.20, random_state=42)



all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
# print("words total, with a vocabulary size of "(len(all_training_words), len(TRAINING_VOCAB)))
# print(max(training_sentence_lengths))

all_test_words = [word for tokens in data_test["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data_test["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))
# print( "words total, with a vocabulary size of"(len(all_test_words), len(TEST_VOCAB)))
# print(max(test_sentence_lengths))
 

word2vec_path = 'C:\MyDrive\Projects\EDD3PROJECT\sem4\GoogleNews-vectors-negative300.bin.gz'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)

training_embeddings = get_word2vec_embeddings(word2vec, data_train, generate_missing=True)
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300


tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(data_train["Text_Final"].tolist())
training_sequences = tokenizer.texts_to_sequences(data_train["Text_Final"].tolist())

train_word_index = tokenizer.word_index
print('Found %s unique tokens.'%len(train_word_index))

train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)
train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
for word,index in train_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(train_embedding_weights.shape)
test_sequences = tokenizer.texts_to_sequences(data_test["Text_Final"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
# print(test_cnn_data)

# CNN

def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=False)
    
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [2,3,4,5,6]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=filter_size,activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)


    l_merge = concatenate(convs, axis=1)

    x = Dropout(0.1)(l_merge)  
    x = Dense(64)(x) #removed activation='relu'
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model

label_names = ['Pos', 'Neg']
y_train = data_train[label_names].values
x_train = train_cnn_data
y_tr = y_train
model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                len(list(label_names)))

num_epochs = 7
batch_size = 150
hist = model.fit(x_train, y_tr, epochs=num_epochs, validation_split=0.25, shuffle=True, batch_size=batch_size)



predictions = model.predict(test_cnn_data, batch_size=30, verbose=1)
labels = [1, 0]
prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])
print(prediction_labels)
print(sum(data_test.Label==prediction_labels)/len(prediction_labels))

# score, acc = model.evaluate(x=test_cnn_data,y=None,batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)


# saving
with open('tokenizer5.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
model.save('cnnmodel5.h5')



