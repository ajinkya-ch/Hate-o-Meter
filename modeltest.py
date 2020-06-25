# test predict
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
import numpy as np
import pandas as pd
import os
import collections
import re
import string
import itertools
from keras.models import load_model
import nltk 
import pickle
from newspaper import Article 
from nltk.tokenize import word_tokenize 
from nltk.stem import *

model=load_model('cnnmodel5.h5')
# loading

with open('tokenizer4.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


print("enter 1 for sentence check enter 2 for url 3 for article")
x=int(input())
if(x==1):
    print("enter text")
    text=input()
    a_list = nltk.tokenize.sent_tokenize(text)
    df=pd.DataFrame(a_list)

elif(x==2):
    url = "https://www.indiatoday.in/india/story/surrogate-mother-need-not-to-be-close-relative-single-woman-can-avail-surrogacy-parliamentary-panel-1643545-2020-02-05"  
    toi_article = Article(url, language="en")
    toi_article.download()
    toi_article.parse() 
    text=toi_article.title
    print(text)
    a_list2 = nltk.tokenize.sent_tokenize(text)
    df=pd.DataFrame(a_list2)

# df = pd.read_csv('testt.csv', header = None,encoding='latin1')
df.columns=['sent']

tokenizer.fit_on_texts(df['sent'].tolist())
test_sequences1 = tokenizer.texts_to_sequences(df['sent'].tolist())
print(test_sequences1)
test_cnn_data1 = pad_sequences(test_sequences1, maxlen=50,dtype='float32')
print(test_cnn_data1)

predictions = model.predict(test_cnn_data1,batch_size=30,verbose=1)
labels = [1, 0]
prediction_labels=[]
ones=0
count=0
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])
    count=count+1
    if(labels[np.argmax(p)]==1):
        ones=ones+1
print(prediction_labels) #labels all sentences either 1 or 0
print((ones/count)*100) #prints percentage of hate





words=word_tokenize(text)
stemmer = PorterStemmer()
plurals = words
all_tokens=[stemmer.stem(plural) for plural in plurals]
print(words)
# religion gender ethnicity disability
rlist=['hindu','hinduism','muslim','islam','islamic','sikh','christian','christianity','catholic','athiest','religion']
glist=['gender','heterosexual','homosexual','lesbian','bisexual','pansexual','asexual','queer','cisgender','transgender','transsexual','bigender','polygender','tran','LGBT','LGBTQ+','man','woman','men','women','gay']
elist=['race','ethnicity','whites','african','americans','multiracial','asian','arab','chinese','black','jew','jewish','hawaiians','indo','palestinian','egyptian','european','sudanese','jamaican','nigerian','indian','browns','blacks','punjabi','marathi','bengali','jihad','mujihads','syrian','iraqi','irani','hispanic','latin','latino']
dlist=['amnesia','amputee','anxiety','disorder','adhd','autism','syndrome','bipolar','blind','palsy','deaf','epilepsy','haemophilia','insomnia','mute','dyslexia','hiv','aid','schizophrenia','albino','tumour','dwarf','dwarfism','gigantism','parkinson','abnormal','retard','retarded','mental']
r=g=e=d=0
for token in all_tokens:
    if token in rlist:
        r=r+1
    elif token in glist:
        g=g+1
    elif token in elist:
        e=e+1
    elif token in dlist:
        d=d+1

print(r,g,e,d) #gives how many of each category are targetted , to make histogram
dict={r:"r",g:"g",e:"e",d:"d"}
print(dict.get(max(dict))) #gives which category is targetted
