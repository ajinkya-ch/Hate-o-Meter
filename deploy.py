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
from flask import Flask,render_template,url_for,request
import os
import collections
import re
import string
import itertools
from keras.models import load_model
from flask import Flask, request
from keras.models import load_model
import nltk 
import pickle
from newspaper import Article 
from nltk.tokenize import word_tokenize 
from nltk.stem import *

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


app = Flask(__name__)
RELIGION_named_entity=[]
@app.route('/')
def index():
	return render_template("index.html")

@app.route('/process', methods = ['GET', 'POST', 'PATCH', 'PUT', 'DELETE'])
def process():
	df=0
	global models
	model = load_model('cnnmodelnew.h5')



	print(request.method)

	if request.method == 'POST':
		choice = request.form['type']
		
		d = []
		if(choice=='rawtext'):
			rawtext = request.form['rawtext']
			a_list = nltk.tokenize.sent_tokenize(rawtext)
			df=pd.DataFrame(a_list)
		if(choice=='url'):	
			url = request.form['urltext']
			print(url)
			print('hello')
			url=url.strip()
			toi_article = Article(url, language="en") # en for English 
			toi_article.download()  
			toi_article.parse() 
			rawtext = toi_article.title
			a_list = nltk.tokenize.sent_tokenize(rawtext)
			df=pd.DataFrame(a_list)
		df.columns=['sent']
		print(df['sent'])
		tokenizer.fit_on_texts(df['sent'].tolist())
		test_sequences1 = tokenizer.texts_to_sequences(df['sent'].tolist())
# print(test_sequences1)
		test_cnn_data1 = pad_sequences(test_sequences1, maxlen=50,dtype='float32')
# print(test_cnn_data1)
		predictions = model.predict(test_cnn_data1, batch_size=20, verbose=1)
		labels = [1, 0]
		prediction_labels=[]
		ones=0
		count=0
		for p in predictions:
			prediction_labels.append(labels[np.argmax(p)])
			count=count+1
			if(prediction_labels==[1]):
				ones=ones+1
		print(prediction_labels) #labels all sentences either 1 or 0
		print((ones/count)*100,count,ones)
		words=word_tokenize(rawtext)
		stemmer = PorterStemmer()
		plurals = words
		all_tokens=[stemmer.stem(plural) for plural in plurals]

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
		cat=dict.get(max(dict))
		percentage=(ones/count)*100
		ones=0
		count=0
		prediction_labels=[]

	
	return render_template("index.html",results=cat,text=rawtext,prediction=percentage,num_of_results = (r+g+e+d))


if __name__ == '__main__':
	app.run(host='0.0.0.0',debug=True)
