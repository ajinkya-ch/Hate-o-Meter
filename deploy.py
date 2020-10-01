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

import sys
from keras.models import load_model
from flask import Flask, request
from keras.models import load_model
import pickle
from newspaper import Article 
from flask import Flask, render_template, Response
from spacy.lang.en import English
 
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))	



with open('tokenizer7.pickle', 'rb') as handle:
	tokenizer= pickle.load(handle)

app = Flask(__name__)


RELIGION_named_entity=[]
@app.route('/')
def index():

	return render_template("index.html")
@app.route('/about')
def about():
	return render_template("about.html")

@app.route('/contact')
def contact():
	return render_template("contact.html")
	
@app.route('/process', methods = ['GET', 'POST', 'PATCH', 'PUT', 'DELETE'])
def process():

	df=0
	global model
	model = load_model('cnnmodel7.h5')




	print(request.method)
	

	if request.method == 'POST':
		choice = request.form['type']
		
		d = []
		if(choice=='rawtext'):
			rawtext = request.form['rawtext']
			rawtext=rawtext.strip()
			rawtext1=rawtext
			doc = nlp(rawtext)
			rawtext = [sent.string.strip() for sent in doc.sents]
			a_list = rawtext
			df=pd.DataFrame(a_list)
		if(choice=='url'):	
			url = request.form['urltext']
			print(url)
			print('hello')
			url=url.strip()
			toi_article = Article(url, language="en") # en for English 
			toi_article.download()  
			toi_article.parse() 
			rawtext = toi_article.text
			rawtext=rawtext.strip()
			rawtext1=rawtext
			doc = nlp(rawtext)
			rawtext = [sent.string.strip() for sent in doc.sents]
			a_list = rawtext	
			
			df=pd.DataFrame(a_list)
		token_select = [token.text.lower() for token in doc]
	
		df.columns=['sent']
		print(df['sent'])
		tokenizer.fit_on_texts(df['sent'].tolist())
		test_sequences1 = tokenizer.texts_to_sequences(df['sent'].tolist())
# print(test_sequences1)
		test_cnn_data1 = pad_sequences(test_sequences1, maxlen=50,dtype='float32')
# print(test_cnn_data1)
		predictions = model.predict(test_cnn_data1, batch_size=20, verbose=1)
		print(predictions)
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
		print((ones/count)*100,count,ones)
		words = [token.text.lower() for token in doc]
		print (words)
		

# religion gender ethnicity disability
		rlist=['hindu','hindus','hindu\'s','hinduism','muslim','muslims','muslim\'s','islam','islamophobia','islamic','sikh','christian','christianity','catholic','athiest','religion']
		glist=['gender','heterosexual','homosexual','lesbian','bisexual','pansexual','asexual','queer','cisgender','transgender','transsexual','bigender','polygender','tran','LGBT','LGBTQ+','man','woman','men','women','gay','genders','heterosexuals','homosexuals','lesbians','bisexuals','pansexuals','asexuals','queers','cisgenders','transgenders','transsexuals','bigenders','polygenders','trans','LGBTs','LGBTQ+s','man','woman','men','women','gays','gay\'s','men\'s','women\'s']
		elist=['race','ethnicity','whites','african','americans','multiracial','asian','arab','chinese','black','jew','jewish','hawaiians','indo','palestinian','egyptian','european','sudanese','jamaican','nigerian','indian','browns','blacks','punjabi','marathi','bengali','jihad','mujihads','syrian','iraqi','irani','hispanic','latin','latino',
		'races','ethnicities','white','africans','american','multiracials','asians','arabs','chinese','blacks','jews','jewishs','hawaiian','indos','palestinians','egyptians','europeans','sudaneses','jamaicans','nigerians','indians','brownss','blackss','punjabis','marathis','bengalis','jihads','mujihads','syrians','iraqis','iranis','hispanics','latins','latinos']
		dlist=['amnesia','amputee','anxiety','disorder','adhd','autism','syndrome','bipolar','blind','palsy','deaf','epilepsy','haemophilia','insomnia','mute','dyslexia','hiv','aid','schizophrenia','albino','tumour','dwarf','dwarfism','gigantism','parkinson','abnormal','retard','retarded','mental','amnesia','amputee','anxieties','anxietic','disorders','adhds','autisms','syndromes','bipolars','blinds','palsys','deafs','epilepsy','haemophiliac','insomniac','mute','dyslexias','hiv','aids','schizophrenias','albinos','tumours','dwarfs','dwarfisms','gigantisms','parkinsons','abnormals','retards','retarded','mentals']
		r=g=e=d=0
		for token in words:
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
	
		values=[r,g,e,d]

	
	return render_template("index.html",results=cat,text=rawtext,prediction=percentage,num_of_results = (r+g+e+d),religion=r,eth=e,dis=d,gender=g)


if __name__ == '__main__':
	app.run(host='127.0.0.1',debug=True)
