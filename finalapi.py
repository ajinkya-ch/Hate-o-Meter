from __future__ import unicode_literals,print_function
from flask import Flask,render_template,url_for,request
import re
import pandas as pd
import spacy
from spacy import displacy
import en_core_web_sm

import spacy
import random
import plac
import random
from pathlib import Path
import spacy
import os
from generate_data import word_to_array
from keras.models import load_model
from flask import Flask, request
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import Sequential
import keras

output_dir=Path("C:\\MyDrive\\Projects\\EDD3PROJECT\\sem4\\NER")

print("Loading from", output_dir)
nlp = spacy.load(output_dir)

app = Flask(__name__)
RELIGION_named_entity=[]
@app.route('/')
def index():
	return render_template("index.html")

@app.route('/process', methods = ['GET', 'POST', 'PATCH', 'PUT', 'DELETE'])
def process():
	df=0
	global models
	models = {}
	models['RNN'] = load_model('xor_model_1')
	global graph
	graph = tf.get_default_graph()
	RELIGION_named_entity=[]
	print(request.method)
	tokenizer = Tokenizer(num_words=4000)

	if request.method == 'POST':
		choice = request.form['taskoption']
		rawtext = request.form['rawtext']
		doc = nlp(rawtext)
		d = []
		
		#toi_article = Article(url, language="en") # en for English 
		#toi_article.download() 
		#To parse the article 
		#toi_article.parse() 
		print(doc)
		for ent in doc.ents:
			d.append((ent.label_, ent.text))
			df = pd.DataFrame(d, columns=('named entity', 'output'))
		
			RELIGION_named_entity=df.loc[df['named entity']=='religion']['named entity']
		
		results = RELIGION_named_entity
		num_of_results = len(results)
		model_name = request.form.get('model')
		model = models[model_name]
		tokenizer.fit_on_texts(rawtext)
		index_list = tokenizer.texts_to_sequences(rawtext)
		arr =  keras.preprocessing.sequence.pad_sequences(index_list, maxlen=30)
		with graph.as_default():
			prediction = model.predict(arr)
			print(prediction[0][0])
	
	return render_template("index.html",results=results,text=rawtext,model_name=model_name,prediction=(prediction[0][0]),num_of_results = num_of_results)


if __name__ == '__main__':
	app.run(debug=True)
