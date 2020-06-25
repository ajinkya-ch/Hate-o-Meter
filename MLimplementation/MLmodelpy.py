#!/usr/bin/env python
# coding: utf-8

# ### Text Classification with ScikitLearn,SpaCy and Interpretation of ML Model with ELi5
#    + Text Preprocessing with SpaCy
#    + Classifying Text With Sklearn
#    + Interpreting Model with Eli5
#     

# In[389]:


# load EDA Pkgs
import pandas as pd
import numpy as np
# Load libraries
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import KFold
import pandas
from sklearn.model_selection import train_test_split
import numpy
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


# In[390]:


# Load NLP pkgs
import spacy


# In[391]:


from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')


# In[392]:


# Use the punctuations of string module
import string
punctuations = string.punctuation


# In[393]:


# Creating a Spacy Parser
from spacy.lang.en import English
parser = English()


# In[249]:


# Build a list of stopwords to use to filter
stopwords = list(STOP_WORDS)


# In[250]:


print(stopwords)


# In[251]:


def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    return mytokens


# In[252]:


ex1 = "He was walking with the walker in the Wall he may had sat and run with the runner"


# In[511]:


spacy_tokenizer(ex1)


# In[394]:


# Load ML Pkgs
# ML Packages
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score 
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


# In[395]:


# Load Interpretation Pkgs
import eli5


# In[396]:


# Load dataset
df = pd.read_csv("newhateegs.csv")


# In[397]:


df.head()


# In[398]:


df.shape


# In[430]:


models = []
models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
results = []
names=[]


# In[431]:


df.columns


# In[432]:


#Custom transformer using spaCy 
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}


# In[433]:


# Basic function to clean the text 
def clean_text(text):     
    return text.strip().lower()


# In[434]:


# Vectorization
vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1)) 
# classifier = LinearSVC()
classifier = SVC(C=10, gamma=0.01, probability=True)


# In[435]:


# Using Tfidf
tfvectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer)


# In[436]:


# Splitting Data Set
from sklearn.model_selection import train_test_split


# In[437]:


# Features and Labels
X = df['Message']
ylabels = df['Target']


# In[438]:


X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3, random_state=42)


# In[454]:


X_train

vectorizer_1 = CountVectorizer()
X_1 = vectorizer_1.fit_transform(X_train)
X_2=vectorizer_1.fit_transform(X_test)


# In[447]:


X_train.shape


# In[466]:


# Create the  pipeline to clean, tokenize, vectorize, and classify 
for name, model in models:
    pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', vectorizer),
                 ('classifier',model)])
    pipe.fit(X_train,y_train)  
    print("Accuracy Score:",pipe.score(X_test, y_test),model)
# #     kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=5)
# #     cv_results = model_selection.cross_val_score(pipe, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(pipe.score(X_test, y_test))
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)
#     print(msg)


# In[449]:


# Fit our data
pipe.fit(X_train,y_train)


# In[485]:


clf = SVC(C=10,gamma=0.1)
pipesvc = Pipeline([("cleaner", predictors()),
                 ('vectorizer', vectorizer),
                 ('classifier',clf)])
pipesvc.fit(X_train,y_train)
svmpredicted=pipesvc.predict(X_test)

print('Training Score:',pipesvc.score(X_train, y_train))
print('Testing Score:',pipesvc.score(X_test, y_test))
print('Accuracy:',accuracy_score(y_test,svmpredicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,svmpredicted))


# In[471]:


sns.heatmap(confusion_matrix(y_test,svmpredicted),annot=True,cmap='YlGn')
plt.title("SVM ")
plt.ylabel("True LABEL")
plt.xlabel("PREDICTED LABEL")
plt.show()


# In[487]:


kkk=KNeighborsClassifier()
pipeneighnour = Pipeline([("cleaner", predictors()),
                 ('vectorizer', vectorizer),
                 ('classifier',kkk)])
pipeneighnour.fit(X_train,y_train)
# kkk.fit(X_train,Y_train)
p=pipeneighnour.predict(X_test)
print('Training Score:',pipeneighnour.score(X_train, y_train))
print('Testing Score:',pipeneighnour.score(X_test, y_test))
print('Accuracy:',accuracy_score(y_test,p))
print('Confusion Matrix: \n', confusion_matrix(y_test,p))


# In[488]:


sns.heatmap(confusion_matrix(y_test,p),annot=True,cmap='YlGn')
plt.title("KNN")
plt.ylabel("True LABEL")
plt.xlabel("PREDICTED LABEL")
plt.show()


# In[489]:


logistic=LogisticRegression()
pipelog = Pipeline([("cleaner", predictors()),
                 ('vectorizer', vectorizer),
                 ('classifier',logistic)])
pipelog.fit(X_train,y_train)
# kkk.fit(X_train,Y_train)
# p=pipe.predict(X_test)
logispredicted=pipelog.predict(X_test)
print('Training Score:',pipelog.score(X_train, y_train))
print('Testing Score:',pipelog.score(X_test, y_test))
print('Accuracy:',accuracy_score(y_test,logispredicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,logispredicted))


# In[490]:


sns.heatmap(confusion_matrix(y_test,logispredicted),annot=True,cmap='YlGn')
plt.title("LOGISTIC REG")
plt.ylabel("True LABEL")
plt.xlabel("PREDICTED LABEL")
plt.show()


# In[492]:


randomforest = RandomForestClassifier(n_estimators=100)
# randomforest.fit(X_train, y_train)
#Predict Output
piperandom = Pipeline([("cleaner", predictors()),
                 ('vectorizer', vectorizer),
                 ('classifier',randomforest)])
piperandom.fit(X_train,y_train)
predicted = piperandom.predict(X_test)

print('Training Score:',piperandom.score(X_train, y_train))
print('Testing Score:',piperandom.score(X_test, y_test))
print('Accuracy:',accuracy_score(y_test,predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,predicted))


# In[493]:


sns.heatmap(confusion_matrix(y_test,predicted),annot=True,cmap='YlGn')
plt.title("RANDOM FOREST ")
plt.ylabel("True LABEL")
plt.xlabel("PREDICTED LABEL")
plt.show()


# In[494]:


models = pd.DataFrame({
    'Model': [ 'Logistic Regression', 'SVM','Random Forest','KNN'],
    'Score': [ pipelog.score(X_train, y_train), pipesvc.score(X_train, y_train), piperandom.score(X_train, y_train),pipeneighnour.score(X_train, y_train)],
    'Test Score': [ pipelog.score(X_test, y_test), pipesvc.score(X_test, y_test), piperandom.score(X_test, y_test),pipeneighnour.score(X_test, y_test)]})
models.sort_values(by='Test Score', ascending=False)


# In[347]:


X_test.shape


# In[ ]:





# In[348]:


X_test.values[1]


# In[495]:


# Predicting with a test dataset

pipelog.predict(X_test)


# In[ ]:





# In[497]:


print("Accuracy Score:",pipelog.score(X_test, y_test))


# In[498]:


# Prediction Results
# 1 = Positive review
# 0 = Negative review
for (sample,pred) in zip(X_test,sample_prediction):
    print(sample,"Prediction=>",pred)


# ### Interpreting Our Model
# + Eli5
# + Data
# + Model
# + Target Names
# + Function

# In[499]:


from eli5.lime import TextExplainer


# In[500]:


pipe.predict_proba


# In[501]:


exp = TextExplainer(random_state=42)


# In[502]:


X_test.values[0]


# In[515]:


a=pipelog.predict([input()])
if  a==1:
    print("hate statement")
elif a == 0:
    print("Not hate bro!")


# In[374]:


exp.fit(X_test.values[0], pipe.predict_proba)


# In[378]:


ylabels.unique()


# In[ ]:





# In[379]:


target_names = ['Negative','Positive']


# In[380]:


exp.show_prediction()


# In[381]:


exp.show_prediction(target_names=target_names)


# In[382]:


exp.metrics_


# - ‘score’ is an accuracy score weighted by cosine distance between generated sample and the original document (i.e. texts which are closer to the example are more important). Accuracy shows how good are ‘top 1’ predictions.
# - ‘mean_KL_divergence’ is a mean Kullback–Leibler divergence for all target classes; it is also weighted by distance. KL divergence shows how well are probabilities approximated; 0.0 means a perfect match.

# In[46]:


exp.show_weights()


# In[47]:


# Check For Vectorizer Used
exp.vec_


# In[48]:


# Check For Classifer Used
exp.clf_


# In[49]:





# In[ ]:




