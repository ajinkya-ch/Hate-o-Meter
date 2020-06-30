# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# This notebook is an attempt to explore the dataset. This notebook needs to be expanded upon.

# %%
import pandas as pd
import re
from textstat.textstat import textstat
from textblob import TextBlob
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("dark")
sns.set_context("talk")


# %%
df = pd.read_csv('../data/twitter-hate-speech.csv', encoding='latin-1')


# %%
df.head()


# %%
df.describe()


# %%
data_path = '../data/twitter-hate-speech.csv'

df = pd.read_csv(data_path, encoding='latin1')
df = df.rename(columns={'does_this_tweet_contain_hate_speech': 'label',  
                        'does_this_tweet_contain_hate_speech:confidence': 'confidence' })

mapping = {'The tweet is not offensive': 'Not offensive', 
           'The tweet uses offensive language but not hate speech': 'Offensive',
           'The tweet contains hate speech': 'Hate speech'
          }
df['label'] = df['label'].map(lambda x: mapping[x])


# %%
text = df['tweet_text']


# %%
text[:10]


# %%
def remove_handles(content):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)"," ",content).split())


# %%
text.apply(remove_handles)[:10]


# %%
data = df[~df['_golden']].dropna(axis=1)


# %%
sns.stripplot(x="label", y="confidence", data=data, size=6, jitter=True);


# %%
data['label'].value_counts()


# %%
data['confidence'].hist(bins=10);

