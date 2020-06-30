# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# 

# %%
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("dark")
sns.set_context("talk")

# %% [markdown]
# 
# %% [markdown]
# 
# %% [markdown]
# 

# %%
path = 'maindatahnh.csv'
df = pd.read_csv(path,encoding='latin1')

# %% [markdown]
# 

# %%
df.head()

# %% [markdown]
# Here are the meanings of the column values, per the original authors:
# 
# `count` = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).
# 
# `hate_speech` = number of CF users who judged the tweet to be hate speech.
# 
# `offensive_language` = number of CF users who judged the tweet to be offensive.
# 
# `neither` = number of CF users who judged the tweet to be neither offensive nor non-offensive.
# 
# `class` = class label for majority of CF users. 0 - hate speech 1 - offensive language 2 - neither
# 
# `tweet` = the actual text of the tweet
# %% [markdown]
# Let's see how many of each class we have

# %%
plt.figure(figsize=(12, 8))
ax = sns.countplot(x="class", data=df)
plt.title('Distribution of Speech in Dataset')
plt.xlabel('') # Don't print "class"
plt.xticks(np.arange(3), ['Hate speech', 'Offensive language', 'Neither'])

# Print the number above each bar
for p in ax.patches:
    x = p.get_bbox().get_points()[:, 0]
    y = int(p.get_bbox().get_points()[1, 1])
    ax.annotate(y, (x.mean(), y), 
            ha='center', va='bottom')

# %% [markdown]
# Note the unusual distribution of types of text. Because all of the tweets were pulled from HateBase, hate speech and offensive language is going to be significantly over-represented, as we see above.
# %% [markdown]
# Let's see how much agreement there was. We'll check if any were called hate speech by at least one person and neither by at least one other.

# %%
hate_neither = df[(df['hate_speech'] != 0) & (df['neither'] != 0)]
hate_neither.sample(20)

# %% [markdown]
# There's a fair amount of disagreement, including some tweets that were marked by at least one person as being in every category.
# %% [markdown]
# This suggests that Bayes error will be relatively high for this dataset. It would be hard to get a good estimate of that, but we can see how many tweets were unanimously agreed upon. This should give a decent baseline for a classifier.

# %%
all_three = df[(df['hate_speech'] != 0) & (df['neither'] != 0) & (df['offensive_language'] != 0)]
hate_offensive = df[(df['hate_speech'] != 0) & (df['offensive_language'] != 0)]
offensive_neither = df[(df['neither'] != 0) & (df['offensive_language'] != 0)]


# %%
all_multiple = pd.concat([hate_neither, hate_offensive, offensive_neither]).drop_duplicates()


# %%
all_multiple.sample(20)

# %% [markdown]
# Let's look at how many were unanimous

# %%
disputed = len(all_multiple)/len(df)
print("{disputed:.1%} of the samples were disputed. {unanimous:.1%} were unanimous.".format(disputed=disputed, unanimous=1-disputed))

# %% [markdown]
# This gives us a good idea for how well we'll be able to do. Even a "great" classifier likely won't agree with the majority all of the time, as 29.5% of the samples had at least one dissenting opinion. This isn't quite Bayes error because it shows any disagreement, but it gives an idea that this won't be easy.
# %% [markdown]
# ### Tweets
# %% [markdown]
# Let's look in more detail at the tweets.

# %%
# Stop pandas from truncating text
pd.set_option('display.max_colwidth', -1)


# %%
df['tweet'].sample(20, random_state=0)

# %% [markdown]
# ## Cleaning the data
# %% [markdown]
# We could clean this out by removing URLs and mentions, but we'll leave hashtags in. Looks like there are also emojis, like this &#128557;, that have been replace with text strings like like this `&#128557;`. We'll remove those too. We'll come up with regexes to remove them.
# 
# It looks like there's also a ampersand issue. We'll fix that too.

# %%
url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
mention_regex = '@[\w\-]+'
emoji_regex = '&#\d*;'
amp_regex = '&amp;'


# %%
text = df['clean']
print(len(text))


# %%
def clean_text(text):
    text = re.sub(url_regex, 'URLHERE', str(text))
    text = re.sub(mention_regex, 'MENTIONHERE', str(text))
    text = re.sub(emoji_regex, ' EMOJIHERE ', str(text))
    text = re.sub(amp_regex, '&', str(text))
    return text


# %%
df['clean'] = df['clean'].apply(clean_text)


# %%
df['clean'].sample(20, random_state=0)

# %% [markdown]
# This looks much better (for hate speech). Now that we've cleaned it we'll save it and start generating some features for classification.

# %%
df.to_csv('cleanhate123.csv')


# %%


