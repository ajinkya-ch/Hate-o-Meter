# Hate-o-Meter
## A CNN based tool to detect hatespeech and its target bias.


### About

Our work aims to increase the awareness of the persistent hate speech in blogs, online-forums and newspapers. Our primary aim is to highlight content promoting violence or hatred against individuals or groups based on religion, gender, ethnicity/race or disability. We trained our model on preprocessed well defined data from different sources. Through our main algorithm we classify given text into hate or non-hate and show which category the statement(s) target. We are achieving this, using word2vec embeddings, convolutional neural networks, and an integration of Flask API and Heroku platform. The final product is a tool which any person or organization could use for their purpose. 

### Our Tool
You can use the tool by clicking the link below.

[Hate-o-Meter](https://hate-o-meter.herokuapp.com/) 


### Shortcomings of the current tool
Guidelines to consider: 
1)	Our current model cannot detect sarcastic hate statements. It is trained on the base of formal statements.
2)	The model does not comprehend sentences with less words.
3)	The API may crash or malfunction on receiving multiple requests in a row.



