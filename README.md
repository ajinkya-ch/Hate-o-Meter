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


- If you do use our tool, please do cite it as follows:

### Citation

@inproceedings{chaudhari2020cnn,
  title={CNN based Hate-o-Meter: A Hate Speech Detecting Tool},
  author={Chaudhari, Ajinkya and Parseja, Akshay and Patyal, Akshit},
  booktitle={2020 Third International Conference on Smart Systems and Inventive Technology (ICSSIT)},
  pages={940--944},
  year={2020},
  organization={IEEE}
}

### Published Work

Click [here]https://ieeexplore.ieee.org/document/9214247 to access the published research paper.
