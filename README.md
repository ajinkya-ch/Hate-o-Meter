# Hate-o-Meter
## A CNN based model to detect hatespeech


### About

Our work aims to increase the awareness of the persistent hate speech in blogs, online-forums and newspapers. Our primary aim is to highlight content promoting violence or hatred against individuals or groups based on religion, gender, ethnicity/race or disability. We trained our model on preprocessed well defined data from different sources. Through our main algorithm we classify given text into hate or non-hate and show which category the statement(s) target. We achieve this using word2vec embeddings, 1D convolutional neural networks, and an integration of Flask API and Heroku platform. The final product is a tool which any person or organization could use for their purpose. 

### Dataset
We used Aitor Garcia’s data which had formal hate statements with zero twitter statements, Waseem’s data of tweets, and added a few newpaper headlines from relevant online sources.

### Our Tool
You can use the tool by clicking the link below.

[Hate-o-Meter](https://hate-o-meter.herokuapp.com/) 


### Future Scope:
Guidelines to consider: 
1)	Increasing the generalising ability of the model.
2)	Including functionality to capture long sentence dependencies.
3)	Working on building a more stable API, which allows multiple requests in a short span.


- If you found our work useful, please do consider citing it as follows:

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

Click [here](https://ieeexplore.ieee.org/document/9214247) to access the published research paper.
