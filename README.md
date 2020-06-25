Our work aims to increase the awareness of the persistent hate speech in blogs, online-forums and newspapers. Our primary aim is to highlight content promoting violence or hatred against individuals or groups based on religion, gender, ethnicity/race or disability. We trained our model on preprocessed well defined data from different sources. Through our main algorithm we classify given text into hate or non-hate and show which category the statement(s) target. We are achieving this, using word2vec embeddings, convolutional neural networks, and an integration of Flask API and Heroku platform. The final product is a tool which any person or organization could use for their purpose. 

Our product: https://hate-o-meter.herokuapp.com/

Guidelines to consider: 
1)	Our current model cannot detect sarcastic hate statements. It is trained on the base of formal statements.
2)	The model does not comprehend sentences with less words.
3)	The API may crash or malfunction on receiving multiple requests in a row.

this link is for .bin.gz file download the zip and directly put in folder: https://drive.google.com/open?id=1mSYneErVe1l8v08rNrbe95qtaVpJdlmc
this link has the .h5 CNN model: https://drive.google.com/file/d/1MXGz-dnxq9Sj_VyIRZ869GrQ0BuPGqOx/view?usp=sharing


The project is open for further developments.
