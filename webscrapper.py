from newspaper import Article 
  
# #A new article from TOI 
# url = "https://www.indiatoday.in/india/story/surrogate-mother-need-not-to-be-close-relative-single-woman-can-avail-surrogacy-parliamentary-panel-1643545-2020-02-05"  
# #For different language newspaper refer above table 
# toi_article = Article(url, language="en") # en for English 
x=input()   
# url = "https://www.indiatoday.in/india/story/surrogate-mother-need-not-to-be-close-relative-single-woman-can-avail-surrogacy-parliamentary-panel-1643545-2020-02-05"  
url="x"
toi_article = Article(url, language="en")
sent2=toi_article.title
print(sent2)
#To download thearticle 
toi_article.download() 
  
#To parse the article 
toi_article.parse() 
  
#To perform natural language processing ie..nlp 
# toi_article.nlp() 
  
#To extract title 
# print("Article's Title:") 
# print(toi_article.title) 
# print("n") 
# toi_article.
# #To extract text 
# print("Article's Text:") 
# print(toi_article.text) 
# print("n") 
  
# #To extract summary 
# print("Article's Summary:") 
# print(toi_article.summary) 
# print("n") 
  
# #To extract keywords 
# print("Article's Keywords:") 
# print(toi_article.keywords) 