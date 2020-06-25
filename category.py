from spacy.lang.en import English

raw_text = 'Hello world?Here are two sentences.'
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))
doc = nlp(raw_text)
sentences = [sent.string.strip() for sent in doc.sents]
print(sentences)

for token in doc:
            print(token.text)
            
lowercase= raw_text.lower()
print(lowercase)


# text = "Hello everyone muslims and hindus are gay friends with aids. Welcome to GeeksforGeeks."
# words=word_tokenize(text)
# stemmer = PorterStemmer()
# plurals = words
# all_tokens=[stemmer.stem(plural) for plural in plurals]

# # religion gender ethnicity disability
# rlist=['hindu',hinduism','muslim','islam','islamic','sikh','christian','christianity','catholic','athiest','religion']
# glist=['gender','heterosexual','homosexual','lesbian','bisexual','pansexual','asexual','queer','cisgender','transgender','transsexual','bigender','polygender','tran','LGBT','LGBTQ+','man','woman','men','women','gay']
# elist=['race','ethnicity','whites','african','americans','multiracial','asian','arab','chinese','black','jew','jewish','hawaiians','indo','palestinian','egyptian','european','sudanese','jamaican','nigerian','indian','browns','blacks','punjabi','marathi','bengali','jihad','mujihads','syrian','iraqi','irani','hispanic','latin','latino']
# dlist=['amnesia','amputee','anxiety','disorder','adhd','autism','syndrome','bipolar','blind','palsy','deaf','epilepsy','haemophilia','insomnia','mute','dyslexia','hiv','aid','schizophrenia','albino','tumour','dwarf','dwarfism','gigantism','parkinson','abnormal','retard','retarded','mental']
# r=g=e=d=0
# for token in all_tokens:
#     if token in rlist:
#         r=r+1
#     elif token in glist:
#         g=g+1
#     elif token in elist:
#         e=e+1
#     elif token in dlist:
#         d=d+1

# print(r,g,e,d)
# dict={r:"r",g:"g",e:"e",d:"d"}
# print(dict.get(max(dict)))