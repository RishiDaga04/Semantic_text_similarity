# importing needed libraries
import pandas as pd
import numpy as np
import gensim
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Importing the csv to dataframe
df = pd.read_csv("DataNeuron_Text_Similarity.csv")

# new_df = df.copy()


# Preprocessing:

def preprocess(text):

    # remove punctuation from all the text
    
    pattern = r'[' + string.punctuation + ']'
    
    text = re.sub(pattern,'',text)
    
    # remove digits as it is not useful in capturing semantoc meaning 
    
    text = re.sub('\d.','',text)
    
    # lower case all the letters
    
    text = text.lower()
    
    # tokenize the text to apply functions on individual token
    
    text = word_tokenize(text)
    
    # remove stopwords but preserve the words like "no" ,"not" ,"nor" as it has the semantic meaning of the sentence
    
    stopword_list = stopwords.words("english")
    stopword_list.remove("no")
    stopword_list.remove("nor")
    stopword_list.remove("not")
    text = [word for word in text if word not in stopword_list]
    
    # lemetization to bring the words to its base form


    lm = WordNetLemmatizer()
    text =[ lm.lemmatize(word) for word in text]

    # returning the final text 
    
    return " ".join(text)


df["text1"] = df["text1"].apply(lambda x : preprocess(x))
df["text2"] = df["text2"].apply(lambda x : preprocess(x))
 


print("preprocessing done")

# Building the vocabulary for applying word2Vec model 
# It is a list which contains the list of all the tokens in the text
# Ex:[["broadband" ,"challenge" ,"tv"....] , ["rap", "bos", "arrested"] ,[].....]

vocab= []

for text in df["text1"]:
    vocab.append(text.split(" "))
    
        
for text in df["text2"]:
    vocab.append(text.split(" "))

# BUilding a word2vec model using gensim with windowsize of 10 and make a vector size of 200
model = gensim.models.Word2Vec(
    window=10,
    min_count=2,
    vector_size=200
)

model.build_vocab(vocab)

# Training the model 

model.train(vocab, total_examples=model.corpus_count, epochs=model.epochs)

print("model_trained")

# It converts individual word embedding of all word in a text to a single embedding of the text by using average 
def document_vector(doc):
    doc = [word for word in doc.split() if word in model.wv.index_to_key]
    return np.mean(model.wv[doc] , axis =0)

X=[]
Y =[]
for doc in tqdm(df["text1"].values):
    X.append(document_vector(doc))

for doc in tqdm(df["text2"].values):
    Y.append(document_vector(doc))

print("Converted to vectors")


# Calculate the similarity score between the two text embeddings  by cosine similarity
sim_score =[]
for i in tqdm(range(len(X))):
    Xi = X[i].reshape(1, -1)
    Yi = Y[i].reshape(1, -1)
    s = cosine_similarity(Xi ,Yi)
    sim_score.append(s[0][0])

# making a new column of similarity score which contains the similarity values rounded off to 2 digits

sim_score = [round(s,2) for s in sim_score]
df["similarity_score"] = (sim_score)

print(df.head())

# Exporting the final csv 
df.to_csv("texts_with_similarity_score.csv")




import pickle
# Making pkl file for deployment

pickle.dump(model , open("model.pkl" , "wb"))

# Final function in which if we input 2 texts it will give the simikarity score between that texts 
def similarity(text1 , text2):
    x = document_vector(preprocess(text1)).reshape(1, -1)
    y = document_vector(preprocess(text2)).reshape(1, -1)
    s = cosine_similarity(x,y)
    return s[0][0]