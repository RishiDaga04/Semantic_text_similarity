# importing needed libraries
import numpy as np
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


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




def document_vector(doc , model):
    doc = [word for word in doc.split() if word in model.wv.index_to_key]
    return np.mean(model.wv[doc] , axis =0)

def similarity(text1 , text2):
    x = document_vector(preprocess(text1)).reshape(1, -1)
    y = document_vector(preprocess(text2)).reshape(1, -1)
    s = cosine_similarity(x,y)
    return s[0][0]


