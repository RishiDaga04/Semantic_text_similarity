from functions import preprocess ,document_vector ,similarity
import pickle 
from fastapi import FastAPI 
import uvicorn
from pydantic import BaseModel
import json
import sklearn

app =FastAPI()

class model_input(BaseModel):

    text1 : str
    text2 : str

# loading the saved model

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get('/')
def index():
    return {"message" : "Hello World"}

@app.post('/sts')
def similarity_pred(input_parameters : model_input ):

    ip = input_parameters.model_dump()

    text1 = ip["text1"]
    text2 = ip["text2"]

    text1 = preprocess(text1)
    text2 = preprocess(text2)
    

    text1 = document_vector(text1 ,model)
    text2 = document_vector(text2 , model)

    x = text1.reshape(1, -1)
    y = text2.reshape(1, -1)
    score = sklearn.metrics.pairwise.cosine_similarity(x,y)
    similarity_score = round(score[0][0],  2)

    return {"similarity_score": str(similarity_score)}



if __name__ == '__main__':
    uvicorn.run("main:app " )


