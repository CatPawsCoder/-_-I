from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str

app = FastAPI()
classifier = pipeline("sentiment-analysis")

@app.get ("/")
async def root ():
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(item: Item):
     """Sentiment analysis for a text"""
     return classifier(item.text )[0]


# @app.get("/predict/")
# def predict ():
#     return classifier("I don't like machine learning engineering!")
