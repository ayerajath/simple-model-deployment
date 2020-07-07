from fastapi import FastAPI
import uvicorn
import torch
from models import ReviewModel
import gc
from utils import load_weights, predict_text, predict_probs
from requests import TextClassificationRequest

app = FastAPI()
# model = load_model()

def load_model():
    model = ReviewModel()
    model = model.to(torch.device('cpu'))
    model = load_weights(model)
    return model

model = load_model()
model.eval()

@app.get('/predict/{text}')
def get_predictions(text: str):
    print("Collected Memory before prediction :", gc.collect())
    label, prob = predict_text(model, text, 0.5)
    prob = round(prob, 3)
    print("Collected Memory after prediction :", gc.collect())
    return {'review' : text, 'prediction' : label, 'probability' : prob}

@app.post('/postpredict')
def post_predictions(req: TextClassificationRequest):
    print("Collected Memory before prediction :", gc.collect())
    label, prob = predict_text(model, req.text, req.threshold)
    prob = round(prob, 3)
    print("Collected Memory after prediction :", gc.collect())
    return {'review' : req.text, 'prediction' : label, 'probability' : prob}


# Some reviews
# It's watchable and intriguing stuff, yet also silly and inconsistent. - Positive
# It's just not special enough. - Negative