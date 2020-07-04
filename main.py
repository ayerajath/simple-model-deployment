from fastapi import FastAPI
import uvicorn
import torch
from models import ReviewModel
import gc
from utils import load_weights, predict_text, predict_probs

app = FastAPI()
# model = load_model()

def load_model():
    model = ReviewModel()
    model = model.to(torch.device('cpu'))
    model = load_weights(model)
    return model

@app.get('/')
def get_root():
    return {'Hello' : 'World'}

@app.get('/predict/{text}')
def get_predictions(text: str):
    model = load_model()
    print("Collected Memorybefore prediction :", gc.collect())
    model.eval()
    label = predict_text(model, text, 0.5)
    print("Collected Memory after prediction :", gc.collect())
    return {'review' : text, 'predictions' : label} 