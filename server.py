# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:54:17 2020

@author: Cihan Ulas
"""

#%%
# Load Model
from joblib import dump, load
filename = "IrisKNNClassiferModel.joblib"
knn_classifier = load(filename)

# Get Model Feature and Label names.
from sklearn.datasets import load_iris
dataset = load_iris()
labels = dataset.target
labelNames = dataset.target_names

#%%

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import numpy as np

templates = Jinja2Templates(directory="templates")

app = FastAPI()

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

@app.get("/predict/")
async def make_prediction(request: Request, L1:float, W1:float, L2:float, W2:float ):
    test_data = np.array([L1, W1, L2, W2]).reshape(-1,4)
    probabilities = knn_classifier.predict_proba(test_data)[0]
    predicted = np.argmax(probabilities)
    probability = probabilities[predicted]
    return templates.TemplateResponse("predict.html", 
                                      {"request": request,
                                       "probabilities": probabilities,
                                       "predicted": labelNames[predicted],
                                       "probability": probability})
