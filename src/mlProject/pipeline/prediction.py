# src/mlProject/pipeline/prediction.py
import joblib 
import numpy as np 
import pandas as pd
from pathlib import Path 

class PredictionPipeline:
    def __init__(self):

        # load the model 
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))


    def predict(self, data):
        return self.model.predict(data)