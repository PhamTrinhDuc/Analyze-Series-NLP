import os
import glob
import nltk
import torch
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from utils.data_loader import load_subtiles_dataset
from configs.configurator import CONFIGURATOR


nltk.download("punkt")
nltk.download("punkt_tab")

class ThemeClassifier:
    def __int__(self, theme_list):
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.MODEL_NAME = CONFIGURATOR.MODEL_NAME
        self.theme_classifier = self.load_model()
    
    def load_model(self):
        """
        load model với task zero-shot-classification sử dụng pipeline từ thư viện transformers của huggingface
        """
        model = pipeline(
            task="zero-shot-classification",
            model=self.MODEL_NAME,
            device=self.device
        )
        return model
    
    