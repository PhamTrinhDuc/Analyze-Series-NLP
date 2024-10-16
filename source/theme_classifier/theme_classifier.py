import os
import glob
import nltk
import torch
import logging
import numpy as np
import pandas as pd
from typing import List, Dict
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from utils.data_loader import load_subtiles_dataset
from configs.configurator import CONFIGURATOR


nltk.download("punkt")
nltk.download("punkt_tab")

class ThemeClassifier:
    def __init__(self, theme_list):
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.MODEL_NAME = CONFIGURATOR.MODEL_NAME_CLASSIFY
        self.theme_classifier = self.load_model()
    
    def load_model(self):
        """
        load model với task zero-shot-classification sử dụng pipeline từ thư viện transformers của huggingface
        """
        try:
            model = pipeline(
                task="zero-shot-classification",
                model=self.MODEL_NAME,
                device=self.device
            )
        except Exception as e:
            logging.error(f"Error when loading model [LOAD_MODEL]: {str(e)}")
        return model
    
    def classify_theme(self, scripts) -> Dict:
        """
        Join từng tập thành các batch và đưa vào model phân loại. Với mỗi label ta sẽ lấy mean score của tất cả các batch
        Args:
            scripts: list các đoạn script cần phân loại chủ đề
        Returns:
            themes_scores: dictionary chứa các chủ đề và điểm số tương ứng
        """
        themes_scores = {}
        try:
            scripts_sentences = sent_tokenize(scripts)

            sentences_batch_size = 20
            scripts_batches = []
            for idx in range(0, len(scripts_sentences), sentences_batch_size):
                sent = " ".join(scripts_sentences[idx:idx + sentences_batch_size])
                scripts_batches.append(sent)
            
            theme_output = self.theme_classifier(
                scripts_batches,
                self.theme_list,
                mutil_label=True
            )

            for output in theme_output:
                for label, score in zip(output['labels'], output['scores']):
                    if label not in themes_scores:
                        themes_scores[label] = []
                    themes_scores[label].append(score)
            
            themes_scores = {key: np.mean(np.array(value)) for key, value in themes_scores.items()}
        except Exception as e:
            logging.error(f"Error when classifying theme [CLASSIFY_THEME]: {str(e)}")
        return themes_scores
    
    def get_themes(self) -> pd.DataFrame:
        if os.path.exists(CONFIGURATOR.SAVE_THEME_PATH):
            df = pd.read_csv(CONFIGURATOR.SAVE_THEME_PATH)
            return df
        
        df = load_subtiles_dataset()
        theme_scores = df['scripts'].apply(self.classify_theme)
        theme_scores_df = pd.DataFrame(theme_scores.tolist())
        df[theme_scores_df.columns] = theme_scores_df

        # Sava path
        os.makedirs(os.path.isdir(CONFIGURATOR.save_theme_path), exist_ok=True)
        df.to_csv(CONFIGURATOR.save_theme_path, index=False)

        return df