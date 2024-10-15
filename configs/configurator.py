import yaml

class Configurator:
    # themeclassifier:
    MODEL_NAME_CLASSIFY = "facebook/bart-large-mnli"

    # data:
    SUBTITLE_PATH =  "data/original/subtitlist"
    DIALOGE_PATH = "data/original/naruto.csv"
    JUTJU_PATH  = "data/original/jutsu.jsonl"
    SAVE_THEME_PATH = "data/save_data/theme_score.csv"
    SAVE_THEME_CLS_PATH = "data/save_data/theme_classification.csv"


CONFIGURATOR = Configurator()