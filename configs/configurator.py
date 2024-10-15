import yaml

class Configurator:
    # themeclassifier:
    MODEL_NAME_CLASSIFY = "facebook/bart-large-mnli"

    # data:
    SUBTITLE_PATH =  "data/original/subtitlist"
    DIALOGE_PATH = "data/original/naruto.csv"
    JUTJU_PATH  = "data/original/jutsu.jsonl"
    SAVE_THEME_PATH = "data/save_data/theme_score.csv"


CONFIGURATOR = Configurator()