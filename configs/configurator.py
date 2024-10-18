import yaml

class Configurator:
    # themeclassifier:
    MODEL_NAME_THEME = "facebook/bart-large-mnli"
    SAVE_THEME_PATH = "data/save_data/theme_score.csv"
    SAVE_THEME_SCRIPT_PATH = "data/save_data/scripts_episodes.csv"

    # jutsu classifier:
    MODEL_NAME_JUTSU = "distilbert/distilbert-base-uncased"

    # data:
    SUBTITLE_PATH =  "data/original/subtitlist"
    DIALOGE_PATH = "data/original/naruto.csv"
    JUTJU_PATH  = "data/original/jutsu.jsonl"

CONFIGURATOR = Configurator()