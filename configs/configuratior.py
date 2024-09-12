import yaml

class Configurator:
    def __init__(self):
        PATH_YAML = "./configs/config.yaml"
        with open(PATH_YAML, "r") as file:
            self.app_config = yaml.load(file, Loader=yaml.FullLoader)
        
        self.load_theme_classification_config()

        
    def load_theme_classification_config(self):
        MODEL_NAME = self.app_config["themeclassifier"]['model_name']


CONFIGURATOR = Configurator()