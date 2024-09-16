import yaml

class Configurator:
    def __init__(self):
        PATH_YAML = "./config/config.yml"
        with open(PATH_YAML, "r") as file:
            self.app_config = yaml.load(file, Loader=yaml.FullLoader)
        
        self.load_theme_classification_config()
        self.load_data_config()

        
    def load_theme_classification_config(self):
        self.MODEL_NAME = self.app_config["themeclassifier"]['model_name']

    def load_data_config(self):
        self.subtitle_path = self.app_config["data"]["subtitle_path"]
        self.dialogue_path = self.app_config["data"]["dialogue_path"]
        self.jutsu_path = self.app_config["data"]["jutsu_path"]

CONFIGURATOR = Configurator()