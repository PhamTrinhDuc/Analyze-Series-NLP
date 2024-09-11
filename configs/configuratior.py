import yaml


class Configurator:
    def __init__(self):
        with open("./configs/config.yaml", 'r') as f:
            self.app_config = yaml.load(f, Loader=yaml.FullLoader)
    
        self.load_crawler_config()

    def load_crawler_config(self):
        self.url_website = self.app_config['crawler']['url_website']
        self.name_data = self.app_config['crawler']['name_data']




CONFIGURATOR = Configurator()