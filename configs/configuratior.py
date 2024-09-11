import yaml


class Configurator:
    def __init__(self):
        PATH_YAML = "./configs/config.yaml"
        with open(PATH_YAML, "r") as file:
            self.app_config = yaml.load(file, Loader=yaml.FullLoader)




CONFIGURATOR = Configurator()