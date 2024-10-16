import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from source.character_network.named_entity_recognizer import NamedEntityRecognizer
from configs.configurator import CONFIGURATOR


class CharacterNetworkGenerator:
    def __init__(self):
        self.df = NamedEntityRecognizer().get_ners()

    def generate_entity_charactor(self, df: pd.DataFrame =  None):
        window_size = 10
        entity_relationship = []

        for row in self.df['ners']:
            previous_entities_in_window = []
            print(type(row))
            for sentence in row:
                print(sentence)
            break