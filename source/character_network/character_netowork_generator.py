import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from source.character_network.named_entity_recognizer import NamedEntityRecognizer
from configs.configurator import CONFIGURATOR


class CharacterNetworkGenerator:
    def __init__(self):
        self.df = NamedEntityRecognizer().get_ners()

    def generate_entity_charactor(self, df: pd.DataFrame =  None) -> pd.DataFrame:
        entity_relationship = []

        for entity in df['ners']:

            for character in entity:
                for i in range(len(entity)):
                    if character != entity[i]:
                        entity_relationship.append(sorted([character, entity[i]]))
        
        entity_relationship = pd.DataFrame({'value': entity_relationship})
        entity_relationship['source'] = entity_relationship['value'].apply(lambda x: x[0])
        entity_relationship['target'] = entity_relationship['value'].apply(lambda x: x[1])
        entity_relationship = entity_relationship.groupby(['source', 'target']).count().reset_index()
        entity_relationship = entity_relationship.sort_values('value', ascending=False)
        return entity_relationship
    
    def draw_entity_graph(self, entity_relationship: pd.DataFrame):
        """
        Vẽ đồ thị thực thể từ DataFrame chứa mối quan hệ giữa các thực thể.
        Hàm này lọc các mối quan hệ có giá trị lớn hơn 1 và lấy tối đa 200 mối quan hệ đầu tiên.
        Sau đó, nó tạo đồ thị từ DataFrame và hiển thị đồ thị này dưới dạng HTML.
        Args:
        -----------
        entity_relationship : pd.DataFrame
            DataFrame chứa các mối quan hệ giữa các thực thể với các cột 'source', 'target' và 'value'.
        Returns:
        -----------
        str
            Chuỗi HTML chứa iframe để hiển thị đồ thị thực thể.
        """

        entity_relationship = entity_relationship[entity_relationship['value'] > 1].head(200)
        G = nx.from_pandas_edgelist(
            entity_relationship, 
            source='source', target='target', 
            edge_attr='value',
            create_using=nx.Graph()
        )
        net = Network(notebook=True, width="1000px", height="700px", bgcolor="#222222", font_color="white")
        node_degree = dict(G.degree)

        nx.set_node_attributes(G, node_degree, 'size')
        net.from_nx(G)
        html = net.generate_html()
        html = html.replace("'", "\"")

        output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
        display-capture; encrypted-media;" sandbox="allow-modals allow-forms
        allow-scripts allow-same-origin allow-popups
        allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
        allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

        return output_html


