# from source.theme_classifier import ThemeClassifier
from source.character_network.named_entity_recognizer import NamedEntityRecognizer
from source.character_network.character_netowork_generator import CharacterNetworkGenerator



def main():

    # theme_classifier = ThemeClassifier()
    # theme_classifier.testing()

    ner = NamedEntityRecognizer()
    # output = ner.get_named_entities(scripts="Okay… Let’s go, Akamaru! Akamaru, Fang Over Fang! Fang Over Fang! Let’s go, Kiba!")
    # print(output)
    df = ner.get_ners()
    print(df.head())

    # character_network = CharacterNetworkGenerator()
    # character_network.generate_entity_charactor()

    
if __name__ == "__main__":
    main()