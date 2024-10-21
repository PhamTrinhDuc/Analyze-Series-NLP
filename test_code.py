# from source.theme_classifier import ThemeClassifier
from source.character_network.named_entity_recognizer import NamedEntityRecognizer
from source.character_network.character_netowork_generator import CharacterNetworkGenerator
from source.jutsu_classify.jutsu_classifier import JutsuClassifier


def main():

    # theme_classifier = ThemeClassifier()
    # theme_classifier.testing()

    ner = NamedEntityRecognizer()
    # output = ner.get_named_entities(scripts="Okay… Let’s go, Akamaru! Akamaru, Fang Over Fang! Fang Over Fang! Let’s go, Kiba!")
    # print(output)
    # df = ner.get_ners()
    # print(df.head())

    # character_network = CharacterNetworkGenerator()
    # character_network.generate_entity_charactor()

    jutsu_classifier = JutsuClassifier()
    train_dateset, test_dataset = jutsu_classifier.load_data()
    jutsu_classifier.training(train_data=train_dateset, test_data=test_dataset)

if __name__ == "__main__":
    main()
    # pass
