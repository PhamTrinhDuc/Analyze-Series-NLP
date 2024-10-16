from source.theme_classifier import ThemeClassifier
from source.character_network.named_entity_recognizer import NamedEntityRecognizer

def main():
#     themes = "friendship, hope, sacrifice, battle, self, development, betrayal, love, dialogue"
#     themes_list = themes.split(sep=", ")
#     themes_list = [theme for theme in themes_list if theme != "dialogue"]
#     output_pd = ThemeClassifier(themes).get_themes()
#     print(output_pd.head())

    ner = NamedEntityRecognizer()
    df  = ner.get_ners()
    print(df.head())


    
if __name__ == "__main__":
    main()