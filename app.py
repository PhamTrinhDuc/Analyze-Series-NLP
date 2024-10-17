import gradio as gr
from source.character_network.named_entity_recognizer import NamedEntityRecognizer
from source.character_network.character_netowork_generator import CharacterNetworkGenerator
from source.theme_classifier import ThemeClassifier

def get_themes(theme_list_str: str):
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes()
    # Remove dialogue from the theme list
    theme_list = [theme.strip() for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list]


    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme','Score']

    output_chart = gr.BarPlot(
        output_df,
        x="Theme",
        y="Score",
        title="Series Themes",
        tooltip=["Theme","Score"],
        vertical=False,
        width=500,
        height=260
    )
    return output_chart

def character_network():
    ner_character = NamedEntityRecognizer()
    df = ner_character.get_ners()

    network_character = CharacterNetworkGenerator()
    relationship_df = network_character.generate_entity_charactor(df=df)
    output_html = network_character.draw_entity_graph(entity_relationship=relationship_df)
    return output_html


def main():
    with gr.Blocks() as iface:
        # Theme Classification Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Claasifiers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes", 
                                                placeholder="list of labels you want to categorize", 
                                                value="dialogue, sacrifice, self, hope, betrayal, friendship, development, love, battle")
                        get_themes_button =gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list], outputs=[plot])

        # Character Network Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1> Character Network</h1>")
                with gr.Row():
                    with gr.Column():
                        html_output = gr.HTML()
                    with gr.Column():
                        character_network_button = gr.Button("Generate Character Network")
                        character_network_button.click(character_network, outputs=[html_output])


    iface.launch(share=True)

if __name__ == "__main__":
    main()