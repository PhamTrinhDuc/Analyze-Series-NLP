import gradio as gr
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
    
    iface.launch(share=True)

if __name__ == "__main__":
    main()