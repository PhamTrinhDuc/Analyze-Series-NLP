import gradio as gr
from source.theme_classifier import ThemeClassifier

def get_themes(themes_list_str: str):
    themes_list = themes_list_str.split(sep=", ")
    themes_list = [theme for theme in themes_list if theme != "dialogue"]
    output_df = ThemeClassifier(themes_list).get_themes()[themes_list]
    output_df = output_df.sum().reset_index()
    output_df.columns = ['theme', 'score']
    print(output_df['theme'])

    output_chart = gr.BarPlot(
        value=output_df,
        x="Theme",
        y='Score',
        title="Series Themes",
        tooltip=["Theme", "Score"],
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
                        theme_list = gr.Textbox(label="Themes")
                        get_themes_button =gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list], outputs=[plot])
    
    
    iface.launch(share=True)

if __name__ == "__main__":
    # output_chart = get_themes("friendship, hope, sacrifice, battle, self, development, betrayal, love, dialogue")
    # print(output_chart)
    main()