import gradio as gr

from model import SentimentModel


def run_gui(model: SentimentModel):
    """
    Run the Gradio GUI for sentiment analysis.
    """
    iface = gr.Interface(
        fn=lambda review: model.classify(review),
        inputs=gr.Textbox(lines=2, placeholder="Enter a review here..."),
        outputs="text",
        title="Sentiment Analysis",
        description="Enter a review and get its sentiment prediction.",
        examples=[
            "Definitely worth the price.",
            "This product is amazing!",
            "I hate this product!",
            "I will never buy this again.",
        ],
    )
    iface.launch(share=True)
