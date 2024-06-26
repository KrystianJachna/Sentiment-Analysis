import argparse
from pathlib import Path

from const import MODEL_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH
from model import SentimentModel
from gradio_gui import run_gui


def main():
    """
    Main function to parse command line arguments and run the appropriate mode.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",
                        help="Specify 'model' to train and save the model, 'gui' to run the GUI, "
                             "or 'predict' to classify a review.")
    parser.add_argument("--review", help="Review text to predict sentiment for, "
                                         "used with 'predict' mode.")
    args = parser.parse_args()

    model = SentimentModel()

    if args.mode == 'model':
        model.train(TRAIN_DATA_PATH, TEST_DATA_PATH)
    elif args.mode == 'gui':
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError("Model not found. Train the model first and try again.")
        model.load(MODEL_PATH)
        run_gui(model)
    elif args.mode == 'predict':
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError("Model not found. Train the model first and try again.")
        model.load(MODEL_PATH)
        prediction = model.classify(args.review)
        print(f"\n\nReview: {args.review}\nSentiment: {prediction}")
    else:
        print("Invalid argument. Please specify either 'model', 'gui', or 'predict'.")


if __name__ == '__main__':
    main()
