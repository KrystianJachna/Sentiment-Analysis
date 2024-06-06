import argparse
import pickle
import warnings
from pathlib import Path

import gradio as gr
import nltk
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from const import MAX_FEATURES, CACHE_DIR, MODEL_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH
from preprocessing.DataCleaner import DataCleaner
from preprocessing.Stemmer import Stemmer

# Ensure NLTK 'punkt' tokenizer is available
if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt')


def load_data(path: Path) -> pd.DataFrame:
    """
    Load data from the given path and return a DataFrame.
    Preprocess the data by removing neutral reviews and converting the labels to binary.
    """
    print(f"Loading data from {path}")
    data = pd.read_csv(path, names=['label', 'review'], usecols=[0, 2])
    data = data[data['label'] != 3]
    data['label'] = data['label'].apply(lambda x: 0 if x < 3 else 1)
    return data


def create_pipeline() -> Pipeline:
    """
    Create a pipeline with the following steps:
    1. DataCleaner: Preprocess the data by cleaning the text.
    2. Stemmer: Stem the tokens using the Porter Stemmer.
    3. CountVectorizer: Convert the text into a matrix of token counts.
    4. TfidfTransformer: Transform the count matrix to a normalized tf-idf representation.
    5. LogisticRegression: Train a logistic regression model to classify the reviews.
    """
    print("Creating pipeline...")
    return Pipeline([
        ('cleaner', DataCleaner()),
        ('stemmer', Stemmer()),
        ('vectorizer', CountVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES)),
        ('tfidf', TfidfTransformer()),
        ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l2', C=7.9, n_jobs=-1, verbose=1))
    ],
        verbose=True,
        memory=CACHE_DIR
    )


def evaluate_model(model: Pipeline):
    """
    Test the model using the test data and print the accuracy, precision, recall, F1 score, and confusion matrix.
    """
    print("Testing model...")
    test_data = load_data(TEST_DATA_PATH)
    y_pred = model.predict(test_data['review'])
    print(f"Accuracy: {metrics.accuracy_score(test_data['label'], y_pred)}")
    print(f"Precision: {metrics.precision_score(test_data['label'], y_pred)}")
    print(f"Recall: {metrics.recall_score(test_data['label'], y_pred)}")
    print(f"F1 Score: {metrics.f1_score(test_data['label'], y_pred)}")
    print(f"Confusion Matrix:\n{metrics.confusion_matrix(test_data['label'], y_pred)}")


def save_model(model: Pipeline):
    """
    Save the model to the MODEL_PATH using pickle.
    """
    print(f"Saving model to {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)


def load_model() -> Pipeline:
    """
    Load the model from the MODEL_PATH using pickle.
    """
    print(f"Loading model from {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)


def preprocess_review(review: pd.Series) -> str:
    """
    Preprocess the review using DataCleaner and Stemmer.
    """
    cleaner = DataCleaner()
    stemmer = Stemmer()
    cleaned_review = cleaner.transform(review)
    stemmed_review = stemmer.transform(cleaned_review)
    return stemmed_review[0]


def classify_review(model: Pipeline, review: str) -> str:
    """
    Classify the given review using the model.
    """
    review_series = pd.Series([review])
    preprocessed_review = preprocess_review(review_series)
    if not preprocessed_review.strip():
        return "Review is too general or empty after preprocessing. Please provide a more detailed review."
    pred = model.predict(review_series)[0]
    return "Positive ✅" if pred == 1 else "Negative ⛔"


def run_gui():
    """
    Run the GUI using gradio to classify the sentiment of a review.
    """
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError("Model not found. Train the model using 'make model' and try again.")
    model = load_model()
    iface = gr.Interface(
        fn=lambda review: classify_review(model, review),
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


def train_and_save_model():
    """
    Train the model using the training data and save it to the MODEL_PATH.
    """
    if not Path(TRAIN_DATA_PATH).exists() or not Path(TEST_DATA_PATH).exists():
        raise FileNotFoundError(
            "Training or testing data not found. Download the data using 'make download' and try again.")
    train_data = load_data(TRAIN_DATA_PATH)
    pipeline = create_pipeline()
    print("Fitting pipeline...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(train_data['review'], train_data['label'])
    evaluate_model(pipeline)
    save_model(pipeline)


def main():
    """
    Main function to run the script based on the provided mode.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",
                        help="Specify 'model' to train and save the model, 'gui' to run the GUI, or 'predict' to classify a review.")
    parser.add_argument("--review", help="Review text to predict sentiment for, used with 'predict' mode.")
    args = parser.parse_args()

    if args.mode == 'model':
        train_and_save_model()
    elif args.mode == 'gui':
        run_gui()
    elif args.mode == 'predict':
        if args.review is None:
            print("Please provide a review for prediction.")
        else:
            model = load_model()
            prediction = classify_review(model, args.review)
            print(f"Review: {args.review}\nSentiment: {prediction}")
    else:
        print("Invalid argument. Please specify either 'model', 'gui', or 'predict'.")


if __name__ == '__main__':
    main()
