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

if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt')


def load_data(path: Path) -> pd.DataFrame:
    """
    Load data from the given path and return a DataFrame.
    Preprocess the data by removing neutral reviews and converting the labels to binary.
    """
    print(f"\nLoading data from {path}")
    data = pd.read_csv(path, names=['label', 'review'], usecols=[0, 2])
    data = data[data['label'] != 3]
    data['label'] = data['label'].apply(lambda x: 0 if x < 3 else 1)
    return data


def get_pipeline():
    """
    Create a pipeline with the following steps:
    1. DataCleaner: Preprocess the data by cleaning the text.
    2. Stemmer: Stem the tokens using the Porter Stemmer.
    3. CountVectorizer: Convert the text into a matrix of token counts.
    4. TfidfTransformer: Transform the count matrix to a normalized tf-idf representation.
    5. LogisticRegression: Train a logistic regression model to classify the reviews.
    """
    print("\n\nCreating pipeline...")
    return Pipeline([
        ('cleaner', DataCleaner()),
        ('stemmer', Stemmer()),
        ('vectorizer', CountVectorizer(ngram_range=((1, 2)), max_features=MAX_FEATURES)),
        ('tfidf', TfidfTransformer()),
        ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l2', C=7.9, n_jobs=-1, verbose=1))
    ],
        verbose=True,
        memory=CACHE_DIR
    )


def test_model(model):
    """
    Test the model using the test data and print the accuracy, precision, recall, f1 score, and confusion matrix.
    """
    print("\n\nTesting model...")
    test_data = load_data(TEST_DATA_PATH)
    y_pred = model.predict(test_data['review'])
    print(f"Accuracy: {metrics.accuracy_score(test_data['label'], y_pred)}")
    print(f"Precision: {metrics.precision_score(test_data['label'], y_pred)}")
    print(f"Recall: {metrics.recall_score(test_data['label'], y_pred)}")
    print(f"F1 Score: {metrics.f1_score(test_data['label'], y_pred)}")
    print(f"Confusion Matrix:\n{metrics.confusion_matrix(test_data['label'], y_pred)}")


def save_model(model):
    """
    Save the model to the MODEL_PATH using pickle.
    """
    print(f"\n\nSaving model to {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)


def load_model():
    """
    Load the model from the MODEL_PATH using pickle.
    """
    print(f"Loading model from {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)


def classify_review(model, review):
    """
    Classify the given review using the model.
    """
    review_series = pd.Series([review])
    return model.predict(review_series)[0]


def gui():
    """
    Run the GUI using gradio to classify the sentiment of a review.
    """
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError("Model not found. Train the model using 'make model' and try again.")
    model = load_model()
    iface = gr.Interface(
        fn=lambda review: "Positive ✅" if classify_review(model, review) == 1 else "Negative ⛔",
        inputs=gr.Textbox(lines=2, placeholder="Enter a review here..."),
        outputs="text",
        title="Sentiment Analysis",
        description="Enter a review and get its sentiment prediction.",
        examples=["Definitely worth the price.", "This product is amazing!", "I hate this product!",
                  "I will never buy this again.",
                  ],
    )
    iface.launch()


def model():
    """
    Train the model using the training data and save it to the MODEL_PATH.
    """
    if not Path(TRAIN_DATA_PATH).exists() or not Path(TEST_DATA_PATH).exists():
        raise FileNotFoundError(
            "Training or testing data not found. Download the data using 'make download' and try again.")
    train_data = load_data(TRAIN_DATA_PATH)
    pipeline = get_pipeline()
    print("\n\nFitting pipeline...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(train_data['review'], train_data['label'])
    test_model(pipeline)
    save_model(pipeline)


# Run the model or GUI based on the argument provided
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Specify 'model' to train and save the model, or 'gui' to run the GUI.")
    args = parser.parse_args()

    if args.mode == 'model':
        model()
    elif args.mode == 'gui':
        gui()
    else:
        print("Invalid argument. Please specify either 'model' or 'gui'.")
