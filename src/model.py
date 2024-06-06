import pickle
import warnings

import nltk
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pathlib import Path

from const import MAX_FEATURES, CACHE_DIR, MODEL_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH
from data_loader import load_data
from preprocessing.DataCleaner import DataCleaner
from preprocessing.Stemmer import Stemmer

# Ensure the 'punkt' tokenizer is available
if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt')

class SentimentModel:
    def __init__(self):
        self.pipeline = self.create_pipeline()

    def create_pipeline(self) -> Pipeline:
        """
        Creates a pipeline with data preprocessing and classification steps.
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

    def train(self, train_data_path: Path, test_data_path: Path):
        """
        Trains the model on the training data and evaluates it on the test data.
        """
        if not train_data_path.exists() or not test_data_path.exists():
            raise FileNotFoundError("Training or testing data not found. Download the data and try again.")
        train_data = load_data(train_data_path)
        print("Fitting pipeline...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pipeline.fit(train_data['review'], train_data['label'])
        self.evaluate(test_data_path)
        self.save(MODEL_PATH)

    def evaluate(self, test_data_path: Path):
        """
        Tests the model on the test data and prints evaluation metrics.
        """
        print("Testing model...")
        test_data = load_data(test_data_path)
        y_pred = self.pipeline.predict(test_data['review'])
        print(f"Accuracy: {metrics.accuracy_score(test_data['label'], y_pred)}")
        print(f"Precision: {metrics.precision_score(test_data['label'], y_pred)}")
        print(f"Recall: {metrics.recall_score(test_data['label'], y_pred)}")
        print(f"F1 Score: {metrics.f1_score(test_data['label'], y_pred)}")
        print(f"Confusion Matrix:\n{metrics.confusion_matrix(test_data['label'], y_pred)}")

    def save(self, model_path: Path):
        """
        Saves the model to a file using pickle.
        """
        print(f"Saving model to {model_path}")
        with open(model_path, 'wb') as file:
            pickle.dump(self.pipeline, file)

    def load(self, model_path: Path):
        """
        Loads the model from a file using pickle.
        """
        print(f"Loading model from {model_path}")
        with open(model_path, 'rb') as file:
            self.pipeline = pickle.load(file)

    def preprocess_review(self, review: pd.Series) -> str:
        """
        Preprocesses a review using DataCleaner and Stemmer.
        """
        cleaner = DataCleaner()
        stemmer = Stemmer()
        cleaned_review = cleaner.transform(review)
        stemmed_review = stemmer.transform(cleaned_review)
        return stemmed_review[0]

    def classify(self, review: str) -> str:
        """
        Classifies a given review using the model.
        """
        review_series = pd.Series([review])
        preprocessed_review = self.preprocess_review(review_series)
        if not preprocessed_review.strip():
            return "Review is too general or empty after preprocessing. Please provide a more detailed review."
        pred = self.pipeline.predict(review_series)[0]
        return "Positive ✅" if pred == 1 else "Negative ⛔"
