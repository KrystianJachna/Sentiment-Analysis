import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing.DataCleaner import DataCleaner
from preprocessing.Stemmer import Stemmer
from const import MAX_FEATURES, CACHE_DIR, MODEL_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH


def load_data(path: Path) -> pd.DataFrame:
    print(f"\nLoading data from {path}")
    data = pd.read_csv(path, names=['label', 'review'], usecols=[0, 2])
    data = data[data['label'] != 3]
    data['label'] = data['label'].apply(lambda x: 0 if x < 3 else 1)
    return data


def get_pipeline():
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
    print("\n\nTesting model...")
    test_data = load_data(TEST_DATA_PATH)
    y_pred = model.predict(test_data['review'])
    print(f"Accuracy: {metrics.accuracy_score(test_data['label'], y_pred)}")
    print(f"Precision: {metrics.precision_score(test_data['label'], y_pred)}")
    print(f"Recall: {metrics.recall_score(test_data['label'], y_pred)}")
    print(f"F1 Score: {metrics.f1_score(test_data['label'], y_pred)}")
    print(f"Confusion Matrix:\n{metrics.confusion_matrix(test_data['label'], y_pred)}")



def save_model(model):
    print(f"\n\nSaving model to {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

def gui():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Specify 'model' to train and save the model, or 'gui' to run the GUI.")
    args = parser.parse_args()

    if args.mode == 'model':
        train_data = load_data(TRAIN_DATA_PATH)
        pipeline = get_pipeline()
        print("\n\nFitting pipeline...")
        pipeline.fit(train_data['review'], train_data['label'])
        test_model(pipeline)
        save_model(pipeline)
    elif args.mode == 'gui':
        gui()
    else:
        print("Invalid argument. Please specify either 'model' or 'gui'.")
