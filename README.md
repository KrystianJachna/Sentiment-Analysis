# Sentiment Analysis Project

This project implements a sentiment analysis classifier to determine if Amazon reviews are positive or negative using a
Logistic Regression model.

## Table of Contents

- [Description](#description)
- [Installation and Setup](#installation-and-setup)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
- [Usage](#usage)
    - [GUI](#gui)
    - [Prediction](#prediction)
    - [Training the Model](#training-the-model)
    - [Help](#help)
- [Model Details](#model-details)
    - [Preprocessing Pipeline](#preprocessing-pipeline)
    - [Classifier](#classifier)
    - [Evaluation](#evaluation)
- [License](#license)

## Description

This is a simple sentiment analysis project written in Python. The sentiment is classified as either positive or
negative with Logistic Regression model. It
uses [Amazon Review Data](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) dataset for training and testing
the model. More information about the dataset can be found [here](data/README.md). Data analysis and parameter tuning
were performed in Jupyter notebooks, which can be found in the [notebooks'](notebooks) directory.

## Installation and Setup

### Prerequisites:

Ensure you have the following installed on your system:

- Python 3.11 or higher
- make
- Poetry

### Installation Steps:

After cloning the repository and navigating to the project directory, follow these steps:

1. Install the required packages, create a virtual environment and additional setup:
    ```bash
    make install
    ```

2. Download the data and model from Google Drive:
    ```bash
    make download
    ```

## Usage

After the installation, you can use the following commands:

### GUI

To run the GUI, use the following command:

```bash
make gui
```

This command starts a Gradio-based GUI, allowing users to input a review and get a sentiment prediction. Access the GUI
in your browser at the URL displayed in the terminal (default: [http://127.0.0.1:7860](http://127.0.0.1:7860)).
Additionally, with the share
option set to True, the model is also hosted on a public URL, which is displayed in the terminal.

![GUI](assets/gradio.png)

### Prediction

To predict the sentiment of a given review in the terminal, use the following command:

```bash
make prediction review='This is a great product!'
```

This command uses the saved model (`model/model.pkl`) to predict the sentiment of the provided review.

### Training the Model

To train and test the model, use the following command:

```bash
make model
```

This command trains and tests the model using `data/train.csv` and `data/test.csv,` saving the trained model
to `model/model.pkl`.

### Help

To see all available commands, use the following command:

```bash
make help
```

## Model Details

### Preprocessing Pipeline

The preprocessing pipeline consists of:

1. Data Cleaning:
    - Convert all words to lowercase.
    - Remove stopwords, punctuation, URLs, handles, emojis, and extra spaces

2. Stemming:
    - Reduce words to their root form using a stemming algorithm.

3. Vectorization:
    - Convert text into a matrix of token counts with ngram_range set to (1, 2).
    - Limit the number of features using a predefined constant MAX_FEATURES (200 000).

4. TF-IDF Transformation:
    - Transform the matrix of token counts into a normalized TF-IDF representation.

### Classifier

The model uses a Logistic Regression classifier with the following hyperparameters after testing different values in
`/notebooks/model_experimentation.ipynb`:

- `C`: 7.9
- `penalty`: 'l2'
- `solver`: 'liblinear'

### Evaluation

The model was evaluated using the following metrics:

| Metric    | Value              |
|-----------|--------------------|
| Accuracy  | 0.8877980769230769 |
| Precision | 0.8867523580472798 |
| Recall    | 0.88915            |
| F1 Score  | 0.887949560498019  |

### Confusion Matrix

|                 | Predicted Negative | Predicted Positive |
|-----------------|--------------------|--------------------|
| Actual Negative | 230476             | 29524              |
| Actual Positive | 28821              | 231179             |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
