# Data for the Sentiment Analysis Project

## Data Description

The data in this directory originates from Amazon reviews and is used for sentiment analysis. 
It consists of 3,000,000 training samples and 650,000 testing samples. 
The reviews are labeled with a number from 1 to 5, representing the number of stars given by the reviewer. 
For our binary classifier, we will treat reviews with 1-2 stars as negative and reviews with 4-5 stars as positive. 
Reviews with 3 stars will not be used in the model. 

## Data Structure

The raw data is stored in CSV files (`raw/train.csv` and `raw/test.csv`). 
Each file contains 3 columns: class index (ranging from 1 to 5), review title, and review text. 
For example, a row in the data might look like this: 

```"1","mens ultrasheer","This model may be ok for sedentary types, but I'm active and get around alot in my job - consistently found these stockings rolled up down by my ankles! Not Good!! Solution: go with the standard compression stocking, 20-30, stock #114622. Excellent support, stays up and gives me what I need. Both pair of these also tore as I struggled to pull them up all the time. Good riddance/bad investment!"```

After preprocessing, the processed data will be stored in the `processed/` directory.

## Data Source

The data was collected by Xiang Zhang (xiang.zhang@nyu.edu) and is used as a benchmark for text classification.

[Xiang Zhang's Google Drive dir](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?resourcekey=0-TLwzfR2O-D2aPitmn5o9VQ)

## Downloading the Data

You can use the provided `Makefile` to download the data. Simply run the following command in the repository directory:
```bash
make data
```

Or you can run the following command in the `data/` directory:
```bash
make download
```

This will download the data from the source and store it in the `raw/` directory.