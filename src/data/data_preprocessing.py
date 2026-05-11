import os
import re
import string
import sys
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    from src.logger import logging
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if "src" in sys.modules:
        del sys.modules["src"]
    from src.logger import logging

ROOT_DIR = Path(__file__).resolve().parents[2]

nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_dataframe(df, col='text'):
    """
    Preprocess a DataFrame by applying text preprocessing to a specific column.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        col (str): The name of the column containing text.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        """Helper function to preprocess a single text string."""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove numbers
        text = ''.join([char for char in text if not char.isdigit()])
        # Convert to lowercase
        text = text.lower()
        # Remove punctuations
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub(r"\s+", " ", text).strip()
        # Remove stop words
        text = " ".join([word for word in text.split() if word not in stop_words])
        # Lemmatization
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text

    # Apply preprocessing to the specified column
    df[col] = df[col].apply(preprocess_text)

    # Remove small sentences (less than 3 words)
    # df[col] = df[col].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)

    # Drop rows with NaN values
    df = df.dropna(subset=[col])
    logging.info("Data pre-processing completed")
    return df


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv(ROOT_DIR / "data" / "raw" / "train.csv")
        test_data = pd.read_csv(ROOT_DIR / "data" / "raw" / "test.csv")
        logging.info('data loaded properly')

        # Transform the data
        train_processed_data = preprocess_dataframe(train_data, 'review')
        test_processed_data = preprocess_dataframe(test_data, 'review')

        # Store the data inside data/processed
        data_path = ROOT_DIR / "data" / "interim"
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(data_path / "train_processed.csv", index=False)
        test_processed_data.to_csv(data_path / "test_processed.csv", index=False)
        
        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
