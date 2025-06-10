"""
_01_load_data.py

This script is the first step in the DSPy optimization pipeline. It handles:
1.  Loading the raw vehicle complaint data from the NHTSA CSV file.
2.  Cleaning and preprocessing the data to ensure it's suitable for the LLM.
3.  Converting the cleaned data into a list of dspy.Example objects,
    which is the standard format used for training and evaluation in DSPy.
"""
import sys
from pathlib import Path
from typing import List

import dspy
import pandas as pd

# Use settings from the central settings.py module
from settings import DATA_DIR, logger

def load_and_clean_data(filepath: Path, sample_size: int = 500) -> pd.DataFrame:
    """
    Loads and cleans the NHTSA vehicle complaints dataset.

    Args:
        filepath (Path): The path to the raw CSV data file.
        sample_size (int): The number of samples to use from the dataset.
                           A smaller sample is faster for development.

    Returns:
        pd.DataFrame: A cleaned DataFrame with necessary columns.
    """
    if not filepath.exists():
        logger.error(f"Dataset not found at {filepath}. Please run the download command in the README.")
        sys.exit(1)

    try:
        df = pd.read_csv(
            filepath,
            encoding='ISO-8859-1',
            low_memory=False,
            on_bad_lines='skip'
        )
        logger.info(f"Successfully loaded {len(df)} rows from {filepath}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    required_columns = {'Narrative': 'narrative', 'Make': 'make', 'Model': 'model', 'Model Year': 'year'}
    df = df[list(required_columns.keys())].rename(columns=required_columns)

    df.dropna(subset=list(required_columns.values()), inplace=True)
    df = df[df['narrative'].str.len() > 50]
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df.dropna(subset=['year'], inplace=True)

    logger.info(f"Cleaned data, {len(df)} rows remaining.")
    return df.sample(n=min(sample_size, len(df)), random_state=42)


def create_dspy_examples(df: pd.DataFrame) -> List[dspy.Example]:
    """Converts a DataFrame into a list of dspy.Example objects."""
    examples = []
    for _, row in df.iterrows():
        example = dspy.Example(
            narrative=row['narrative'],
            make=str(row['make']),
            model=str(row['model']),
            year=str(int(row['year']))  # ← CONVERT TO STRING!
        ).with_inputs("narrative")
        examples.append(example)
    return examples


def get_dataset() -> List[dspy.Example]:
    """Main function to get the processed dataset."""
    data_file = DATA_DIR / "NHTSA_complaints.csv"
    cleaned_df = load_and_clean_data(data_file)
    dspy_examples = create_dspy_examples(cleaned_df)
    return dspy_examples


if __name__ == "__main__":
    logger.info("Running data loading test...")
    dataset = get_dataset()
    if dataset:
        logger.info(f"Successfully loaded and processed {len(dataset)} examples.")
        logger.info("Here are the first two examples:")
        for i in range(min(2, len(dataset))):
            print(dataset[i])
    else:
        logger.error("Failed to load dataset.")