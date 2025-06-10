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
        logger.error(
            f"Dataset not found at {filepath}. Please run the download command in the README."
        )
        sys.exit(1)

    try:
        df = pd.read_csv(
            filepath, encoding="ISO-8859-1", low_memory=False, on_bad_lines="skip"
        )
        logger.info(f"Successfully loaded {len(df)} rows from {filepath}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Map and rename columns
    required_columns = {
        "Narrative": "narrative",
        "Make": "make",
        "Model": "model",
        "Model Year": "year",
    }
    df = df[list(required_columns.keys())].rename(columns=required_columns)

    # Drop rows with missing required values
    df.dropna(subset=list(required_columns.values()), inplace=True)
    logger.info(f"After dropping NaN values: {len(df)} rows remaining.")

    # NEW: Filter out redacted narratives!
    initial_count = len(df)
    df = df[~df["narrative"].str.contains(r"\[REDACTED", na=False)]
    df = df[~df["narrative"].str.contains("MAY CONTAIN CONFIDENTIAL", na=False)]
    redacted_removed = initial_count - len(df)
    logger.info(
        f"After filtering redacted content: {len(df)} rows remaining (removed {redacted_removed} redacted rows)."
    )

    # Enhanced narrative length filtering (increased from 50 to 100)
    df = df[df["narrative"].str.len() > 100]
    logger.info(
        f"After filtering short narratives (>100 chars): {len(df)} rows remaining."
    )

    # Convert year to numeric and handle errors
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df.dropna(subset=["year"], inplace=True)

    logger.info(f"Cleaned data, {len(df)} rows remaining.")
    return df.sample(n=min(sample_size, len(df)), random_state=42)


def create_dspy_examples(df: pd.DataFrame) -> List[dspy.Example]:
    """
    Converts a pandas DataFrame into a list of DSPy Example objects.

    This function transforms the cleaned vehicle complaint data into the format
    required by DSPy for training and evaluation. Each row becomes a dspy.Example
    with the narrative as input and vehicle information (make, model, year) as
    the expected outputs.

    Args:
        df (pd.DataFrame): A cleaned DataFrame containing vehicle complaint data.
                          Must have columns: 'narrative', 'make', 'model', 'year'.

    Returns:
        List[dspy.Example]: A list of DSPy Example objects where each example
                           contains:
                           - narrative (input): The complaint text
                           - make (output): Vehicle manufacturer
                           - model (output): Vehicle model
                           - year (output): Vehicle year as string

    Example:
        >>> df = pd.DataFrame({
        ...     'narrative': ['2023 Tesla Model Y had brake issues'],
        ...     'make': ['Tesla'],
        ...     'model': ['Model Y'],
        ...     'year': [2023]
        ... })
        >>> examples = create_dspy_examples(df)
        >>> len(examples)
        1
    """
    examples = []
    for _, row in df.iterrows():
        example = dspy.Example(
            narrative=row["narrative"],
            make=str(row["make"]),
            model=str(row["model"]),
            year=str(int(row["year"])),  # ← CONVERT TO STRING!
        ).with_inputs("narrative")
        examples.append(example)
    return examples


def get_dataset() -> List[dspy.Example]:
    """
    Main function to load, clean, and convert the NHTSA dataset into DSPy format.

    This function orchestrates the complete data loading pipeline:
    1. Loads the raw NHTSA complaints CSV file
    2. Cleans and filters the data (removes redacted content, short narratives)
    3. Converts the cleaned data into DSPy Example objects

    Returns:
        List[dspy.Example]: A list of DSPy Example objects ready for training
                           and evaluation. Each example contains a vehicle
                           complaint narrative and the corresponding vehicle
                           information (make, model, year).

    Raises:
        SystemExit: If the dataset file is not found or cannot be loaded.

    Example:
        >>> dataset = get_dataset()
        >>> print(f"Loaded {len(dataset)} examples")
        Loaded 500 examples
        >>> print(dataset[0].narrative[:50])
        The owner of a 2023 Tesla Model Y contacted customer...
    """
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
