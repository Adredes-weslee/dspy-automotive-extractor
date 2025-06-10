"""
_03_define_program.py

This script defines the core components of our DSPy program:
1.  The `ExtractionModule`: This is a dspy.Module that encapsulates the logic
    for our task. It uses a `dspy.TypedPredictor`, which is designed to work
    with Pydantic models to ensure structured, validated outputs.
2.  The `extraction_metric`: This is the evaluation function. It takes a ground
    truth example and a model's prediction, compares them, and returns a
    numerical score (F1-score). This score is what the DSPy optimizer will
    try to maximize.
"""
import dspy
import sys
from pathlib import Path
from typing import Any

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src._02_define_schema import ExtractionSignature, VehicleInfo
from src.settings import logger

# --- DSPy Module Definition ---

class ExtractionModule(dspy.Module):
    """A DSPy module for vehicle information extraction."""
    def __init__(self):
        super().__init__()
        # TypedPredictor is ideal for forcing structured Pydantic outputs
        self.extractor = dspy.TypedPredictor(ExtractionSignature)

    def forward(self, narrative: str) -> dspy.Prediction:
        """
        The forward pass of the module.

        Args:
            narrative (str): The unstructured complaint text.

        Returns:
            dspy.Prediction: A prediction object containing the extracted VehicleInfo.
        """
        return self.extractor(narrative=narrative)


# --- Evaluation Metric ---

def extraction_metric(gold: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
    """
    Calculates a score based on how well the prediction matches the ground truth.
    We use a simplified F1-score calculation for the three fields.

    Args:
        gold (dspy.Example): The ground truth example, with correct make, model, and year.
        pred (dspy.Prediction): The model's prediction, containing the extracted vehicle_info.

    Returns:
        float: A score between 0.0 (no match) and 1.0 (perfect match).
    """
    try:
        # Extract the Pydantic model from the prediction
        pred_info: VehicleInfo = pred.vehicle_info
        
        # Ground truth values
        gold_make = str(gold.make).upper()
        gold_model = str(gold.model).upper()
        gold_year = int(gold.year)

        # Predicted values
        pred_make = str(pred_info.make).upper()
        pred_model = str(pred_info.model).upper()
        pred_year = int(pred_info.year)

        # Calculate True Positives (correctly matched fields)
        tp = 0
        if gold_make == pred_make:
            tp += 1
        if gold_model == pred_model:
            tp += 1
        if gold_year == pred_year:
            tp += 1

        # False Positives and False Negatives are based on the number of fields (3)
        fp = 3 - tp # If a pred field isn't a TP, it's counted as a miss.
        fn = 3 - tp # If a gold field isn't a TP, it's counted as a miss.

        # Calculate Precision, Recall, and F1 Score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1

    except Exception as e:
        logger.error(f"Error in metric calculation: {e}. Prediction was: {pred}. Gold was: {gold}")
        return 0.0


if __name__ == "__main__":
    # This block allows you to run this script directly to test the metric
    print("Testing the evaluation metric...")

    # Create dummy gold standard and prediction objects
    gold_example = dspy.Example(make="TESLA", model="MODEL Y", year=2022)
    
    # Test case 1: Perfect match
    pred_vehicle_info_1 = VehicleInfo(make="TESLA", model="MODEL Y", year=2022)
    pred_1 = dspy.Prediction(vehicle_info=pred_vehicle_info_1)
    score_1 = extraction_metric(gold_example, pred_1)
    print(f"Perfect match score: {score_1}") # Expected: 1.0
    assert score_1 == 1.0

    # Test case 2: Partial match (2 out of 3)
    pred_vehicle_info_2 = VehicleInfo(make="TESLA", model="MODEL 3", year=2022)
    pred_2 = dspy.Prediction(vehicle_info=pred_vehicle_info_2)
    score_2 = extraction_metric(gold_example, pred_2)
    print(f"Partial match score (2/3): {score_2}") # Expected: ~0.667
    assert abs(score_2 - 0.6666) < 0.01

    # Test case 3: No match
    pred_vehicle_info_3 = VehicleInfo(make="FORD", model="F-150", year=2020)
    pred_3 = dspy.Prediction(vehicle_info=pred_vehicle_info_3)
    score_3 = extraction_metric(gold_example, pred_3)
    print(f"No match score: {score_3}") # Expected: 0.0
    assert score_3 == 0.0
    
    print("\nMetric testing successful!")
