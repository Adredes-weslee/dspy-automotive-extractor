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
from typing import Any

from _02_define_schema import ExtractionSignature, VehicleInfo
from settings import logger

# --- DSPy Module Definition ---

class ExtractionModule(dspy.Module):
    """A DSPy module for vehicle information extraction."""
    def __init__(self):
        super().__init__()
        self.extractor = dspy.TypedPredictor(ExtractionSignature)

    def forward(self, narrative: str) -> dspy.Prediction:
        """The forward pass of the module."""
        return self.extractor(narrative=narrative)


# --- Evaluation Metric ---

def extraction_metric(gold: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
    """
    Calculates a score based on how well the prediction matches the ground truth.
    We use a simplified F1-score calculation for the three fields.
    """
    try:
        pred_info: VehicleInfo = pred.vehicle_info
        
        gold_make = str(gold.make).upper()
        gold_model = str(gold.model).upper()
        gold_year = int(gold.year)

        pred_make = str(pred_info.make).upper()
        pred_model = str(pred_info.model).upper()
        pred_year = int(pred_info.year)

        tp = (1 if gold_make == pred_make else 0) + \
             (1 if gold_model == pred_model else 0) + \
             (1 if gold_year == pred_year else 0)
        
        fp = 3 - tp
        fn = 3 - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1

    except Exception as e:
        logger.error(f"Error in metric calculation: {e}. Prediction was: {pred}. Gold was: {gold}")
        return 0.0

if __name__ == "__main__":
    print("Testing the evaluation metric...")
    gold_example = dspy.Example(make="TESLA", model="MODEL Y", year=2022)
    
    pred_vehicle_info_1 = VehicleInfo(make="TESLA", model="MODEL Y", year=2022)
    pred_1 = dspy.Prediction(vehicle_info=pred_vehicle_info_1)
    score_1 = extraction_metric(gold_example, pred_1)
    print(f"Perfect match score: {score_1}")
    assert score_1 == 1.0

    pred_vehicle_info_2 = VehicleInfo(make="TESLA", model="MODEL 3", year=2022)
    pred_2 = dspy.Prediction(vehicle_info=pred_vehicle_info_2)
    score_2 = extraction_metric(gold_example, pred_2)
    print(f"Partial match score (2/3): {score_2}")
    assert abs(score_2 - 0.6666) < 0.01

    pred_vehicle_info_3 = VehicleInfo(make="FORD", model="F-150", year=2020)
    pred_3 = dspy.Prediction(vehicle_info=pred_vehicle_info_3)
    score_3 = extraction_metric(gold_example, pred_3)
    print(f"No match score: {score_3}")
    assert score_3 == 0.0
    
    print("\nMetric testing successful!")