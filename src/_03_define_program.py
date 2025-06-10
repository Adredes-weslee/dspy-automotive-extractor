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
        self.extractor = dspy.Predict(ExtractionSignature)

    def forward(self, narrative: str) -> dspy.Prediction:
        """The forward pass of the module."""
        try:
            prediction = self.extractor(narrative=narrative)
            
            # Create VehicleInfo with STRING years (not integers!)
            vehicle_info = VehicleInfo(
                make=getattr(prediction, 'make', 'UNKNOWN'),
                model=getattr(prediction, 'model', 'UNKNOWN'),
                year=getattr(prediction, 'year', 'UNKNOWN')  # ← STRING, not 0!
            )
            
            return dspy.Prediction(
                vehicle_info=vehicle_info,
                make=vehicle_info.make,
                model=vehicle_info.model,
                year=vehicle_info.year
            )
            
        except Exception as e:
            # Fallback for any errors - use STRING "UNKNOWN", not integer 0!
            vehicle_info = VehicleInfo(make="UNKNOWN", model="UNKNOWN", year="UNKNOWN")
            return dspy.Prediction(
                vehicle_info=vehicle_info,
                make="UNKNOWN",
                model="UNKNOWN",
                year="UNKNOWN",  # ← STRING, not 0!
                error=str(e)
            )


# --- Evaluation Metric ---

def extraction_metric(gold: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
    """Calculate F1-score for vehicle extraction."""
    try:
        pred_info: VehicleInfo = pred.vehicle_info
        
        # Add debugging to see what's happening
        logger.info(f"Gold: make={gold.make}, model={gold.model}, year={gold.year}")
        logger.info(f"Pred: make={pred_info.make}, model={pred_info.model}, year={pred_info.year}")
        
        gold_make = str(gold.make).upper()
        gold_model = str(gold.model).upper()
        gold_year = str(gold.year)  # Keep as string

        pred_make = str(pred_info.make).upper()
        pred_model = str(pred_info.model).upper()
        pred_year = str(pred_info.year)  # Already a string

        # Calculate scores
        make_score = 1.0 if gold_make == pred_make else 0.0
        model_score = 1.0 if gold_model == pred_model else 0.0
        
        # Year comparison as strings
        if pred_year.upper() == "UNKNOWN":
            year_score = 0.0  # Unknown gets 0 points
        else:
            year_score = 1.0 if gold_year == pred_year else 0.0

        total_score = (make_score + model_score + year_score) / 3.0
        
        logger.debug(f"Scores: make={make_score}, model={model_score}, year={year_score}, total={total_score}")
        return total_score

    except Exception as e:
        logger.error(f"Error in extraction_metric: {e}")
        logger.error(f"Prediction type: {type(pred)}")
        logger.error(f"Prediction content: {pred}")
        return 0.0

if __name__ == "__main__":
    print("Testing the evaluation metric...")
    # Use STRING years in test cases too!
    gold_example = dspy.Example(make="TESLA", model="MODEL Y", year="2022")  # ← STRING
    
    pred_vehicle_info_1 = VehicleInfo(make="TESLA", model="MODEL Y", year="2022")  # ← STRING
    pred_1 = dspy.Prediction(vehicle_info=pred_vehicle_info_1)
    score_1 = extraction_metric(gold_example, pred_1)
    print(f"Perfect match score: {score_1}")
    assert score_1 == 1.0

    pred_vehicle_info_2 = VehicleInfo(make="TESLA", model="MODEL 3", year="2022")  # ← STRING
    pred_2 = dspy.Prediction(vehicle_info=pred_vehicle_info_2)
    score_2 = extraction_metric(gold_example, pred_2)
    print(f"Partial match score (2/3): {score_2}")
    assert abs(score_2 - 0.6666) < 0.01

    pred_vehicle_info_3 = VehicleInfo(make="FORD", model="F-150", year="2020")  # ← STRING
    pred_3 = dspy.Prediction(vehicle_info=pred_vehicle_info_3)
    score_3 = extraction_metric(gold_example, pred_3)
    print(f"No match score: {score_3}")
    assert score_3 == 0.0
    
    print("\nMetric testing successful!")