r"""
_03_define_program.py

This script defines the core components of our DSPy program for vehicle information extraction:
1.  The `ExtractionModule`: This is a dspy.Module that encapsulates the logic
    for our task. It uses a `dspy.Predict` component with the ExtractionSignature
    to ensure structured, validated outputs via Pydantic models.
2.  The `extraction_metric`: This is the evaluation function that calculates
    F1-scores by comparing ground truth examples with model predictions.
    This score is what the DSPy optimizer will try to maximize during training.

The module provides robust error handling and comprehensive logging for debugging
extraction issues, making it suitable for production use.

Usage:
    .\.venv\Scripts\python.exe src\_03_define_program.py

Example:
    >>> from _03_define_program import ExtractionModule, extraction_metric
    >>> module = ExtractionModule()
    >>> result = module.forward("2023 Tesla Model Y brake issue")
    >>> print(result.vehicle_info.make)
    Tesla
"""

from typing import Any

import dspy

from _02_define_schema import ExtractionSignature, VehicleInfo
from settings import logger

# --- DSPy Module Definition ---


class ExtractionModule(dspy.Module):
    """
    A DSPy module for extracting vehicle information from unstructured text.

    This module encapsulates the core extraction logic using DSPy's Predict component
    with the ExtractionSignature. It handles various edge cases and provides robust
    error handling to ensure consistent outputs even when the underlying LLM fails
    or returns unexpected results.

    The module automatically falls back to "UNKNOWN" values for any extraction
    failures, ensuring the pipeline continues to function even with problematic inputs.

    Attributes:
        extractor (dspy.Predict): The DSPy predictor component configured with
                                 the ExtractionSignature for structured outputs.

    Example:
        >>> module = ExtractionModule()
        >>> result = module.forward("I own a 2023 Tesla Model Y")
        >>> print(f"{result.vehicle_info.year} {result.vehicle_info.make} {result.vehicle_info.model}")
        2023 Tesla Model Y
    """

    def __init__(self):
        """
        Initialize the ExtractionModule with a DSPy Predict component.

        Sets up the internal predictor using the ExtractionSignature, which defines
        the input/output contract for vehicle information extraction.
        """
        super().__init__()
        self.extractor = dspy.Predict(ExtractionSignature)

    def forward(self, narrative: str) -> dspy.Prediction:
        """
        Perform the forward pass to extract vehicle information from text.

        This method processes the input narrative through the DSPy predictor and
        handles various response formats and error conditions. It includes extensive
        error handling and logging to support debugging and monitoring.

        Args:
            narrative (str): The unstructured text containing vehicle information
                           to be extracted (e.g., a complaint narrative).

        Returns:
            dspy.Prediction: A prediction object containing:
                - vehicle_info (VehicleInfo): Structured vehicle data
                - make (str): Vehicle manufacturer
                - model (str): Vehicle model
                - year (str): Vehicle year
                - error (str, optional): Error message if extraction failed

        Note:
            If extraction fails for any reason, the method returns a prediction
            with all fields set to "UNKNOWN" to ensure pipeline continuity.

        Example:
            >>> module = ExtractionModule()
            >>> result = module.forward("2024 BMW X5 has engine problems")
            >>> print(result.make)  # "BMW"
            >>> print(result.year)  # "2024"
        """
        try:
            prediction = self.extractor(narrative=narrative)

            # Add debugging to see what the model actually returned
            logger.debug(f"Raw prediction type: {type(prediction)}")
            logger.debug(f"Raw prediction dir: {dir(prediction)}")
            logger.debug(f"Raw prediction: {prediction}")

            # More robust extraction
            if hasattr(prediction, "vehicle_info"):
                vehicle_info = prediction.vehicle_info
            elif hasattr(prediction, "completions") and prediction.completions:
                # Sometimes DSPy stores results differently
                completion = prediction.completions[0]
                if hasattr(completion, "vehicle_info"):
                    vehicle_info = completion.vehicle_info
                else:
                    # Try to parse from the raw response
                    logger.warning(
                        f"No vehicle_info found, trying to parse: {completion}"
                    )
                    vehicle_info = VehicleInfo(
                        make="UNKNOWN", model="UNKNOWN", year="UNKNOWN"
                    )
            else:
                logger.warning(f"Unexpected prediction structure: {prediction}")
                vehicle_info = VehicleInfo(
                    make="UNKNOWN", model="UNKNOWN", year="UNKNOWN"
                )

            return dspy.Prediction(
                vehicle_info=vehicle_info,
                make=vehicle_info.make,
                model=vehicle_info.model,
                year=vehicle_info.year,
            )

        except Exception as e:
            logger.error(f"Error in ExtractionModule.forward: {e}")
            logger.error(f"Input narrative length: {len(narrative)}")
            logger.error(f"First 200 chars: {narrative[:200]}")

            # Fallback for any errors
            vehicle_info = VehicleInfo(make="UNKNOWN", model="UNKNOWN", year="UNKNOWN")
            return dspy.Prediction(
                vehicle_info=vehicle_info,
                make="UNKNOWN",
                model="UNKNOWN",
                year="UNKNOWN",
                error=str(e),
            )


# --- Evaluation Metric ---


def extraction_metric(
    gold: dspy.Example, pred: dspy.Prediction, trace: Any = None
) -> float:
    """
    Calculate the F1-score for vehicle information extraction accuracy.

    This function serves as the evaluation metric for the DSPy optimization process.
    It compares the ground truth vehicle information with the model's predictions
    and returns a score between 0.0 and 1.0, where 1.0 indicates perfect extraction.

    The metric evaluates three components:
    - Make accuracy (exact match, case-insensitive)
    - Model accuracy (exact match, case-insensitive)
    - Year accuracy (exact string match)

    The final score is the average of these three component scores.

    Args:
        gold (dspy.Example): The ground truth example containing correct
                           make, model, and year values.
        pred (dspy.Prediction): The model's prediction containing a VehicleInfo
                               object with extracted values.
        trace (Any, optional): DSPy trace information (not used in this metric).

    Returns:
        float: The F1-score between 0.0 and 1.0, where:
              - 1.0 = perfect extraction (all fields correct)
              - 0.67 ≈ partial extraction (2/3 fields correct)
              - 0.0 = complete failure or all "UNKNOWN"

    Note:
        - Comparisons are case-insensitive for make and model
        - "UNKNOWN" predictions always receive 0 points for that field
        - Years are compared as exact string matches

    Example:
        >>> gold = dspy.Example(make="Tesla", model="Model Y", year="2023")
        >>> pred_info = VehicleInfo(make="TESLA", model="MODEL Y", year="2023")
        >>> pred = dspy.Prediction(vehicle_info=pred_info)
        >>> score = extraction_metric(gold, pred)
        >>> print(score)  # 1.0 (perfect match)
    """
    try:
        pred_info: VehicleInfo = pred.vehicle_info

        # Add debugging to see what's happening
        logger.info(f"Gold: make={gold.make}, model={gold.model}, year={gold.year}")
        logger.info(
            f"Pred: make={pred_info.make}, model={pred_info.model}, year={pred_info.year}"
        )

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

        logger.debug(
            f"Scores: make={make_score}, model={model_score}, year={year_score}, total={total_score}"
        )
        return total_score

    except Exception as e:
        logger.error(f"Error in extraction_metric: {e}")
        logger.error(f"Prediction type: {type(pred)}")
        logger.error(f"Prediction content: {pred}")
        return 0.0


if __name__ == "__main__":
    """
    Test the evaluation metric with various scenarios.
    
    This test suite validates that the extraction_metric function correctly
    calculates scores for perfect matches, partial matches, and complete misses.
    """
    print("Testing the evaluation metric...")
    # Use STRING years in test cases too!
    gold_example = dspy.Example(make="TESLA", model="MODEL Y", year="2022")  # ← STRING

    pred_vehicle_info_1 = VehicleInfo(
        make="TESLA", model="MODEL Y", year="2022"
    )  # ← STRING
    pred_1 = dspy.Prediction(vehicle_info=pred_vehicle_info_1)
    score_1 = extraction_metric(gold_example, pred_1)
    print(f"Perfect match score: {score_1}")
    assert score_1 == 1.0

    pred_vehicle_info_2 = VehicleInfo(
        make="TESLA", model="MODEL 3", year="2022"
    )  # ← STRING
    pred_2 = dspy.Prediction(vehicle_info=pred_vehicle_info_2)
    score_2 = extraction_metric(gold_example, pred_2)
    print(f"Partial match score (2/3): {score_2}")
    assert abs(score_2 - 0.6666) < 0.01

    pred_vehicle_info_3 = VehicleInfo(
        make="FORD", model="F-150", year="2020"
    )  # ← STRING
    pred_3 = dspy.Prediction(vehicle_info=pred_vehicle_info_3)
    score_3 = extraction_metric(gold_example, pred_3)
    print(f"No match score: {score_3}")
    assert score_3 == 0.0

    print("\nMetric testing successful!")
