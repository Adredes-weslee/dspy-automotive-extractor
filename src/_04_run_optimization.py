"""
_04_run_optimization.py

This is the main orchestration script for the DSPy optimization pipeline.
It brings together all components to:
1.  Set up the environment using settings.py.
2.  Load and split the dataset.
3.  Allow the user to select a prompting strategy via command-line argument.
4.  Run the DSPy optimizer.
5.  Capture the evaluation score and Langfuse trace URL.
6.  Save the optimized program to a unique JSON file.
7.  Update a central `results_summary.json` with the outcome of the experiment.
"""

import argparse
import gc
import json
import sys
from datetime import datetime

import dspy
from dspy.teleprompt import BootstrapFewShot
from sklearn.model_selection import train_test_split

from _01_load_data import get_dataset
from _02_define_schema import PROMPT_STRATEGIES, ExtractionSignature, get_strategy
from _03_define_program import ExtractionModule, extraction_metric

# Use settings from the central settings.py module
from settings import RESULTS_DIR, logger, setup_environment


def update_results_summary(
    strategy_name: str, score: float, trace_url: str, optimized_path: str
):
    """Reads, updates, and writes the central results summary JSON."""
    summary_path = RESULTS_DIR / "results_summary.json"
    summary_data = {}
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary_data = json.load(f)

    summary_data[strategy_name] = {
        "final_score": round(score, 3),
        "trace_url": trace_url,
        "program_path": optimized_path,
        "timestamp": datetime.now().isoformat(),
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Updated results summary at {summary_path}")


def main(strategy_name: str):
    """Main function to run the optimization pipeline for a given strategy."""
    logger.info(f"--- Starting optimization for strategy: {strategy_name} ---")

    langfuse_handler = setup_environment()

    dataset = get_dataset()

    # Use sklearn's train_test_split (works perfectly with lists)
    trainset, devset = train_test_split(
        dataset, train_size=0.9, random_state=42, shuffle=True
    )

    logger.info(
        f"Loaded dataset: {len(trainset)} training, {len(devset)} validation examples."
    )

    strategy = get_strategy(strategy_name)
    ExtractionSignature.__doc__ = strategy.get_docstring()
    logger.info("Applied prompting strategy to signature.")

    program = ExtractionModule()
    optimizer = BootstrapFewShot(
        metric=extraction_metric, max_bootstrapped_demos=4, max_labeled_demos=16
    )

    logger.info("Starting DSPy compilation... This may take a while.")
    optimized_program = optimizer.compile(
        student=program,
        trainset=trainset,
    )
    logger.info("Compilation complete.")

    # Evaluate the optimized program on the dev set to get a final score
    evaluator = dspy.Evaluate(
        devset=devset, metric=extraction_metric, num_threads=4, display_progress=True
    )
    final_score = evaluator(optimized_program)
    logger.info(f"Final evaluation score for '{strategy_name}': {final_score:.3f}")

    # Capture Langfuse trace URL
    try:
        trace_url = langfuse_handler.get_trace_url() if langfuse_handler else "N/A"
    except Exception as e:
        logger.warning(f"Could not get trace URL: {e}")
        trace_url = "http://localhost:3000"  # Default Langfuse URL
    logger.info(f"Langfuse Trace URL: {trace_url}")

    # Save the optimized program and update summary
    output_filename = f"optimized_{strategy_name}.json"
    output_path = RESULTS_DIR / output_filename
    optimized_program.save(str(output_path))
    logger.info(f"Optimized program saved to: {output_path}")

    update_results_summary(strategy_name, final_score, trace_url, str(output_path))

    # NEW: Add explicit cleanup and exit
    logger.info(f"--- Optimization complete for strategy: {strategy_name} ---")
    logger.info("Script finished successfully.")

    # Force cleanup
    gc.collect()

    # Explicit exit to prevent background processes
    logger.info("Exiting cleanly...")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DSPy prompt optimization.")
    parser.add_argument(
        "strategy",
        type=str,
        choices=list(PROMPT_STRATEGIES.keys()),
        help="The prompting strategy to use.",
    )
    args = parser.parse_args()

    try:
        main(args.strategy)
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)
