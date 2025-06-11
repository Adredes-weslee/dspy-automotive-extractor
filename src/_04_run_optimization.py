r"""
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

This script serves as the entry point for running optimization experiments with
different prompting strategies. It orchestrates the entire pipeline from data
loading to model optimization and results storage.

Usage:
    .\.venv\Scripts\python.exe src\_04_run_optimization.py <strategy_name>

Example:
    .\.venv\Scripts\python.exe src\_04_run_optimization.py naive
    .\.venv\Scripts\python.exe src\_04_run_optimization.py cot
"""

import argparse
import gc
import json
import sys
import threading
import time
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
) -> None:
    """
    Updates the central results summary JSON file with optimization results.

    This function maintains a persistent record of all optimization experiments
    by reading the existing results summary, adding the new result, and writing
    it back to disk. Each entry includes the strategy name, final score, trace
    URL for debugging, and the path to the optimized model.

    Args:
        strategy_name (str): The name of the prompting strategy that was optimized
                           (e.g., 'naive', 'chain_of_thought', 'plan_and_solve').
        score (float): The final evaluation score achieved by the optimized model,
                      typically a value between 0.0 and 1.0 representing accuracy.
        trace_url (str): The Langfuse trace URL for debugging and analysis,
                        or "N/A" if not available.
        optimized_path (str): The file path where the optimized DSPy program
                             was saved (e.g., 'results/optimized_naive.json').

    Returns:
        None

    Side Effects:
        - Creates or updates 'results/results_summary.json'
        - Logs the update operation

    Example:
        >>> update_results_summary(
        ...     strategy_name="naive",
        ...     score=0.427,
        ...     trace_url="http://localhost:3000/trace/123",
        ...     optimized_path="results/optimized_naive.json"
        ... )
        # Creates/updates results_summary.json with the new entry
    """
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


def main(strategy_name: str) -> None:
    """
    Main orchestration function that runs the complete DSPy optimization pipeline.

    This function coordinates all aspects of the optimization process:
    1. Sets up the DSPy environment and Langfuse logging
    2. Loads and splits the vehicle complaints dataset
    3. Applies the specified prompting strategy to the extraction signature
    4. Runs DSPy's BootstrapFewShot optimizer to improve the model
    5. Evaluates the optimized model on a held-out validation set
    6. Saves the optimized program and updates the results summary

    Args:
        strategy_name (str): The name of the prompting strategy to use for
                           optimization. Must be one of the strategies defined
                           in PROMPT_STRATEGIES (e.g., 'naive', 'chain_of_thought',
                           'plan_and_solve', 'self_refine', 'contrastive_cot').

    Returns:
        None

    Raises:
        SystemExit: Always exits with code 0 on successful completion, or code 1
                   on failure. This prevents background processes from continuing.

    Side Effects:
        - Configures DSPy with the specified LLM and settings
        - Creates optimized model files in the results directory
        - Updates the central results summary JSON file
        - Logs detailed progress information throughout the process

    Example:
        >>> main("naive")
        # Runs optimization with naive prompting strategy
        # Outputs: Final evaluation score, saved model path, results summary

    Note:
        This function includes explicit cleanup and exit procedures to prevent
        DSPy background processes from continuing after optimization completes.
    """
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

    logger.info(f"--- Optimization complete for strategy: {strategy_name} ---")
    logger.info("Script finished successfully.")

    # Enhanced thread cleanup
    logger.info("Waiting for background threads to complete...")

    # Step 1: Wait for active threads
    active_threads = threading.active_count()
    max_wait = 60  # Increased timeout
    wait_time = 0

    while active_threads > 1 and wait_time < max_wait:
        time.sleep(2)  # Longer intervals
        wait_time += 2
        current_threads = threading.active_count()
        if current_threads != active_threads:
            logger.info(f"Active threads: {current_threads}")
            active_threads = current_threads

    # Step 2: Force LiteLLM cleanup
    try:
        import litellm

        # Clear any pending callbacks
        litellm._async_success_callback = []
        litellm._success_callback = []
        logger.info("Cleared LiteLLM callbacks")
    except Exception as e:
        logger.warning(f"LiteLLM cleanup warning: {e}")

    # Step 3: Final thread check
    final_threads = threading.active_count()
    logger.info(f"Final thread count: {final_threads}")

    logger.info("Background cleanup complete")

    # Force cleanup
    gc.collect()

    # Clean exit
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
