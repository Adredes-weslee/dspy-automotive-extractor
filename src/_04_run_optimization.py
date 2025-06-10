"""
_04_run_optimization.py

This is the main orchestration script for the DSPy optimization pipeline.
It brings together all the components from the previous scripts to:
1.  Set up the environment, including the LLM and Langfuse for tracing.
2.  Load and split the dataset into training and validation sets.
3.  Allow the user to select a prompting strategy via a command-line argument.
4.  Initialize the DSPy optimizer (`BootstrapFewShot`).
5.  Run the compilation process to find the best prompt/demonstrations.
6.  Save the optimized program to a JSON file in the `results/` directory.
"""
import os
import sys
import dspy
import argparse
from pathlib import Path
from dotenv import load_dotenv

from langfuse.callback import DspyCallbackHandler

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import our custom modules
from src._01_load_data import get_dataset
from src._02_define_schema import ExtractionSignature, get_strategy, PROMPT_STRATEGIES
from src._03_define_program import ExtractionModule, extraction_metric
from src.settings import RESULTS_DIR, logger

def setup_environment():
    """Sets up the DSPy environment with Ollama and Langfuse."""
    load_dotenv() # Load environment variables from .env file

    # Initialize the Ollama-based language model
    # Model name can be configured via an environment variable
    model_name = os.getenv("OLLAMA_MODEL", "gemma3:12b")
    llm = dspy.OllamaLocal(model=model_name, max_tokens=2048)
    dspy.settings.configure(lm=llm)
    logger.info(f"DSPy configured with Ollama model: {model_name}")

    # Initialize Langfuse for tracing if keys are provided
    if all(k in os.environ for k in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]):
        logger.info("Langfuse credentials found, initializing tracker.")
        langfuse_handler = DspyCallbackHandler()
        return langfuse_handler
    else:
        logger.warning("Langfuse credentials not found. Skipping tracing.")
        return None

def main(strategy_name: str):
    """
    Main function to run the optimization pipeline for a given strategy.

    Args:
        strategy_name (str): The name of the prompting strategy to use for optimization.
    """
    logger.info(f"--- Starting optimization for strategy: {strategy_name} ---")

    # 1. Setup Environment
    langfuse_handler = setup_environment()
    callbacks = [langfuse_handler] if langfuse_handler else []

    # 2. Load and Prepare Data
    dataset = get_dataset()
    trainset = [x for x in dataset if x not in dataset[0:30]] # Use first 30 for dev
    devset = dataset[0:30]
    logger.info(f"Loaded dataset: {len(trainset)} training examples, {len(devset)} validation examples.")

    # 3. Apply the selected prompting strategy
    strategy = get_strategy(strategy_name)
    ExtractionSignature.__doc__ = strategy.get_docstring()
    logger.info("Applied prompting strategy to signature.")
    print(f"Using instruction:\n---\n{ExtractionSignature.__doc__}\n---")

    # 4. Initialize the DSPy Program and Optimizer
    program = ExtractionModule()
    optimizer = dspy.teleprompt.BootstrapFewShot(metric=extraction_metric, max_bootstrapped_demos=2)

    # 5. Run the Optimization (Compilation)
    logger.info("Starting DSPy compilation... This may take a while.")
    optimized_program = optimizer.compile(
        student=program,
        trainset=trainset,
        valset=devset,
        callbacks=callbacks
    )
    logger.info("Compilation complete.")

    # 6. Save the Optimized Program
    output_filename = f"optimized_{strategy_name}.json"
    output_path = RESULTS_DIR / output_filename
    RESULTS_DIR.mkdir(exist_ok=True)
    optimized_program.save(str(output_path))
    logger.info(f"Optimized program saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DSPy prompt optimization for a given strategy.")
    parser.add_argument(
        "strategy",
        type=str,
        choices=PROMPT_STRATEGIES.keys(),
        help="The prompting strategy to use for optimization."
    )
    args = parser.parse_args()
    main(args.strategy)