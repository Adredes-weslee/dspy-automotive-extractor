"""
settings.py

This module centralizes all project configurations and environment setup for the
DSPy automotive extractor pipeline. It handles:

1.  Defining constant paths for data, results, and source directories using pathlib.
2.  Loading environment variables from a .env file using python-dotenv.
3.  Setting up a standardized logger for consistent logging across all modules.
4.  Configuring DSPy with Ollama models for LLM interactions.
5.  Initializing Langfuse tracking for observability and debugging.

This module ensures that all project components have access to consistent
configuration and logging infrastructure.

Usage:
    >>> from settings import logger, setup_environment, DATA_DIR, RESULTS_DIR
    >>> langfuse_handler = setup_environment()
    >>> logger.info("Pipeline started")

Example:
    .\.venv\Scripts\python.exe -c "from settings import *; setup_environment()"
"""

import logging
import os
import sys
from pathlib import Path

import dspy
import litellm
from dotenv import load_dotenv
from langfuse.callback import CallbackHandler

# --- Project Root ---
# This ensures that all paths are relative to the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# --- Directory Paths ---
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

# --- Environment Variable Loading ---
# Load the .env file from the project root
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    print(f"Warning: .env file not found at {env_path}. Some features may not work.")


# --- Logging Configuration ---
# Create a logger instance that can be imported and used throughout the project
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Create required directories if they don't exist ---
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def setup_environment():
    """
    Set up the complete environment for DSPy with Ollama and configure Langfuse tracking.

    This function performs the essential initialization steps for the entire pipeline:
    1. Loads environment variables from .env file
    2. Configures DSPy with the specified Ollama model
    3. Sets up Langfuse tracking for observability (if credentials are available)
    4. Configures LiteLLM callbacks for comprehensive logging

    The function is designed to be fault-tolerant - if Langfuse credentials are
    missing or invalid, the pipeline will continue without tracking functionality.

    Returns:
        CallbackHandler | None: A Langfuse callback handler instance if successful,
                               None if Langfuse setup failed or credentials missing.

    Environment Variables:
        OLLAMA_MODEL (str): The Ollama model to use (default: "gemma3:12b")
        LANGFUSE_SECRET_KEY (str): Secret key for Langfuse authentication
        LANGFUSE_PUBLIC_KEY (str): Public key for Langfuse authentication
        LANGFUSE_HOST (str): Langfuse host URL for tracking

    Side Effects:
        - Configures DSPy global settings with the specified LLM
        - Sets up LiteLLM callback functions for logging
        - Creates necessary project directories
        - Logs configuration status and any errors

    Example:
        >>> langfuse_handler = setup_environment()
        DSPy configured with Ollama model: gemma3:12b
        Langfuse credentials found, initializing tracker.
        >>> # Pipeline is now ready for use

    Note:
        This function should be called once at the beginning of any script
        that uses the DSPy pipeline components.
    """
    # Load environment variables
    load_dotenv()

    # Configure DSPy with Ollama
    model_name = os.getenv("OLLAMA_MODEL", "gemma3:12b")
    llm = dspy.LM(model=f"ollama/{model_name}")
    dspy.settings.configure(lm=llm)
    logger.info(f"DSPy configured with Ollama model: {model_name}")

    # Set up Langfuse tracking
    langfuse_handler = None
    try:
        secret = os.getenv("LANGFUSE_SECRET_KEY")
        public = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST")

        if all([secret, public, host]):
            # Configure LiteLLM callbacks for Langfuse
            configure_litellm_callbacks()

            # Create CallbackHandler for trace URLs
            langfuse_handler = CallbackHandler(
                secret_key=secret, public_key=public, host=host
            )

            logger.info("Langfuse credentials found, initializing tracker.")
        else:
            logger.warning(
                "Langfuse credentials not found, proceeding without tracking."
            )

    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")

    return langfuse_handler


def configure_litellm_callbacks() -> None:
    """
    Configure LiteLLM to use Langfuse as a callback for logging model interactions.

    This function sets up comprehensive logging of all LLM interactions through
    LiteLLM's callback system. It ensures that both successful and failed model
    calls are captured in Langfuse for observability and debugging.

    The callbacks capture:
    - Model requests and responses
    - Token usage and costs
    - Latency metrics
    - Error details for failed calls

    Returns:
        None

    Side Effects:
        - Modifies LiteLLM global callback settings
        - Enables automatic logging of all subsequent LLM calls
        - Logs the configuration status

    Example:
        >>> configure_litellm_callbacks()
        Configured LiteLLM to use Langfuse as a callback.
        >>> # All subsequent LLM calls will be logged to Langfuse

    Note:
        This function is automatically called by setup_environment() when
        Langfuse credentials are available. It should not be called directly
        unless you're setting up custom callback configurations.
    """
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]
    logger.info("Configured LiteLLM to use Langfuse as a callback.")
