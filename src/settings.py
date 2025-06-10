"""
settings.py

This module centralizes all project configurations. It handles:
1.  Defining constant paths for data, results, and source directories using pathlib.
2.  Loading environment variables from a .env file using python-dotenv.
3.  Setting up a standardized logger for consistent logging across all modules.
"""
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import litellm
import dspy
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
    """Set up DSPy with Ollama and configure Langfuse tracking."""
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
                secret_key=secret,
                public_key=public,
                host=host
            )
            
            logger.info("Langfuse credentials found, initializing tracker.")
        else:
            logger.warning("Langfuse credentials not found, proceeding without tracking.")
            
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        
    return langfuse_handler

def configure_litellm_callbacks() -> None:
    """Configure LiteLLM to use Langfuse as a callback for logging model interactions."""
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"] 
    logger.info("Configured LiteLLM to use Langfuse as a callback.")