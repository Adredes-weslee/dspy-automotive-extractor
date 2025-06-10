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

