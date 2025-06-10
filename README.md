# DSPy-Powered Prompt Optimization for Automotive Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-DSPy-orange" alt="Framework">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Package%20Manager-uv-purple" alt="Package Manager">
</p>

**Keywords**: `DSPy`, `Prompt Optimization`, `LLMs`, `Structured Data Extraction`, `Ollama`, `Langfuse`, `Streamlit`, `Enterprise AI`, `Automotive`, `NHTSA`

---

### 1. Project Overview

This project demonstrates a systematic and programmatic framework for optimizing Large Language Models (LLMs) to perform high-accuracy structured data extraction from unstructured automotive service complaints. Leveraging Stanford's **DSPy** library, this system transitions from manual "prompt engineering" to a more robust "prompt compilation" paradigm.

The core objective is to take raw, unstructured text from the official **NHTSA vehicle complaints database** and reliably extract structured entities like the vehicle's **Make, Model, and Year**. The system uses DSPy optimizers to automatically find the most effective prompt instructions and few-shot examples based on a quantitative evaluation metric, ensuring the final prompt is both highly accurate and auditable.

The entire pipeline is designed for local-first execution with **Ollama** and is deeply integrated with **Langfuse** for fine-grained observability, meeting key enterprise requirements for data privacy and traceability.

### 2. The Core Idea: From Prompt Engineering to Prompt Compilation

Manually iterating on prompts is inefficient and lacks rigor. This project adopts the DSPy philosophy: **treat LLM pipelines as programs that can be compiled and optimized.**

Instead of guessing which prompt is best, we:
1.  **Define a Task Signature**: We specify the desired input (`NARRATIVE`) and output fields (`MAKE`, `MODEL`, `YEAR`) in a structured way.
2.  **Provide an Evaluation Metric**: We write a Python function that scores the LLM's accuracy by comparing its output to the ground truth columns in the dataset.
3.  **Run an Optimizer**: We use a DSPy teleprompter (e.g., `BootstrapFewShot`) that automatically explores the solution space of prompts and few-shot examples, guided by the metric, to produce a high-performing, "compiled" prompt.

### 3. Key Features

| Feature | Description | Benefit |
| :--- | :--- | :--- |
| 🤖 **Automated Optimization** | Uses DSPy optimizers to programmatically find the best-performing prompts. | Eliminates manual trial-and-error; provides a data-driven, repeatable process. |
| 📊 **Quantitative Evaluation** | Employs a concrete F1-score metric to objectively measure extraction accuracy against ground truth. | Ensures changes are measured and that the final prompt is demonstrably superior. |
| 🚀 **Local & Secure** | Runs entirely on-premises with Ollama, ensuring no sensitive data is sent to external APIs. | Guarantees 100% data privacy and control, crucial for enterprise applications. |
| 🔍 **Fine-Grained Observability** | Integrated with Langfuse to trace every step of the optimization process. | Provides deep insights for debugging, auditing, and understanding the LLM's behavior. |
| 🧪 **Modular & Testable Workflow** | The pipeline is broken into numbered scripts, allowing each step to be run and validated independently. | Simplifies debugging and ensures a robust, step-by-step development process. |
| ✨ **Interactive Showcase** | A Streamlit application visually compares the performance of a naive prompt vs. the final optimized prompt. | Clearly and effectively demonstrates the value of the optimization process to any audience. |

### 4. Project Architecture & Workflow

The project is organized into a `src` directory with numbered Python scripts that represent a sequential pipeline. This modular structure allows for easy testing and debugging of each component.

```
dspy-automotive-extractor/
├── .gitignore
├── pyproject.toml
├── README.md
├── .env.template               # New: Template for environment variables
├── data/
│   └── NHTSA_complaints.csv
├── results/
│   ├── optimized_program_cot.json
│   └── results_summary.json    # New: Central summary of all experiments
└── src/
    ├── __init__.py
    ├── settings.py             # New: Central configuration and settings
    ├── _01_load_data.py        # Handles loading and preparing the dataset
    ├── _02_define_schema.py    # Defines the DSPy Signature and Prompt Strategies
    ├── _03_define_program.py   # Defines the DSPy Module and the evaluation metric
    ├── _04_run_optimization.py # The main script to run the optimizer for a given strategy
    └── app.py                  # The Streamlit dashboard for showcasing results
```

### 5. Setup and Installation

These instructions assume you are using **Windows PowerShell**.

#### **Prerequisites**
1.  **Python**: Ensure you have Python 3.11+ installed and available in your PATH.
2.  **Git**: Ensure Git is installed for cloning the repository.
3.  **Ollama**: Install and run Ollama.

#### **Step 1: Clone the Repository**
```powershell
git clone https://github.com/Adredes-weslee/dspy-automotive-extractor.git
cd dspy-automotive-extractor
```

#### **Step 2: Download the Dataset**
Create the `data` directory and then run the command to download the dataset.

```powershell
mkdir data
# This command downloads the file and saves it as NHTSA_complaints.csv in the data folder
curl.exe -L -k "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADAS.csv" -o "data/NHTSA_complaints.csv"
```

#### **Step 3: Download Ollama Models**
This project supports multiple models for different hardware capabilities.

* **For high-performance GPUs (>= 8GB VRAM):**
    ```powershell
    ollama pull gemma3:12b
    ```
* **For CPU-only or lower-end GPUs (fallback):**
    ```powershell
    ollama pull qwen3:4b
    ```
The scripts will default to `gemma3:12b` but can be easily configured using the `.env` file.

#### **Step 4: Setup Environment and Install Dependencies with `uv`**
This project uses `uv` for fast package management.

```powershell
# Install uv if you haven't already
pip install uv

# Create a virtual environment using uv
uv venv

# Activate the virtual environment
# Note: You may need to set your execution policy. In an Admin PowerShell, run: Set-ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support first (using pip for better compatibility)
pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126

# Install remaining project dependencies from pyproject.toml
uv pip install -e .
```

*(Optional) For Bash/Linux users:*
```bash
# Activate the virtual environment
source .venv/bin/activate

# Install PyTorch with CUDA support first
pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126 

# Install remaining project dependencies
uv pip install -e .
```

#### **Step 5: Configure Langfuse & Environment**
1.  Copy the `.env.template` file to a new file named `.env`.
    ```powershell
    copy .env.template .env
    ```
2.  Follow the [Langfuse quickstart](https://langfuse.com/docs/get-started) to run the Docker container.
3.  Open the new `.env` file and fill in your `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`. You can also change the `OLLAMA_MODEL` if needed.

### 6. Running the Pipeline

The pipeline is designed to be run step-by-step for testing or by executing the main optimization script for each strategy.

```powershell
# Ensure your virtual environment is active before running these commands

# 1. Test data loading
python src/_01_load_data.py

# 2. Verify schema definitions
python src/_02_define_schema.py

# 3. Test the program and metric logic
python src/_03_define_program.py

# 4. Run the full optimization pipeline for a specific strategy
#    Replace 'cot' with any strategy name: naive, plan_and_solve, self_refine, etc.
python src/_04_run_optimization.py cot
```

After running `_04_run_optimization.py` for each strategy, the `results/` folder will contain the compiled programs and the `results_summary.json`.

### 7. Showcasing the Results

Once you have run at least one optimization experiment, launch the Streamlit dashboard.

```powershell
streamlit run src/app.py
```
This will open a web browser showing a dashboard of your experiment results, with links to the Langfuse traces and a live demo area.

### 8. Experimenting with Prompting Techniques

This framework is designed for experimentation. To test the pre-configured prompting techniques, simply run the optimization script with the desired strategy name.

| # | Strategy Name (`<strategy>`) | Example Instruction for Extracting Make, Model, and Year |
| :--- | :--- | :--- |
| 1 | `naive` | "Extract the vehicle make, model, and year from the text." |
| 2 | `cot` | "Let's think step by step. First, identify the vehicle's make. Second, identify its model. Third, find the model year. Finally, provide the structured output." |
| 3 | `plan_and_solve` | "First, devise a plan to extract the vehicle's make, model, and year. Then, execute the plan, detailing each step of the extraction to arrive at the final answer." |
| 4 | `self_refine` | "Generate a draft extraction of the vehicle's make, model, and year. Then, critique your draft for accuracy and completeness. Finally, based on your critique, provide a final, refined structured answer." |
| 5 | `contrastive_cot` | "To extract the vehicle's make, model, and year, you must reason correctly. A good example of reasoning is: 'The text mentions a 2022 Tesla Model Y. Therefore, the make is Tesla, the model is Model Y, and the year is 2022.' A bad example is: 'The text mentions a steering wheel, so the make is car.' Now, analyze the following text." |

To add a new technique, simply create a new class in `src/_02_define_schema.py` following the Strategy pattern, add it to the `PROMPT_STRATEGIES` dictionary, and then run the optimization script with your new strategy's name.

### 9. License

This project is licensed under the MIT License.
