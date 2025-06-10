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
├── pyproject.toml              # Project dependencies for `uv`
├── README.md                   # This file
├── data/
│   └── NHTSA_complaints.csv    # The dataset used for training and evaluation
├── results/
│   └── optimized_program.json  # The output of the optimization process
└── src/
    ├── __init__.py
    ├── _01_load_data.py        # Handles loading and preparing the dataset
    ├── _02_define_schema.py    # Defines the DSPy Signature and Pydantic models
    ├── _03_define_program.py   # Defines the DSPy Module and the evaluation metric
    ├── _04_run_optimization.py # The main script to run the optimizer and save the results
    └── app.py                  # The Streamlit application for showcasing results
```

### 5. Setup and Installation

These instructions assume you are using **Windows PowerShell**.

#### **Prerequisites**
1.  **Python**: Ensure you have Python 3.11+ installed and available in your PATH.
2.  **Git**: Ensure Git is installed for cloning the repository.
3.  **Ollama**: Install and run Ollama.

#### **Step 1: Clone the Repository**
```powershell
git clone [https://github.com/Adredes-weslee/dspy-automotive-extractor.git](https://github.com/Adredes-weslee/dspy-automotive-extractor.git)
cd dspy-automotive-extractor
```

#### **Step 2: Download the Dataset**
The script will download a complaint dataset from the NHTSA. Create the `data` directory and then run the command.

```powershell
mkdir data
# This command downloads the file and saves it as NHTSA_complaints.csv in the data folder
curl.exe -L -k "[https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADAS.csv](https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADAS.csv)" -o "data/NHTSA_complaints.csv"
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
The scripts will default to `gemma3:12b` but can be easily configured to use the smaller model.

#### **Step 4: Setup Environment and Install Dependencies with `uv`**

This project uses `uv`, a high-performance package manager. We will use it to create a virtual environment and install the dependencies listed in `pyproject.toml`.

**First, install uv using one of these methods:**

**Option A: Using pipx (recommended):**
```powershell
pip install pipx
pipx install uv
```

**Option B: Using pip:**
```powershell
pip install uv
```

**Then set up the project:**

```powershell
# Create a virtual environment using uv
uv venv

# Activate the virtual environment
# Note: You may need to set your execution policy. In an Admin PowerShell, run: Set-ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support first (using pip for better compatibility)
pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --extra-index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)

# Install remaining project dependencies from pyproject.toml
uv pip install -e .
```

*(Optional) For Bash/Linux users:*
```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

#### **Step 5: Configure Langfuse**
Langfuse is used for observability. You can run it locally via Docker.

1.  Follow the [Langfuse quickstart](https://langfuse.com/docs/get-started) to run the Docker container.
2.  Set the following environment variables in your PowerShell session.

```powershell
$env:LANGFUSE_PUBLIC_KEY="pk-lf-..."
$env:LANGFUSE_SECRET_KEY="sk-lf-..."
$env:LANGFUSE_HOST="http://localhost:3000"
```

### 6. Running the Pipeline

The pipeline is designed to be run step-by-step to ensure each part is working correctly.

```powershell
# Ensure your virtual environment is active before running these commands

# 1. Test data loading and processing. This will also show you a sample of the data.
python src/_01_load_data.py

# 2. Verify the DSPy Signature and Pydantic models (prints the schema).
python src/_02_define_schema.py

# 3. Verify the DSPy Program and evaluation metric logic.
python src/_03_define_program.py

# 4. Run the full optimization pipeline. This is computationally intensive.
python src/_04_run_optimization.py
```

After running `_04_run_optimization.py`, the best-found prompt program will be saved to `results/optimized_program.json`.

### 7. Showcasing the Results

Once the optimization is complete, launch the Streamlit application to see the comparison.

```powershell
streamlit run src/app.py
```
This will open a web browser where you can input a sample service complaint and see the structured output from both a naive prompt and your highly optimized DSPy program side-by-side.

### 8. Experimenting with Prompting Techniques

This framework is designed for experimentation. To test different prompting techniques from your resume, you simply modify the docstring of the `Signature` class in `src/_02_define_schema.py` and re-run the optimization.

Here are the initial techniques to test, progressing from simple to complex:

| # | Technique Category | Example Instruction for Extracting Make, Model, and Year |
| :--- | :--- | :--- |
| 1 | **Naive Prompt** | "Extract the vehicle make, model, and year from the text." |
| 2 | **Thought Generation (CoT)** | "Let's think step by step. First, identify the vehicle's make. Second, identify its model. Third, find the model year. Finally, provide the structured output." |
| 3 | **Decomposition (Plan-and-Solve)** | "First, devise a plan to extract the vehicle's make, model, and year. Then, execute the plan, detailing each step of the extraction to arrive at the final answer." |
| 4 | **Self-Criticism (Self-Refine)** | "Generate a draft extraction of the vehicle's make, model, and year. Then, critique your draft for accuracy and completeness. Finally, based on your critique, provide a final, refined structured answer." |
| 5 | **Contrastive CoT** | "To extract the vehicle's make, model, and year, you must reason correctly. A good example of reasoning is: 'The text mentions a 2022 Tesla Model Y. Therefore, the make is Tesla, the model is Model Y, and the year is 2022.' A bad example is: 'The text mentions a steering wheel, so the make is car.' Now, analyze the following text." |

By running the optimization for each of these and saving the results, the Streamlit app can be extended to compare all of them, creating a powerful visualization that directly validates the skills highlighted on your resume.

### 9. License

This project is licensed under the MIT License.