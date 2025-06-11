# DSPy-Powered Prompt Optimization for Automotive Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-DSPy-orange" alt="Framework">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Package%20Manager-uv-purple" alt="Package Manager">
  <img src="https://img.shields.io/badge/LLM-Ollama-green" alt="LLM Runtime">
  <img src="https://img.shields.io/badge/Observability-Langfuse-blue" alt="Observability">
</p>

**Keywords**: `DSPy`, `Prompt Optimization`, `LLMs`, `Structured Data Extraction`, `Ollama`, `Langfuse`, `Streamlit`, `Enterprise AI`, `Automotive`, `NHTSA`

---

## 1. Project Overview

This project demonstrates a systematic and programmatic framework for optimizing Large Language Models (LLMs) to perform high-accuracy structured data extraction from unstructured automotive service complaints. Leveraging Stanford's **DSPy** library, this system transitions from manual "prompt engineering" to a more robust "prompt compilation" paradigm.

The core objective is to take raw, unstructured text from the official **NHTSA vehicle complaints database** and reliably extract structured entities like the vehicle's **Make, Model, and Year**. The system uses DSPy optimizers to automatically find the most effective prompt instructions and few-shot examples based on a quantitative evaluation metric, ensuring the final prompt is both highly accurate and auditable.

The entire pipeline is designed for local-first execution with **Ollama** and is deeply integrated with **Langfuse** for fine-grained observability, meeting key enterprise requirements for data privacy and traceability.

## 2. The Core Idea: From Prompt Engineering to Prompt Compilation

Manually iterating on prompts is inefficient and lacks rigor. This project adopts the DSPy philosophy: **treat LLM pipelines as programs that can be compiled and optimized.**

Instead of guessing which prompt is best, we:

1. **Define a Task Signature**: We specify the desired input (`NARRATIVE`) and output fields (`MAKE`, `MODEL`, `YEAR`) in a structured way using Pydantic models for type safety.
2. **Provide an Evaluation Metric**: We write a Python function that scores the LLM's accuracy by comparing its output to the ground truth columns in the dataset using F1-score calculations.
3. **Run an Optimizer**: We use a DSPy teleprompter (e.g., `BootstrapFewShot`) that automatically explores the solution space of prompts and few-shot examples, guided by the metric, to produce a high-performing, "compiled" prompt.

## 3. Key Features

| Feature | Description | Benefit |
| :--- | :--- | :--- |
| 🤖 **Automated Optimization** | Uses DSPy optimizers to programmatically find the best-performing prompts through systematic exploration. | Eliminates manual trial-and-error; provides a data-driven, repeatable process with measurable improvements. |
| 📊 **Quantitative Evaluation** | Employs a concrete F1-score metric to objectively measure extraction accuracy against ground truth data. | Ensures changes are measured and that the final prompt is demonstrably superior with statistical confidence. |
| 🚀 **Local & Secure** | Runs entirely on-premises with Ollama, ensuring no sensitive data is sent to external APIs. | Guarantees 100% data privacy and control, crucial for enterprise applications handling sensitive automotive data. |
| 🔍 **Fine-Grained Observability** | Integrated with Langfuse to trace every step of the optimization process with detailed logging. | Provides deep insights for debugging, auditing, and understanding the LLM's behavior throughout optimization. |
| 🧪 **Modular & Testable Workflow** | The pipeline is broken into numbered scripts with comprehensive error handling and logging. | Simplifies debugging and ensures a robust, step-by-step development process with clear failure points. |
| ✨ **Interactive Showcase** | A Streamlit application visually compares performance across all strategies with live demo capabilities. | Clearly demonstrates the value of optimization with interactive testing and performance visualization. |
| 📋 **Comprehensive Documentation** | All modules include Google-style docstrings with detailed examples and usage patterns. | Ensures maintainability and enables easy extension with new prompting strategies and evaluation metrics. |

## 4. Project Architecture & Workflow

The project is organized into a `src` directory with numbered Python scripts that represent a sequential pipeline. Each module is fully documented and independently testable for robust development.

### 4.1 Module Descriptions

| Module | Purpose | Key Components |
| :--- | :--- | :--- |
| `settings.py` | Central configuration hub | Environment setup, logging, Ollama/Langfuse integration |
| `_01_load_data.py` | Data pipeline foundation | CSV loading, cleaning, filtering redacted content, DSPy Example conversion |
| `_02_define_schema.py` | Schema and strategy definitions | Pydantic VehicleInfo model, DSPy Signature, 5 prompting strategies |
| `_03_define_program.py` | Core DSPy implementation | ExtractionModule class, F1-score metric, robust error handling |
| `_04_run_optimization.py` | Optimization orchestration | BootstrapFewShot compilation, evaluation, result persistence |
| `verify_gpu.py` | System diagnostics | PyTorch CUDA, nvidia-smi, Ollama status, resource monitoring |
| `app.py` | Interactive dashboard | Streamlit UI, performance comparison, live demo with best model |

```
dspy-automotive-extractor/
├── .gitignore
├── pyproject.toml
├── README.md
├── .env.template               # Template for environment variables
├── data/
│   └── NHTSA_complaints.csv
├── results/
│   ├── optimized_program_*.json
│   └── results_summary.json    # Central summary of all experiments
└── src/
    ├── __init__.py
    ├── settings.py             # Central configuration and settings
    ├── _01_load_data.py        # Data loading and cleaning with redacted filtering
    ├── _02_define_schema.py    # DSPy Signatures and prompting strategies
    ├── _03_define_program.py   # DSPy modules and evaluation metrics
    ├── _04_run_optimization.py # Optimization orchestration and result persistence
    ├── verify_gpu.py           # System diagnostics and GPU verification
    └── app.py                  # Streamlit dashboard for results visualization
```

## 5. Setup and Installation

These instructions assume you are using **Windows PowerShell**. The setup process includes GPU verification and comprehensive dependency management.

### Prerequisites
1. **Python**: Ensure you have Python 3.11+ installed and available in your PATH.
2. **Git**: Ensure Git is installed for cloning the repository.
3. **Ollama**: Install and run Ollama with appropriate models.
4. **NVIDIA Drivers**: For GPU acceleration (optional but recommended).

### Step 1: Clone the Repository
```powershell
git clone https://github.com/Adredes-weslee/dspy-automotive-extractor.git
cd dspy-automotive-extractor
```

### Step 2a: Download the Dataset
Create the `data` directory and download the NHTSA complaints dataset.

```powershell
mkdir data
# This command downloads the file and saves it as NHTSA_complaints.csv in the data folder
curl.exe -L -k "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADAS.csv" -o "data/NHTSA_complaints.csv"
```

### Step 2b: Download the Full 2021 Dataset
Create the `data` directory and download all 12 months of 2021 NHTSA complaints.

```powershell
mkdir data

# Download all months of 2021 (this will take a few minutes)
for ($month = 1; $month -le 12; $month++) {
    $monthStr = $month.ToString("00")
    $url = "https://static.nhtsa.gov/odi/ffdd/sgo-2021-$monthStr/SGO-2021-$monthStr" + "_Incident_Reports_ADAS.csv"
    $filename = "data/NHTSA_complaints_2021_$monthStr.csv"
    curl.exe -L -k $url -o $filename
}



### Step 3: Download Ollama Models
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

### Step 4: Setup Environment and Install Dependencies
This project uses `uv` for fast package management and includes PyTorch with CUDA support.

```powershell
# Install uv if you haven't already
pip install uv

# Create a virtual environment using uv
python -m uv venv .venv

# Activate the virtual environment
# Note: You may need to set your execution policy. In an Admin PowerShell, run: Set-ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

# Install pip in the UV environment
.\.venv\Scripts\python.exe -m ensurepip --upgrade

# Install PyTorch with CUDA support first (using the full path as in README)
.\.venv\Scripts\python.exe -m pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126

# Install UV directly in your virtual environment
.\.venv\Scripts\python.exe -m pip install uv

# Install remaining project dependencies from pyproject.toml (using Python module approach)
python -m uv pip install -e .
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

### Step 5: Configure Langfuse & Environment
1. Copy the `.env.template` file to a new file named `.env`.
    ```powershell
    copy .env.template .env
    ```
2. Follow the [Langfuse quickstart](https://langfuse.com/docs/get-started) to run the Docker container.
3. Open the new `.env` file and fill in your `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`. You can also change the `OLLAMA_MODEL` if needed.

### Step 6: Verify Your Setup
Run the comprehensive system verification script to ensure everything is configured correctly.

```powershell
.\.venv\Scripts\python.exe src/verify_gpu.py
```

This script will check:
- ✅ PyTorch CUDA availability and GPU information
- ✅ NVIDIA GPU status and memory usage  
- ✅ Ollama service status and loaded models
- ✅ System resources (CPU, RAM, disk)
- ✅ DSPy inference capabilities

## 6. Running the Pipeline

The pipeline is designed to be run step-by-step for testing or by executing the main optimization script for each strategy. Each script includes comprehensive logging and error handling.

```powershell
# Ensure your virtual environment is active before running these commands

# 1. Test data loading
.\.venv\Scripts\python.exe src/_01_load_data.py

# 2. Verify schema definitions
.\.venv\Scripts\python.exe src/_02_define_schema.py

# 3. Test the program and metric logic
.\.venv\Scripts\python.exe src/_03_define_program.py

# 4. Run the full optimization pipeline for a specific strategy
#    Replace 'cot' with any strategy name: naive, plan_and_solve, self_refine, etc.
.\.venv\Scripts\python.exe src/_04_run_optimization.py cot
```

### 6.1 Optimization Process Details
Each optimization run includes:
- **Data Splitting**: 90% training, 10% validation with stratified sampling
- **BootstrapFewShot**: Automatically selects optimal few-shot examples
- **Evaluation**: F1-score calculation across make, model, and year fields
- **Result Persistence**: Saves optimized program and updates central summary
- **Langfuse Tracking**: Complete observability with trace URLs for debugging

**Expected runtime**: 10-30 minutes per strategy depending on your hardware and model size.

After running `_04_run_optimization.py` for each strategy, the `results/` folder will contain the compiled programs and the `results_summary.json`.

## 7. Results Analysis and Visualization

### 7.1 Streamlit Dashboard
Once you have run at least one optimization experiment, launch the interactive dashboard.

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app.py
```

The dashboard provides:
- 📊 **Performance Comparison Table**: F1-scores and timestamps for all strategies
- 🔗 **Langfuse Integration**: Direct links to detailed optimization traces
- 🧪 **Live Demo**: Test the best-performing model on custom automotive complaints
- 📈 **Strategy Analysis**: Compare different prompting approaches side-by-side

This will open a web browser showing a dashboard of your experiment results, with links to the Langfuse traces and a live demo area.

### 7.2 Results Summary
The `results_summary.json` file contains a comprehensive summary of all experiments:

```json
{
  "naive": {
    "f1_score": 0.427,
    "timestamp": "2025-06-11T10:30:45Z",
    "trace_url": "https://localhost:3000/trace/...",
    "optimization_time_minutes": 8.5
  },
  "cot": {
    "f1_score": 0.427,
    "timestamp": "2025-06-11T11:15:22Z", 
    "trace_url": "https://localhost:3000/trace/...",
    "optimization_time_minutes": 12.3
  }
}
```

## 8. Experimenting with Prompting Techniques

This framework is designed for systematic experimentation with different prompting approaches. Each strategy is implemented using the Strategy Pattern for easy extension and comparison.

| # | Strategy Name | Approach | Expected Use Case |
| :--- | :--- | :--- | :--- |
| 1 | `naive` | Direct extraction instructions with examples | Baseline comparison and simple use cases |
| 2 | `cot` | Chain of Thought step-by-step reasoning | Complex extractions requiring systematic analysis |
| 3 | `plan_and_solve` | Planning phase followed by execution | Multi-step problems requiring strategic approach |
| 4 | `self_refine` | Draft → Critique → Refine cycle | High-accuracy scenarios requiring self-correction |
| 5 | `contrastive_cot` | Good vs bad reasoning examples | Training scenarios with common failure modes |

### 8.1 Strategy Implementation Examples

**Chain of Thought (CoT):**
```
"Let's think step by step. First, identify the vehicle's make by looking for manufacturer names like Toyota, Ford, etc. Second, identify the model by finding the specific vehicle name. Third, find the model year, usually a 4-digit number. Finally, provide the structured output."
```

**Self-Refine:**
```
"Generate a draft extraction of the vehicle's make, model, and year. Then, critique your draft for accuracy and completeness. Finally, based on your critique, provide a final, refined structured answer."
```

### 8.2 Adding Custom Strategies
To create a new prompting strategy:

1. **Define the Strategy Class** in `_02_define_schema.py`:
```python
class CustomStrategy(PromptStrategy):
    def get_instructions(self) -> str:
        return "Your custom instruction here..."
```

2. **Register in the Strategy Dictionary**:
```python
PROMPT_STRATEGIES["custom"] = CustomStrategy()
```

3. **Run the Optimization**:
```powershell
.\.venv\Scripts\python.exe src/_04_run_optimization.py custom
```

## 9. Troubleshooting and Common Issues

### 9.1 GPU and CUDA Issues
If you encounter GPU-related problems:

```powershell
# Check CUDA availability
.\.venv\Scripts\python.exe -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Verify GPU status
nvidia-smi

# Test Ollama GPU usage
ollama run gemma3:12b "Test message"
```

### 9.2 Ollama Connection Issues
- **Ensure Ollama is running**: `ollama serve`
- **Verify model availability**: `ollama list`
- **Check model loading**: `ollama run gemma3:12b`

### 9.3 Memory Issues
For systems with limited RAM:
- Use smaller models: `qwen3:4b` instead of `gemma3:12b`
- Reduce sample size in `_01_load_data.py`
- Monitor system resources during optimization

### 9.4 Langfuse Integration
- Verify Docker container is running
- Check environment variables in `.env`
- Confirm network connectivity to `localhost:3000`

## 10. Performance Expectations

Based on testing with the NHTSA dataset:

| Strategy | Expected F1-Score Range | Runtime (GPU) | Runtime (CPU) |
| :--- | :--- | :--- | :--- |
| Naive | 0.35 - 0.45 | 5-10 min | 20-30 min |
| Chain of Thought | 0.45 - 0.55 | 10-15 min | 30-45 min |
| Plan and Solve | 0.40 - 0.50 | 10-15 min | 30-45 min |
| Self Refine | 0.50 - 0.60 | 15-25 min | 45-60 min |
| Contrastive CoT | 0.45 - 0.55 | 10-15 min | 30-45 min |

*Performance varies based on hardware specifications and model choice.*

## 11. Future Extensions

This framework is designed for extensibility:

- **New Domains**: Adapt for medical, legal, or financial document extraction
- **Advanced Metrics**: Implement domain-specific evaluation metrics
- **Multi-Modal**: Extend to handle images, PDFs, or audio transcripts
- **Production Deployment**: Add API endpoints and model serving capabilities
- **Advanced Optimizers**: Experiment with other DSPy teleprompters like MIPRO or COPRO

## 12. License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📚 Additional Resources

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Ollama Model Library](https://ollama.ai/library)
- [NHTSA Data Portal](https://www.nhtsa.gov/nhtsa-datasets-and-apis)

**Happy optimizing! 🚀**
