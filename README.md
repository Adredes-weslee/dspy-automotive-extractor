# DSPy-Powered Prompt Optimization for Automotive Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-DSPy-orange" alt="Framework">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Package%20Manager-uv-purple" alt="Package Manager">
  <img src="https://img.shields.io/badge/LLM-Ollama-green" alt="LLM Runtime">
  <img src="https://img.shields.io/badge/Observability-Langfuse-blue" alt="Observability">
  <img src="https://img.shields.io/badge/Cloud-Streamlit-red" alt="Cloud Ready">
</p>

**Keywords**: `DSPy`, `Prompt Optimization`, `LLMs`, `Structured Data Extraction`, `Ollama`, `Langfuse`, `Streamlit`, `Enterprise AI`, `Automotive`, `NHTSA`, `Meta-Optimization`, `Reasoning Fields`

---

## 🎯 Project Overview

This project demonstrates a systematic and programmatic framework for optimizing Large Language Models (LLMs) to perform high-accuracy structured data extraction from unstructured automotive service complaints. Leveraging Stanford's **DSPy** library, this system transitions from manual "prompt engineering" to a more robust "prompt compilation" paradigm with **groundbreaking experimental insights**.

### 🔬 Experimental Breakthroughs

Through comprehensive two-phase experimentation, this project has **validated critical insights** about DSPy optimization:

- **Phase 1**: Reasoning fields provide **universal improvement** (+2.0% to +8.66%) across all strategies
- **Phase 2**: Meta-optimization **fails to exceed reasoning field baselines** (-2.0% regression)
- **Key Discovery**: **Reasoning fields + DSPy alignment = optimization sweet spot**

The core objective is to take raw, unstructured text from the official **NHTSA vehicle complaints database** and reliably extract structured entities like the vehicle's **Make, Model, and Year**. The system uses DSPy optimizers to automatically find the most effective prompt instructions and few-shot examples based on quantitative evaluation metrics.

## 🔬 The Core Innovation: From Prompt Engineering to Prompt Science

This project adopts the DSPy philosophy: **treat LLM pipelines as programs that can be compiled and optimized** with rigorous experimental validation.

### Two-Phase Experimental Design

#### Phase 1: Reasoning Field Impact Analysis ✅ **CONFIRMED**
- **Hypothesis**: Explicit reasoning tokens improve extraction accuracy
- **Method**: Compare identical strategies with/without reasoning output field
- **Results**: **Universal improvement** across all 5 strategies (100% success rate)
- **Champion**: Contrastive CoT + Reasoning achieved **51.33% F1-score**

#### Phase 2: Meta-Optimization Effectiveness ❌ **REFUTED**
- **Hypothesis**: Meta-optimization techniques can enhance baseline strategies
- **Method**: Apply 6 meta-optimization techniques to reasoning-enhanced baselines
- **Results**: **Failed to exceed 51.33% ceiling** (best meta-optimized: 49.33%)
- **Critical Discovery**: Instruction conflicts create performance degradation

## 🏆 Key Features & Achievements

| Feature | Description | Experimental Validation |
| :--- | :--- | :--- |
| 🤖 **Automated Optimization** | DSPy teleprompters find optimal prompts through systematic exploration | **51.33%** peak performance achieved |
| 📊 **Quantitative Evaluation** | F1-score metrics with rigorous train/validation splits | **26 strategies** tested across 2 phases |
| 🚀 **Local & Secure** | On-premises execution with Ollama, zero external API calls | **100% data privacy** maintained |
| 🔍 **Fine-Grained Observability** | Langfuse integration for complete optimization traceability | **Full debugging** capabilities |
| 🧪 **Modular & Testable** | Sequential pipeline with comprehensive error handling | **Google-style docstrings** throughout |
| ✨ **Interactive Dashboards** | Multiple Streamlit apps for local and cloud deployment | **Cloud-compatible** with demo data |
| 📋 **Research Validation** | Two-phase experimental design with statistical rigor | **Published insights** in ANALYSIS.md |
| 🌐 **Cloud Deployment** | Streamlit Community Cloud ready with embedded demo data | **Zero local dependencies** |

## 🏗️ Project Architecture & Experimental Pipeline

The project is organized into a systematic pipeline with numbered scripts representing the complete optimization workflow, plus advanced meta-optimization capabilities and cloud deployment options.

### 4.1 Core Pipeline Modules

| Module | Purpose | Experimental Role | Key Contributions |
| :--- | :--- | :--- | :--- |
| `settings.py` | Central configuration hub | Environment setup, logging, Ollama/Langfuse integration | **Reproducible experiments** |
| `_01_load_data.py` | Data pipeline foundation | CSV loading, cleaning, redacted content filtering | **500 clean examples** |
| `_02_define_schema.py` | Schema and strategy definitions | Pydantic models, 5 prompting strategies | **Strategy Pattern** implementation |
| `_03_define_program.py` | Core DSPy implementation | ExtractionModule, F1-score metric, error handling | **Robust evaluation** framework |
| `_04_run_optimization.py` | Basic optimization orchestration | BootstrapFewShot compilation, result persistence | **Phase 1** experiments |
| `_05_meta_optimizers.py` | Meta-optimization techniques | 6 enhancement patterns, research-backed approaches | **Phase 2** foundation |
| `_06_run_meta_optimization.py` | Advanced optimization pipeline | Meta-optimization experiments, comprehensive analysis | **Phase 2** execution |
| `verify_gpu.py` | System diagnostics | PyTorch CUDA, nvidia-smi, Ollama status verification | **Environment validation** |

### 4.2 Dashboard Applications

| Application | Purpose | Deployment | Key Features |
| :--- | :--- | :--- | :--- |
| `app.py` | Full-featured local dashboard | Local with Ollama | Live demo, model inspection, complete analysis |
| `app_cloud.py` | Cloud-compatible dashboard | Streamlit Community Cloud | Demo data, full analytics, no local dependencies |

```
dspy-automotive-extractor/
├── .gitignore
├── pyproject.toml
├── requirements.txt            # Cloud deployment dependencies
├── README.md
├── ANALYSIS.md                 # Complete experimental findings
├── .env.template              # Template for environment variables
├── download_2021_data.ps1     # Enhanced data download script
├── data/
│   └── NHTSA_complaints.csv   # Automotive complaints dataset
├── results/
│   ├── optimized_program_*.json        # Optimized DSPy programs
│   └── results_summary.json            # Central experimental results
└── src/
    ├── __init__.py
    ├── settings.py                     # Central configuration
    ├── _01_load_data.py               # Data loading with redacted filtering
    ├── _02_define_schema.py           # DSPy Signatures and 5 strategies
    ├── _03_define_program.py          # DSPy modules and F1-score metric
    ├── _04_run_optimization.py        # Basic optimization (Phase 1)
    ├── _05_meta_optimizers.py         # Meta-optimization techniques
    ├── _06_run_meta_optimization.py   # Advanced optimization (Phase 2)
    ├── verify_gpu.py                  # System diagnostics
    ├── app.py                         # Local dashboard with live demo
    └── app_cloud.py                   # Cloud-ready dashboard
```

## 🚀 Setup and Installation

### Prerequisites
1. **Python 3.11+**: Ensure Python is installed and available in PATH
2. **Git**: Required for repository cloning
3. **Ollama**: Install for local LLM inference (local deployment only)
4. **NVIDIA Drivers**: Optional but recommended for GPU acceleration

### Step 1: Clone the Repository
```powershell
git clone https://github.com/Adredes-weslee/dspy-automotive-extractor.git
cd dspy-automotive-extractor
```

### Step 2: Download the Dataset

#### Option A: Single Month Sample (Quick Start)
```powershell
mkdir data
curl.exe -L -k "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADAS.csv" -o "data/NHTSA_complaints.csv"
```

#### Option B: Complete 2021 Dataset (Comprehensive)
```powershell
# Run the enhanced PowerShell script for automatic discovery and download
.\download_2021_data.ps1
```

This script automatically:
- Discovers available NHTSA data across multiple years (2010-2025)
- Downloads all available monthly files
- Combines them into a comprehensive dataset
- Handles error checking and file validation

### Step 3: Download Ollama Models (Local Deployment Only)

For local deployments with live demo capabilities:

**High-performance GPUs (≥8GB VRAM):**
```powershell
ollama pull gemma3:12b
```

**CPU-only or lower-end GPUs:**
```powershell
ollama pull qwen3:4b
```

### Step 4: Environment Setup

#### Option A: Local Development (Full Features)
```powershell
# Install uv package manager
pip install uv

# Create virtual environment
python -m uv venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support
.\.venv\Scripts\python.exe -m pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126

# Install project dependencies
python -m uv pip install -e .
```

#### Option B: Cloud Deployment (Streamlit Community Cloud)
For Streamlit Community Cloud deployment, the requirements.txt is automatically used:
```txt
dspy-ai==2.6.27
langfuse==2.36.0
langchain==0.3.25
psutil==7.0.0
streamlit==1.35.0
pandas==2.2.2
python-dotenv==1.0.1
pydantic==2.7.4
lark==1.1.9
scikit-learn==1.5.1
plotly==5.17.0
```

### Step 5: Configure Environment (Local Only)
```powershell
# Copy template and configure
copy .env.template .env
# Edit .env with your Langfuse credentials and model preferences
```

### Step 6: Verify Setup (Local Only)
```powershell
.\.venv\Scripts\python.exe src/verify_gpu.py
```

## 🧪 Running the Experimental Pipeline

### Phase 1: Reasoning Field Experiments (Baseline)

Test the impact of reasoning fields on extraction accuracy:

```powershell
# Run individual strategies
.\.venv\Scripts\python.exe src/_04_run_optimization.py naive_without_reasoning
.\.venv\Scripts\python.exe src/_04_run_optimization.py naive_with_reasoning

# Compare all baseline strategies
.\.venv\Scripts\python.exe src/_06_run_meta_optimization.py baseline
```

**Expected Results**: Universal improvement with reasoning fields (+2.0% to +8.66%)

### Phase 2: Meta-Optimization Experiments (Advanced)

Test advanced prompt engineering techniques:

```powershell
# Run priority meta-optimized strategies
.\.venv\Scripts\python.exe src/_06_run_meta_optimization.py meta

# Run single meta-optimized strategy
.\.venv\Scripts\python.exe src/_06_run_meta_optimization.py single --strategy contrastive_cot_domain_expertise

# Run comprehensive ablation study
.\.venv\Scripts\python.exe src/_06_run_meta_optimization.py ablation
```

**Expected Results**: Failure to exceed reasoning field baseline (49.33% vs 51.33%)

### Analyze Results

```powershell
# Generate comprehensive analysis
.\.venv\Scripts\python.exe src/_06_run_meta_optimization.py analyze

# List all available strategies
.\.venv\Scripts\python.exe src/_06_run_meta_optimization.py --list-strategies
```

## 📊 Interactive Dashboards

### Local Dashboard (Full Features)
```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app.py
```

**Features**:
- 🧪 **Live Demo**: Test best-performing models on custom inputs
- 🔍 **Model Inspection**: Examine optimized prompts and signatures
- 📈 **Performance Analysis**: Compare strategies with interactive charts
- 🔗 **Langfuse Integration**: Direct links to optimization traces

### Cloud Dashboard (Demo Data)
```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app_cloud.py
```

**Features**:
- 📊 **Complete Analytics**: Full experimental results visualization
- 🌐 **Cloud Compatible**: Works without local dependencies
- 🎯 **Interactive Filtering**: Strategy type and performance controls
- 📋 **Comprehensive Insights**: All research findings included

## 🔬 Experimental Strategies & Results

### Baseline Strategies (Phase 1)

| Strategy | Without Reasoning | With Reasoning | Improvement | Status |
|----------|------------------|----------------|-------------|---------|
| **Contrastive CoT** | 42.67% | **51.33%** | **+8.66%** | 🏆 Champion |
| **Naive** | 42.67% | 46.67% | +4.0% | ✅ Improved |
| **Chain of Thought** | 42.67% | 46.0% | +3.33% | ✅ Improved |
| **Plan & Solve** | 43.33% | 46.0% | +3.33% | ✅ Improved |
| **Self-Refine** | 43.33% | 45.33% | +2.0% | ✅ Improved |

### Meta-Optimization Techniques (Phase 2)

| Technique | Description | Best Result | vs Baseline |
|-----------|-------------|-------------|-------------|
| **Domain Expertise** | Automotive knowledge injection | 49.33% | **-2.0%** ❌ |
| **Specificity** | Detailed extraction guidelines | 47.33% | -4.0% ❌ |
| **Error Prevention** | Common failure mode avoidance | 46.67% | -4.66% ❌ |
| **Context Anchoring** | Role-playing and context framing | 45.33% | -6.0% ❌ |
| **Constitutional** | Multi-principle reasoning framework | 46.0% | -5.33% ❌ |
| **Format Enforcement** | Strict output format requirements | **27.33%** | **-24.0%** 💥 |

## 🎯 Key Experimental Insights

### ✅ Validated Discoveries

1. **Reasoning fields are the optimization sweet spot** - Universal +4.26% average improvement
2. **DSPy framework alignment beats prompt complexity** - Native optimization outperforms external techniques
3. **Performance ceilings exist** - More complexity ≠ better results (51.33% ceiling established)
4. **Instruction conflicts cause severe degradation** - Format enforcement dropped performance 24%
5. **Simple + reasoning > complex + meta-optimization** - Architecture alignment principle

### 🚀 Strategic Recommendations

#### For Maximum Performance
- **Use Contrastive CoT + Reasoning** (proven 51.33% performance)
- **Always include reasoning fields** for structured extraction tasks
- **Avoid meta-optimization** for DSPy-optimized baselines
- **Test framework compatibility** before prompt enhancements

#### For Research & Development
- **Reason fields first, meta-optimization second** - Establish baseline before enhancement
- **Monitor for instruction conflicts** in complex prompts
- **Validate framework alignment** before deployment
- **Document performance ceilings** for task-specific optimization

## 🌐 Cloud Deployment

### Streamlit Community Cloud

This project is **cloud deployment ready** with automatic fallback to demonstration data:

1. **Fork the repository** to your GitHub account
2. **Connect to Streamlit Community Cloud**
3. **Set main file path**: app_cloud.py
4. **Deploy automatically** - no additional configuration needed

The cloud version provides:
- ✅ Complete experimental results analysis
- ✅ Interactive visualizations and filtering
- ✅ All research insights and recommendations
- ✅ Embedded demo data for full functionality

### Local vs Cloud Features

| Feature | Local (`app.py`) | Cloud (`app_cloud.py`) |
|---------|------------------|------------------------|
| Live Demo | ✅ Full LLM inference | ❌ Demo data only |
| Model Inspection | ✅ Complete | ✅ Complete |
| Results Analysis | ✅ Real data | ✅ Demo data |
| Performance Charts | ✅ Interactive | ✅ Interactive |
| Research Insights | ✅ Complete | ✅ Complete |
| Dependencies | Ollama + PyTorch | Streamlit only |

## 🔧 Troubleshooting & Common Issues

### GPU and CUDA Issues
```powershell
# Check CUDA availability
.\.venv\Scripts\python.exe -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Verify GPU status
nvidia-smi

# Test Ollama GPU usage
ollama run gemma3:12b "Test message"
```

### Ollama Connection Issues
- **Ensure Ollama is running**: `ollama serve`
- **Verify model availability**: `ollama list`
- **Check model loading**: `ollama run gemma3:12b`

### Memory and Performance
- **Use smaller models**: `qwen3:4b` instead of `gemma3:12b`
- **Reduce sample size** in `_01_load_data.py`
- **Monitor resources** during optimization with `verify_gpu.py`

### Cloud Deployment Issues
- **Use requirements.txt** for dependencies
- **Set main file**: app_cloud.py
- **No environment variables** needed for cloud version

## 📈 Performance Expectations

Based on comprehensive experimental validation:

| Strategy Type | F1-Score Range | Runtime (GPU) | Runtime (CPU) | Experimental Status |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (- Reasoning)** | 42.67% - 43.33% | 5-10 min | 20-30 min | ✅ **Validated** |
| **Baseline (+ Reasoning)** | 45.33% - 51.33% | 10-15 min | 30-45 min | ✅ **Validated** |
| **Meta-Optimized** | 27.33% - 49.33% | 15-25 min | 45-60 min | ❌ **Regression** |
| **Peak Performance** | **51.33%** | 12 min | 35 min | 🏆 **Champion** |

*Performance validated through rigorous two-phase experimental design with 26 strategies tested.*

## 🔮 Future Research Directions

Based on experimental findings, promising research areas include:

### Immediate Opportunities
- **Reasoning quality optimization**: Improve reasoning example curation
- **Framework-native enhancements**: Work within DSPy architectural constraints
- **Task-specific reasoning patterns**: Domain-specific reasoning templates
- **Performance ceiling investigation**: Understanding optimization limits

### Advanced Extensions
- **Multi-domain adaptation**: Medical, legal, financial document extraction
- **Multi-modal integration**: Images, PDFs, audio transcript processing
- **Production optimization**: API endpoints and model serving
- **Cross-framework validation**: Comparative analysis with other optimization frameworks

## 📚 Research Publications & Documentation

- **ANALYSIS.md**: Complete experimental findings and theoretical analysis
- **[DSPy Documentation](https://dspy-docs.vercel.app/)**: Framework documentation
- **[Langfuse Documentation](https://langfuse.com/docs)**: Observability platform
- **[NHTSA Data Portal](https://www.nhtsa.gov/nhtsa-datasets-and-apis)**: Dataset source

## 📄 License & Attribution

This project is licensed under the **MIT License** - see the LICENSE file for details.

### Citation
If you use this research or code in your work, please cite:
```bibtex
@software{dspy_automotive_extractor_2025,
  title={DSPy-Powered Prompt Optimization for Automotive Intelligence},
  author={Lee, Wes},
  year={2025},
  url={https://github.com/Adredes-weslee/dspy-automotive-extractor},
  note={Experimental validation of reasoning fields vs meta-optimization}
}
```

---

## 🎉 Quick Start Summary

1. **Clone & Setup**: `git clone` → `pip install uv` → `uv pip install -e .`
2. **Download Data**: Run download_2021_data.ps1 for comprehensive dataset
3. **Install Ollama**: `ollama pull gemma3:12b` (local only)
4. **Run Experiments**: `python _04_run_optimization.py contrastive_cot_with_reasoning`
5. **View Results**: `streamlit run src/app_cloud.py` (no local dependencies)

**Expected Outcome**: 51.33% F1-score with Contrastive CoT + Reasoning strategy 🏆

**Happy optimizing! 🚀**

---

*This project represents a significant contribution to the DSPy optimization methodology, providing the first comprehensive experimental validation of reasoning fields vs meta-optimization approaches with rigorous statistical analysis and reproducible results.*