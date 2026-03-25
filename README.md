# DSPy-Powered Prompt Optimization for Automotive Intelligence

This repo is a DSPy pipeline for extracting `make`, `model`, and `year` from NHTSA complaint narratives, with the schema and prompt strategies defined in `src/_02_define_schema.py` and `src/_03_define_program.py`.

It also saves compiled prompt artifacts and a central results file so strategies can be inspected later in the dashboards.

## Why This Repository Exists

- The task is to turn unstructured automotive complaint text into a consistent structured record that can be compared across prompting strategies with one shared score.
- The optimization scripts exist to answer which prompt strategy works best for this extraction task, not to provide a general-purpose automotive chatbot.

## Architecture at a Glance

- src/settings.py centralizes paths, `.env` loading, DSPy/Ollama setup, and optional Langfuse callbacks.
- src/_01_load_data.py loads `data/NHTSA_complaints.csv`, removes missing/redacted/short rows, converts rows to `dspy.Example`s, and samples up to 500 examples.
- src/_02_define_schema.py defines `VehicleInfo`, `ExtractionSignature`, five base strategies, and the docstring-swapping prompt pattern.
- `src/_03_define_program.py` wraps `dspy.Predict` in `ExtractionModule`; its metric is an exact-field average across make, model, and year rather than a standard F1 score.
- src/_04_run_optimization.py runs `BootstrapFewShot`, while src/_06_run_meta_optimization.py adds `MIPROv2`, batch modes, and result analysis.
- src/_05_meta_optimizers.py defines seven meta-optimizers; src/app.py is the local dashboard and src/app_cloud.py is the cloud version with fallback demo data.
- src/verify_gpu.py is the environment check script for CUDA, NVIDIA, Ollama, system resources, and a DSPy inference test.

## Repository Layout

- `data/`
- `results/`
- `src/`
- `.env.template`
- `.gitignore`
- `ANALYSIS.md`
- `download_2021_data.ps1`
- `pyproject.toml`
- `README.md`
- `requirements.txt`

## Setup and Run

1. Local setup in pyproject.toml assumes Python 3.11+, `uv`, a separate PyTorch CUDA install, and a repo-root working directory; requirements.txt is only the cloud subset and is not enough for the local pipeline.
2. The environment template is `.env.template`; `LANGFUSE_*` is optional, and `OLLAMA_MODEL` defaults to `gemma3:12b` there.
3. Data preparation is either a direct CSV download into `data/NHTSA_complaints.csv` or download_2021_data.ps1, which actually scans NHTSA monthly files across 2010-2025 and builds the combined file.
4. Typical commands are `python src/_04_run_optimization.py <strategy>`, `python src/_06_run_meta_optimization.py <mode>`, `python src/verify_gpu.py`, and `streamlit run src/app.py` or `streamlit run src/app_cloud.py`.
5. The local dashboard needs Ollama and saved results; the cloud dashboard runs without local inference and falls back to embedded demo data if results are missing.

## Core Workflows

- Data prep: load `data/NHTSA_complaints.csv`, filter redacted and short narratives, and create `dspy.Example` objects.
- Baseline run: pick one of the five strategies, compile with `BootstrapFewShot`, evaluate on a 90/10 split, and save `results/optimized_<strategy>.json` plus `results_summary.json`.
- Comparison run: use `src/_06_run_meta_optimization.py` to run baseline, priority meta, comprehensive, ablation, or analysis modes, optionally with `MIPROv2`.
- Inspection run: open `src/app.py`, read `results_summary.json`, inspect a saved program JSON, or run the live demo against the best-scoring saved program.
- Cloud run: open `src/app_cloud.py` for analysis-only visualization when local inference is unavailable.
- Verification run: use `src/verify_gpu.py` before attempting a local Ollama-backed inference demo.

## Known Limitations

- Some bundled docs still contain garbled characters and duplicated sections and need UTF-8 cleanup.
- pyproject.toml still uses placeholder repository URLs, and there is no top-level `LICENSE` file despite the MIT badge and license field.
- results/results_summary.json stores absolute Windows `program_path` values, so copied artifacts are machine-specific unless regenerated.
- The results folder contains filename drift for `optimized_self_refine_without_reasoning`; check `results_summary.json` before hardcoding paths.
- The archived results have `trace_url: null`, so Langfuse is wired in code but not reflected in the saved outputs.
- The metric labeled F1 in current docs is closer to an exact-field average than to a standard F1 score.
- There is no checked-in test suite or CI scaffolding, so verification is script- and dashboard-based.
- src/app_cloud.py is analysis-only with embedded demo data, not a live inference surface.
- The repo checks in generated `data/` and `results/` artifacts, so reproducible steps need to be distinguished from example outputs.
- requirements.txt and pyproject.toml intentionally differ, including Plotly pins, so local and cloud setup differ.
- download_2021_data.ps1 is PowerShell-specific, so that data-download path is Windows-only.
