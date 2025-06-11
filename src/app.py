r"""
app.py

This script creates a Streamlit web application that serves as an interactive
"Results Dashboard" for the DSPy prompt optimization experiments.

The application provides a comprehensive interface for analyzing and interacting
with optimization results:

1.  **Experiment Results Summary**: Reads and displays results from the central
    `results_summary.json` file in a clean, sortable table format.
2.  **Performance Comparison**: Shows F1-scores and timestamps for all tested
    prompting strategies (naive, chain_of_thought, plan_and_solve, etc.).
3.  **Langfuse Integration**: Provides direct links to detailed Langfuse traces
    for each experiment, enabling deep analysis of the optimization process.
4.  **Live Demo**: Interactive section that allows users to test the best-performing
    optimized program against new, user-provided automotive complaint narratives.

The dashboard automatically identifies the best-performing strategy and loads
its optimized program for real-time inference testing.

Usage:
    .\.venv\Scripts\streamlit run src\app.py

Example:
    >>> .\.venv\Scripts\streamlit run src\app.py
    # Opens web browser to http://localhost:8501
    # Dashboard shows experiment results and live demo interface

Requirements:
    - Completed optimization experiments (results_summary.json must exist)
    - Ollama service running with configured models
    - Streamlit installed in the virtual environment
"""

import json
import os
from pathlib import Path
from typing import Optional

import dspy
import pandas as pd
import streamlit as st

# Import necessary modules for the live demo
from _03_define_program import ExtractionModule

# Use settings from the central settings.py module
from settings import RESULTS_DIR, logger

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DSPy Optimization Dashboard",
    page_icon="📊",
    layout="wide",
)


# --- Caching ---
@st.cache_resource
def load_optimized_program(path: str) -> Optional[ExtractionModule]:
    """
    Load a compiled DSPy program from a JSON file with caching optimization.

    This function loads a previously optimized and saved DSPy program from disk.
    It uses Streamlit's caching mechanism to avoid reloading the same program
    multiple times, which improves dashboard performance when switching between
    different demo inputs.

    The function temporarily configures DSPy with a lightweight model for loading
    purposes, as the actual inference model will be configured separately when
    the program is used for predictions.

    Args:
        path (str): The file path to the saved DSPy program JSON file
                   (e.g., 'results/optimized_naive.json').

    Returns:
        Optional[ExtractionModule]: The loaded DSPy program if successful,
                                   None if loading failed.

    Side Effects:
        - Temporarily configures DSPy global settings for loading
        - Displays Streamlit error messages if loading fails
        - Logs successful loading operations

    Example:
        >>> program = load_optimized_program("results/optimized_naive.json")
        >>> if program:
        ...     result = program.forward("2023 Tesla Model Y issue")
        ...     print(result.vehicle_info.make)
        Tesla

    Note:
        This function is decorated with @st.cache_resource to ensure that
        the same program file is only loaded once per Streamlit session,
        improving performance for repeated demo usage.
    """
    try:
        # Configure a dummy LLM for loading, it will be replaced later
        model_name = os.getenv("OLLAMA_MODEL", "gemma3:12b")
        llm = dspy.LM(model=f"ollama/{model_name}")
        dspy.settings.configure(lm=llm)
        program = ExtractionModule()
        program.load(path)
        logger.info(f"Loaded optimized program from {path}")
        return program
    except Exception as e:
        st.error(f"Failed to load program from {path}: {e}")
        return None


# --- Main Application UI ---
st.title("📊 DSPy Prompt Optimization Dashboard")
st.markdown("### Comparing Prompting Strategies for Automotive Data Extraction")

# --- Results Summary Section ---
st.header("📈 Experiment Results Summary")

summary_path = RESULTS_DIR / "results_summary.json"
summary_data = {}
if not summary_path.exists():
    st.warning(
        "`results_summary.json` not found. Run the optimization script to generate results."
    )
else:
    with open(summary_path, "r") as f:
        summary_data = json.load(f)

    # Convert to DataFrame for better display
    df_data = []
    for strategy, data in summary_data.items():
        df_data.append(
            {
                "Strategy": strategy.replace("_", " ").title(),
                "Final F1 Score": data.get("final_score", "N/A"),
                "Trace URL": data.get("trace_url", "N/A"),
                "Timestamp": data.get("timestamp", "N/A"),
            }
        )

    df = pd.DataFrame(df_data)
    st.dataframe(
        df,
        column_config={
            "Trace URL": st.column_config.LinkColumn(
                "🔗 View Trace", display_text="View on Langfuse"
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

# --- Live Demo Section ---
st.header("🔬 Live Demo")

if not summary_data:
    st.info("Run an optimization experiment to enable the live demo.")
else:
    # Find the best performing strategy to use for the demo
    best_strategy = max(
        summary_data, key=lambda k: summary_data[k].get("final_score", 0)
    )
    best_program_path = summary_data[best_strategy].get("program_path")

    st.info(
        f"Using the best-performing program: **{best_strategy.replace('_', ' ').title()}**"
    )

    if best_program_path and Path(best_program_path).exists():
        # Load the best program
        best_program = load_optimized_program(best_program_path)

        default_narrative = "THE CONTACT OWNS A 2022 TESLA MODEL Y. THE CONTACT STATED THAT WHILE DRIVING AT 65 MPH, THE VEHICLE'S AUTONOMOUS BRAKING SYSTEM ACTIVATED INDEPENDENTLY, CAUSING AN ABRUPT STOP IN TRAFFIC."
        narrative_input = st.text_area(
            "Enter a complaint narrative to test the extraction:",
            value=default_narrative,
            height=150,
        )

        if st.button("Extract Vehicle Info"):
            if best_program and narrative_input:
                try:
                    # Configure the LLM for inference
                    model_name = os.getenv("OLLAMA_MODEL", "gemma3:12b")
                    llm = dspy.LM(model=f"ollama/{model_name}")
                    dspy.settings.configure(lm=llm)
                    st.success(
                        f"Connected to Ollama model: `{model_name}` for inference."
                    )

                    with st.spinner("Running extraction..."):
                        prediction = best_program(narrative=narrative_input)

                        # Display results in a more readable format
                        st.subheader("🚗 Extracted Vehicle Information")

                        if hasattr(prediction, "vehicle_info"):
                            vehicle_data = prediction.vehicle_info
                            if hasattr(vehicle_data, "model_dump"):
                                # For Pydantic models
                                data = vehicle_data.model_dump()
                            else:
                                # For dict-like objects
                                data = (
                                    dict(vehicle_data)
                                    if hasattr(vehicle_data, "__dict__")
                                    else vehicle_data
                                )

                            # Display in columns for better UX
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Make", data.get("make", "N/A"))
                            with col2:
                                st.metric("Model", data.get("model", "N/A"))
                            with col3:
                                st.metric("Year", data.get("year", "N/A"))

                            # Show full JSON for debugging
                            with st.expander("View Raw JSON Output"):
                                st.json(data)
                        else:
                            st.json(str(prediction))

                except Exception as e:
                    st.error(f"Failed to run inference: {e}")
                    logger.error(f"Inference error: {e}")
            else:
                st.error("Could not load the best-performing program.")
    else:
        st.error(
            f"Could not find the program file for the best strategy: {best_program_path}"
        )
