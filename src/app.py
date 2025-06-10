"""
app.py

This script creates the Streamlit web application to showcase the results
of the DSPy prompt optimization.

The application allows users to:
1.  Select one of the pre-optimized prompt programs from a dropdown menu.
2.  Input a sample vehicle complaint narrative.
3.  See a side-by-side comparison of the structured output from:
    a) A naive, un-optimized prompt.
    b) The selected, highly-optimized DSPy program.
4.  View performance metrics and information about the different techniques.
"""
import os
import sys
import dspy
import json
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src._02_define_schema import ExtractionSignature, VehicleInfo
from src._03_define_program import ExtractionModule
from src.settings import RESULTS_DIR, logger

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DSPy Automotive Extractor",
    page_icon="🤖",
    layout="wide",
)

# --- Caching ---
# Cache the loaded DSPy programs to avoid reloading on every interaction
@st.cache_resource
def load_optimized_program(path: Path) -> Optional[ExtractionModule]:
    """Loads a compiled DSPy program from a JSON file."""
    try:
        program = ExtractionModule()
        program.load(str(path))
        logger.info(f"Loaded optimized program from {path}")
        return program
    except Exception as e:
        st.error(f"Failed to load program from {path}: {e}")
        return None

# --- Main Application UI ---
st.title("🚀 DSPy-Powered Prompt Optimization Showcase")
st.markdown("Comparing a naive prompt against DSPy-compiled programs for structured data extraction.")

# --- Sidebar for Program Selection and Info ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # Find available optimized programs
    available_programs = {
        f.stem.replace("optimized_", ""): f for f in RESULTS_DIR.glob("*.json")
    }

    if not available_programs:
        st.warning("No optimized programs found in the 'results' folder. Please run the optimization script first.")
        selected_strategy_name = None
    else:
        selected_strategy_name = st.selectbox(
            "Choose an Optimized Prompting Strategy:",
            options=list(available_programs.keys()),
            index=0
        )

    st.header("About This Project")
    st.info(
        """
        This demo uses the DSPy framework to programmatically optimize prompts
        for extracting structured data (Make, Model, Year) from unstructured
        NHTSA vehicle complaint narratives.

        The `_04_run_optimization.py` script was run for each strategy, generating
        the optimized programs you can select here.
        """
    )


# --- Main Content Area ---

# Create a naive, un-optimized program for comparison
naive_program = ExtractionModule()
# Set its instruction to the most basic prompt
ExtractionSignature.__doc__ = "Extract the vehicle make, model, and year from the text."


# Input area for user
st.subheader("📝 Enter a Vehicle Complaint Narrative")
default_narrative = "THE CONTACT OWNS A 2022 TESLA MODEL Y. THE CONTACT STATED THAT WHILE DRIVING AT 65 MPH, THE VEHICLE'S AUTONOMOUS BRAKING SYSTEM ACTIVATED INDEPENDENTLY, CAUSING AN ABRUPT STOP IN TRAFFIC. THIS WAS A RECURRING ISSUE. THE DEALER WAS NOTIFIED BUT COULD NOT REPLICATE THE FAILURE."
narrative_input = st.text_area(
    "Paste a narrative here:",
    value=default_narrative,
    height=150
)

if st.button("🔍 Extract Information", disabled=(not selected_strategy_name)):
    if selected_strategy_name and narrative_input:
        # Load the selected optimized program
        program_path = available_programs[selected_strategy_name]
        optimized_program = load_optimized_program(program_path)

        if optimized_program:
            # Configure the LLM for inference
            try:
                model_name = os.getenv("OLLAMA_MODEL", "qwen3:4b") # Use smaller model for faster app inference
                llm = dspy.OllamaLocal(model=model_name)
                dspy.settings.configure(lm=llm)
                st.success(f"Connected to Ollama model: `{model_name}`")
            except Exception as e:
                st.error(f"Failed to connect to Ollama. Is it running? Error: {e}")
                st.stop()


            # --- Display Results Side-by-Side ---
            col1, col2 = st.columns(2)

            # Run the Naive Program
            with col1:
                st.subheader("Baseline (Naive Prompt)")
                with st.spinner("Running naive extraction..."):
                    try:
                        prediction = naive_program(narrative=narrative_input)
                        st.json(prediction.vehicle_info.model_dump_json(indent=2))
                    except Exception as e:
                        st.error(f"Naive program failed: {e}")

            # Run the Optimized Program
            with col2:
                st.subheader(f"Optimized ({selected_strategy_name})")
                with st.spinner(f"Running optimized extraction for '{selected_strategy_name}'..."):
                    try:
                        # Set the signature for the optimized program
                        # Note: The optimizer saves the instructions it found best
                        prediction = optimized_program(narrative=narrative_input)
                        st.json(prediction.vehicle_info.model_dump_json(indent=2))
                    except Exception as e:
                        st.error(f"Optimized program failed: {e}")
