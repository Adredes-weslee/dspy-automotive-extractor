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
3.  **Analysis & Insights**: Theoretical analysis of why certain strategies work
4.  **Model Details**: Inspection of optimized programs and prompts
5.  **Live Demo**: Interactive section that allows users to test the best-performing
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
from typing import Any, Dict, Optional, Tuple

import dspy
import pandas as pd
import streamlit as st

# Import necessary modules for the live demo
from _03_define_program import ExtractionModule

# Use settings from the central settings.py module
from settings import RESULTS_DIR, logger, setup_environment


@st.cache_resource
def load_optimized_program(path: str) -> Optional[ExtractionModule]:
    """Load a compiled DSPy program from a JSON file with caching optimization.

    This function loads a previously optimized and saved DSPy program from disk.
    It uses Streamlit's caching mechanism to avoid reloading the same program
    multiple times, which improves dashboard performance when switching between
    different demo inputs.

    The function temporarily configures DSPy with a lightweight model for loading
    purposes, as the actual inference model will be configured separately when
    the program is used for predictions.

    Args:
        path: The file path to the saved DSPy program JSON file
            (e.g., 'results/optimized_naive.json').

    Returns:
        The loaded DSPy program if successful, None if loading failed.

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
        # Configure DSPy once when loading the program
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


@st.cache_resource
def configure_dspy() -> Tuple[dspy.LM, Any]:
    """Configure DSPy with Langfuse logging for Streamlit app startup.

    This function sets up the DSPy framework with Ollama language model integration
    and Langfuse observability logging for the Streamlit dashboard application.
    It uses Streamlit's caching mechanism to ensure configuration occurs only once
    per session, preventing redundant setup calls and potential threading issues.

    The function performs the following initialization steps:
    1. Sets up Langfuse logging environment and handlers via setup_environment()
    2. Retrieves the Ollama model name from environment variables with fallback
    3. Creates a DSPy language model instance pointing to the local Ollama service
    4. Configures DSPy global settings with the initialized language model
    5. Logs successful configuration for debugging and monitoring purposes

    Environment Variables:
        OLLAMA_MODEL (str, optional): Name of the Ollama model to use for inference.
                                     Defaults to "gemma3:12b" if not specified.

    Returns:
        A tuple containing:
            - llm: Configured DSPy language model instance ready for inference
            - langfuse_handler: Langfuse logging handler for trace collection

    Raises:
        ConnectionError: If Ollama service is not running or unreachable
        EnvironmentError: If required environment variables are missing
        ConfigurationError: If DSPy configuration fails due to invalid settings

    Side Effects:
        - Configures global DSPy settings (dspy.settings.configure)
        - Initializes Langfuse logging handlers and callbacks
        - Creates HTTP connections to local Ollama service
        - Writes configuration success message to application logs

    Example:
        >>> llm, langfuse_handler = configure_dspy()
        >>> # DSPy is now ready for inference with Langfuse logging
        >>> prediction = my_program(input_text="2023 Tesla Model Y issue")
        >>> # Traces will appear in Langfuse dashboard at localhost:3000

    Note:
        This function is decorated with @st.cache_resource to ensure that
        DSPy configuration occurs exactly once per Streamlit session, which
        prevents the "dspy.settings can only be changed by the thread that
        initially configured it" error in multi-threaded Streamlit environments.

        The function assumes that:
        - Ollama service is running on localhost:11434 (default port)
        - Langfuse is configured and accessible (setup_environment() succeeds)
        - The specified Ollama model is downloaded and available locally
    """
    # Set up Langfuse logging
    langfuse_handler = setup_environment()

    model_name = os.getenv("OLLAMA_MODEL", "gemma3:12b")
    llm = dspy.LM(model=f"ollama/{model_name}")
    dspy.settings.configure(lm=llm)

    logger.info("DSPy configured with Langfuse logging for Streamlit app")
    return llm, langfuse_handler


def load_summary_data() -> Dict[str, Any]:
    """Load experiment results summary from JSON file.

    Returns:
        Dictionary containing experiment results, empty dict if file not found.
    """
    summary_path = RESULTS_DIR / "results_summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            return json.load(f)
    return {}


def display_results_tab() -> None:
    """Display the results comparison tab with performance metrics and rankings.

    This function creates a comprehensive view of experimental results including:
    - Side-by-side comparison of strategies with/without reasoning
    - Performance improvement calculations
    - Best performing models ranking with medal indicators

    Side Effects:
        - Renders Streamlit UI components
        - Displays dataframes and metrics
        - Shows warning if no results available
    """
    st.header("📈 Experiment Results")

    summary_data = load_summary_data()

    if not summary_data:
        st.warning("⚠️ `results_summary.json` not found. Please run optimization first.")
        return

    # Create comparison visualization
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Performance Comparison")

        # Extract with/without reasoning pairs
        without_reasoning = {
            k: v for k, v in summary_data.items() if "without_reasoning" in k
        }
        with_reasoning = {
            k: v for k, v in summary_data.items() if "with_reasoning" in k
        }

        # Create DataFrame for visualization
        comparison_data = []
        for strategy in [
            "naive",
            "cot",
            "plan_and_solve",
            "self_refine",
            "contrastive_cot",
        ]:
            without_key = f"{strategy}_without_reasoning"
            with_key = f"{strategy}_with_reasoning"

            without_score = without_reasoning.get(without_key, {}).get(
                "final_score", None
            )
            with_score = with_reasoning.get(with_key, {}).get("final_score", None)

            if without_score is not None:
                comparison_data.append(
                    {
                        "Strategy": strategy.replace("_", " ").title(),
                        "Without Reasoning": f"{without_score}%",
                        "With Reasoning": f"{with_score}%"
                        if with_score
                        else "Running...",
                        "Improvement": f"+{with_score - without_score:.2f}%"
                        if with_score
                        else "TBD",
                    }
                )

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("🏆 Best Performing Models")

        # Find best scores
        all_scores = [
            (k, v["final_score"])
            for k, v in summary_data.items()
            if v.get("final_score")
        ]
        all_scores.sort(key=lambda x: x[1], reverse=True)

        for i, (strategy, score) in enumerate(all_scores[:5]):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
            strategy_clean = strategy.replace("_", " ").title()
            st.metric(f"{medal} {strategy_clean}", f"{score}%")


def display_analysis_tab() -> None:
    """Display analysis and insights tab with theoretical foundations and findings.

    This function provides comprehensive analysis including:
    - Key experimental findings and performance patterns
    - Theoretical explanations for observed results
    - Current experiment status tracking
    - Dynamic insights based on available results

    Side Effects:
        - Renders Streamlit markdown content
        - Displays experiment progress dataframes
        - Shows dynamic analysis based on current results
    """
    st.header("🧠 Analysis & Insights")

    summary_data = load_summary_data()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 Key Findings
        
        #### ✅ Reasoning Field Impact
        - **Contrastive CoT**: 42.67% → 51.33% (**+8.66% improvement**) 🏆
        - **Naive strategy**: 42.67% → 46.67% (**+4.0% improvement**)
        - **CoT & Plan & Solve**: Both show **+3.33% improvement**
        - **Self-Refine**: 43.33% → 45.33% (**+2.0% improvement**)
        
        #### 🏆 Strategy Performance Rankings
        1. **Contrastive CoT + Reasoning**: 51.33% (new champion!)
        2. **Naive + Reasoning**: 46.67%
        3. **CoT + Reasoning**: 46.0%
        4. **Plan & Solve + Reasoning**: 46.0%
        5. **Self-Refine + Reasoning**: 45.33%
        
        #### 🚨 Surprising Insights
        - **Complex strategies benefit MORE** from reasoning than simple ones
        - **Contrastive examples** create the strongest reasoning patterns
        - **All strategies** show consistent improvement with reasoning
        """)

    with col2:
        st.markdown("""
        ### 🧬 Theoretical Foundations
        
        #### Why Contrastive CoT Dominates
        - **Positive/negative examples** create robust reasoning patterns
        - **Error avoidance** explicitly taught through bad examples
        - **Decision boundaries** clearer with contrasting cases
        
        #### Reasoning Field Mechanics
        - **Avg. improvement**: +4.0% across all strategies
        - **Range**: +2.0% to +8.66% improvement
        - **Consistency**: 100% of strategies benefit
        
        #### Model Learning Patterns
        - **Complex strategies** have more room for reasoning improvement
        - **Bootstrap learning** enhanced by explicit reasoning traces
        - **Error correction** happens within reasoning chains
        """)

    # Dynamic insights based on current results
    st.subheader("📊 Complete Experiment Results")

    if summary_data:
        strategies = [
            "naive",
            "cot",
            "plan_and_solve",
            "self_refine",
            "contrastive_cot",
        ]
        results_data = []

        for strategy in strategies:
            without_key = f"{strategy}_without_reasoning"
            with_key = f"{strategy}_with_reasoning"

            without_score = summary_data.get(without_key, {}).get("final_score", 0)
            with_score = summary_data.get(with_key, {}).get("final_score", 0)

            improvement = (
                with_score - without_score if (without_score and with_score) else 0
            )

            results_data.append(
                {
                    "Strategy": strategy.replace("_", " ").title(),
                    "Without Reasoning": f"{without_score:.2f}%"
                    if without_score
                    else "N/A",
                    "With Reasoning": f"{with_score:.2f}%" if with_score else "N/A",
                    "Improvement": f"+{improvement:.2f}%" if improvement > 0 else "N/A",
                    "Status": "✅ Complete"
                    if (without_score and with_score)
                    else "❌ Incomplete",
                }
            )

        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        # Summary statistics
        completed_improvements = [
            float(row["Improvement"].replace("+", "").replace("%", ""))
            for row in results_data
            if row["Improvement"] != "N/A"
        ]

        if completed_improvements:
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric(
                    "Average Improvement",
                    f"+{sum(completed_improvements) / len(completed_improvements):.2f}%",
                )
            with col4:
                st.metric("Best Improvement", f"+{max(completed_improvements):.2f}%")
            with col5:
                st.metric("Strategies Improved", f"{len(completed_improvements)}/5")

    else:
        st.info("No experimental data available yet. Run optimization to see progress.")


def display_model_details_tab() -> None:
    """Display model details tab for inspecting optimized programs and prompts.

    This function provides detailed inspection capabilities including:
    - Strategy selection dropdown
    - Performance metrics display
    - Optimized prompt and signature inspection
    - Example demonstrations viewing
    - Full program JSON download functionality

    Side Effects:
        - Renders Streamlit UI components
        - Manages session state for program inspection
        - Provides file download functionality
        - Displays JSON and code content
    """
    st.header("🔍 Optimized Model Details")

    summary_data = load_summary_data()

    if not summary_data:
        st.warning("No results available yet. Run optimization experiments first.")
        return

    # Strategy selector
    strategy_options = list(summary_data.keys())
    selected_strategy = st.selectbox(
        "Select Strategy to Inspect:",
        strategy_options,
        help="Choose a strategy to view its optimized prompts and configuration",
    )

    if selected_strategy:
        strategy_data = summary_data[selected_strategy]
        program_path = strategy_data.get("program_path")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("F1 Score", f"{strategy_data['final_score']}%")
            st.text(f"Timestamp: {strategy_data['timestamp'][:19]}")

            if st.button(
                "🔍 Inspect Optimized Program", key=f"inspect_{selected_strategy}"
            ):
                st.session_state[f"show_program_{selected_strategy}"] = True

        with col2:
            if st.session_state.get(f"show_program_{selected_strategy}", False):
                if program_path and Path(program_path).exists():
                    with open(program_path, "r") as f:
                        program_data = json.load(f)

                    st.subheader("📝 Optimized Prompts")

                    # Extract and display prompts
                    if "predictor" in program_data:
                        predictor_data = program_data["predictor"]

                        # Show signature
                        if "signature" in predictor_data:
                            st.markdown("**Signature:**")
                            st.code(str(predictor_data["signature"]), language="text")

                        # Show instruction
                        if "extended_signature" in predictor_data:
                            ext_sig = predictor_data["extended_signature"]
                            if "instructions" in ext_sig:
                                st.markdown("**Instructions:**")
                                st.code(ext_sig["instructions"], language="text")

                        # Show demos
                        if "demos" in predictor_data and predictor_data["demos"]:
                            st.markdown("**Example Demonstrations:**")
                            for i, demo in enumerate(
                                predictor_data["demos"][:3]
                            ):  # Show first 3
                                with st.expander(f"Demo {i + 1}"):
                                    st.json(demo)

                    # Download button for full program
                    st.download_button(
                        "📥 Download Full Program JSON",
                        data=json.dumps(program_data, indent=2),
                        file_name=f"{selected_strategy}_program.json",
                        mime="application/json",
                        help="Download the complete optimized program configuration",
                    )
                else:
                    st.error("Program file not found!")


def display_live_demo_tab() -> None:
    """Display live demo tab for testing optimized models on new inputs.

    This function provides interactive testing capabilities including:
    - Automatic best model selection
    - Text input for new narratives
    - Real-time inference execution
    - Formatted results display with metrics
    - Raw JSON output for debugging

    Side Effects:
        - Configures DSPy for inference
        - Loads optimized programs
        - Executes model predictions
        - Displays results in Streamlit UI
        - Logs inference operations
    """
    st.header("🔬 Live Demo")

    summary_data = load_summary_data()

    if not summary_data:
        st.info("Run an optimization experiment to enable the live demo.")
        return

    # Configure DSPy once
    llm, _ = configure_dspy()

    # Find the best performing strategy
    best_strategy = max(
        summary_data, key=lambda k: summary_data[k].get("final_score", 0)
    )
    best_program_path = summary_data[best_strategy].get("program_path")

    st.info(
        f"Using the best-performing program: **{best_strategy.replace('_', ' ').title()}** "
        f"({summary_data[best_strategy]['final_score']}%)"
    )

    if best_program_path and Path(best_program_path).exists():
        # Load the best program
        best_program = load_optimized_program(best_program_path)

        default_narrative = (
            "THE CONTACT OWNS A 2022 TESLA MODEL Y. THE CONTACT STATED THAT "
            "WHILE DRIVING AT 65 MPH, THE VEHICLE'S AUTONOMOUS BRAKING SYSTEM "
            "ACTIVATED INDEPENDENTLY, CAUSING AN ABRUPT STOP IN TRAFFIC."
        )

        narrative_input = st.text_area(
            "Enter a complaint narrative to test the extraction:",
            value=default_narrative,
            height=150,
            help="Enter automotive complaint text containing vehicle information",
        )

        if st.button("🚗 Extract Vehicle Info"):
            if best_program and narrative_input:
                try:
                    st.success(
                        f"Using Ollama model: `{os.getenv('OLLAMA_MODEL', 'gemma3:12b')}` for inference."
                    )

                    with st.spinner("Running extraction..."):
                        prediction = best_program(narrative=narrative_input)

                        # Display results in a more readable format
                        st.subheader("🚗 Extracted Vehicle Information")

                        if hasattr(prediction, "vehicle_info"):
                            vehicle_data = prediction.vehicle_info
                            if hasattr(vehicle_data, "model_dump"):
                                data = vehicle_data.model_dump()
                            else:
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

                            # Show reasoning if available
                            if (
                                hasattr(prediction, "reasoning")
                                and prediction.reasoning
                            ):
                                with st.expander("🧠 View Reasoning Chain"):
                                    st.text(prediction.reasoning)

                            # Show full JSON for debugging
                            with st.expander("🔍 View Raw JSON Output"):
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


def main() -> None:
    """Main application entry point with tabbed interface.

    This function initializes the Streamlit application and creates a comprehensive
    dashboard with multiple tabs for different aspects of the DSPy optimization
    analysis and interaction.

    Tabs:
        - Results: Performance comparison and rankings
        - Analysis: Theoretical insights and experimental findings
        - Model Details: Optimized program inspection
        - Live Demo: Interactive testing interface

    Side Effects:
        - Configures Streamlit page settings
        - Renders the complete dashboard interface
        - Manages tab navigation and content display
    """
    # Page configuration
    st.set_page_config(
        page_title="DSPy Automotive Extractor",
        page_icon="🚗",
        layout="wide",
    )

    st.title("🚗 DSPy Automotive Extractor Dashboard")
    st.markdown("*Optimized prompting strategies for vehicle information extraction*")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Results", "🧠 Analysis", "🔍 Model Details", "🔬 Live Demo"]
    )

    with tab1:
        display_results_tab()

    with tab2:
        display_analysis_tab()

    with tab3:
        display_model_details_tab()

    with tab4:
        display_live_demo_tab()


if __name__ == "__main__":
    main()
