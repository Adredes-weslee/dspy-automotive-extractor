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
import plotly.express as px
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


def display_enhanced_results_tab() -> None:
    """Enhanced results tab with dynamic data and visualizations."""
    st.header("📈 Experiment Results")

    summary_data = load_summary_data()

    if not summary_data:
        st.warning("⚠️ `results_summary.json` not found. Please run optimization first.")
        return

    # Sidebar filters
    with st.sidebar:
        st.header("🎛️ Results Filters")
        show_baseline = st.checkbox("Show Baseline Strategies", value=True)
        show_meta = st.checkbox("Show Meta-Optimized Strategies", value=True)
        show_mipro = st.checkbox("Show MIPRO Strategies", value=True)
        min_score = st.slider("Minimum F1 Score (%)", 0.0, 100.0, 0.0, 5.0)

    # Process data dynamically with ENHANCED LOGIC FOR REASONING DETECTION
    df_data = []
    for strategy, data in summary_data.items():
        score = data.get("final_score", 0)
        if score < min_score:
            continue

        # ENHANCED strategy type detection including reasoning variants
        strategy_type_from_data = data.get("strategy_type", "")

        # Check for reasoning variants first
        has_reasoning = "with_reasoning" in strategy
        no_reasoning = "without_reasoning" in strategy

        # Method 1: Check explicit strategy_type field (most reliable)
        if strategy_type_from_data == "meta_optimized":
            strategy_type = "Meta-Optimized"
            is_baseline = False
            is_meta = True
            is_mipro = False
        # Method 2: Check for MIPRO in strategy name
        elif "mipro" in strategy.lower():
            strategy_type = "MIPRO"
            is_baseline = False
            is_meta = False
            is_mipro = True
        # Method 3: Check if it ends with bootstrap (backup for meta-optimized)
        elif strategy.endswith("_bootstrap"):
            strategy_type = "Meta-Optimized"
            is_baseline = False
            is_meta = True
            is_mipro = False
        # Method 4: Baseline strategies with reasoning distinction
        else:
            if has_reasoning:
                strategy_type = "Baseline (+ Reasoning)"
            elif no_reasoning:
                strategy_type = "Baseline (- Reasoning)"
            else:
                strategy_type = "Baseline"
            is_baseline = True
            is_meta = False
            is_mipro = False

        # Apply filters
        if is_baseline and not show_baseline:
            continue
        if is_meta and not show_meta:
            continue
        if is_mipro and not show_mipro:
            continue

        df_data.append(
            {
                "Strategy": strategy.replace("_", " ").title(),
                "F1 Score": score,
                "Type": strategy_type,
                "Timestamp": data.get("timestamp", "N/A"),
                "Raw Strategy": strategy,
            }
        )

    if df_data:
        df = pd.DataFrame(df_data)

        fig = px.bar(
            df.sort_values("F1 Score", ascending=True),
            x="F1 Score",
            y="Strategy",
            color="Type",
            title="F1 Scores by Strategy",
            height=max(500, len(df) * 25),  # Dynamic height
            color_discrete_map={
                "Baseline (- Reasoning)": "#87CEEB",
                "Baseline (+ Reasoning)": "#1f77b4",
                "Baseline": "#1f77b4",
                "Meta-Optimized": "#ff7f0e",
                "MIPRO": "#2ca02c",
            },
        )

        # Force show all y-axis labels
        fig.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(df))),
                ticktext=df.sort_values("F1 Score", ascending=True)[
                    "Strategy"
                ].tolist(),
            ),
            margin=dict(l=150),  # Add left margin for longer strategy names
        )

        st.plotly_chart(fig, use_container_width=True)

        # Enhanced dataframe display
        st.dataframe(
            df[["Strategy", "F1 Score", "Type", "Timestamp"]],
            use_container_width=True,
            hide_index=True,
        )

        # Reasoning Impact Analysis Section (NEW)
        baseline_with = df[df["Type"] == "Baseline (+ Reasoning)"]
        baseline_without = df[df["Type"] == "Baseline (- Reasoning)"]

        if len(baseline_with) > 0 and len(baseline_without) > 0:
            st.header("🧠 Reasoning Field Impact Analysis")

            # Calculate reasoning impact
            reasoning_comparison = []

            # Match strategies by base name
            for with_row in baseline_with.itertuples():
                strategy_base = with_row._5.replace(
                    "_with_reasoning", ""
                )  # Raw Strategy

                # Find corresponding without reasoning
                without_match = baseline_without[
                    baseline_without["Raw Strategy"].str.contains(
                        strategy_base.replace("_with_reasoning", "")
                    )
                ]

                if len(without_match) > 0:
                    without_score = without_match.iloc[0]["F1 Score"]
                    with_score = with_row._2  # F1 Score
                    improvement = with_score - without_score

                    reasoning_comparison.append(
                        {
                            "Strategy": strategy_base.replace("_", " ").title(),
                            "Without Reasoning": f"{without_score:.2f}%",
                            "With Reasoning": f"{with_score:.2f}%",
                            "Improvement": f"+{improvement:.2f}%",
                            "Improvement_Value": improvement,
                        }
                    )

            if reasoning_comparison:
                df_reasoning = pd.DataFrame(reasoning_comparison)

                # Reasoning impact chart
                fig_reasoning = px.bar(
                    df_reasoning.sort_values("Improvement_Value", ascending=True),
                    x="Improvement_Value",
                    y="Strategy",
                    title="Reasoning Field Impact by Strategy (+/- Improvement)",
                    color_discrete_sequence=["#2E8B57"],  # Sea green
                    text="Improvement",
                )
                fig_reasoning.update_traces(textposition="outside")
                fig_reasoning.update_layout(xaxis_title="F1 Score Improvement (%)")
                st.plotly_chart(fig_reasoning, use_container_width=True)

                # Summary table
                st.dataframe(
                    df_reasoning[
                        [
                            "Strategy",
                            "Without Reasoning",
                            "With Reasoning",
                            "Improvement",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                # Reasoning impact metrics
                avg_improvement = df_reasoning["Improvement_Value"].mean()
                best_improvement = df_reasoning["Improvement_Value"].max()
                best_strategy = df_reasoning.loc[
                    df_reasoning["Improvement_Value"].idxmax(), "Strategy"
                ]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Reasoning Gain", f"+{avg_improvement:.2f}%")
                with col2:
                    st.metric("Best Reasoning Gain", f"+{best_improvement:.2f}%")
                with col3:
                    st.success(f"🏆 **Top Gainer**: {best_strategy}")

        # Meta-Optimization Analysis Section (CORRECTED)
        meta_df = df[
            df["Type"] == "Meta-Optimized"
        ]  # Use exact match instead of contains
        if len(meta_df) > 0:
            st.header("🔬 Meta-Optimization Analysis")

            # Group by meta-optimizer type
            meta_analysis = []
            for strategy_row in meta_df.itertuples():
                strategy_name = strategy_row._5  # Raw Strategy column (index 5)
                for opt_name in [
                    "domain_expertise",
                    "specificity",
                    "error_prevention",
                    "context_anchoring",
                    "format_enforcement",
                    "constitutional",
                ]:
                    if opt_name in strategy_name:
                        meta_analysis.append(
                            {
                                "Meta Optimizer": opt_name.replace("_", " ").title(),
                                "Strategy": strategy_row.Strategy,  # index 1
                                "F1 Score": strategy_row._2,  # F1 Score column (index 2)
                            }
                        )
                        break  # Only match the first optimizer found

            if meta_analysis:
                df_meta = pd.DataFrame(meta_analysis)

                # Create a grouped bar chart for meta-optimizers
                fig_meta = px.bar(
                    df_meta.sort_values("F1 Score", ascending=True),
                    x="F1 Score",
                    y="Strategy",
                    color="Meta Optimizer",
                    title="Meta-Optimization Performance by Technique",
                    height=400,
                )
                st.plotly_chart(fig_meta, use_container_width=True)

                # Summary table
                st.subheader("📊 Meta-Optimizer Performance Summary")
                meta_summary = (
                    df_meta.groupby("Meta Optimizer")
                    .agg({"F1 Score": ["count", "mean", "max", "min"]})
                    .round(2)
                )
                meta_summary.columns = ["Count", "Average", "Best", "Worst"]
                st.dataframe(meta_summary, use_container_width=True)

                # Insights based on meta-optimization results
                best_meta = df_meta.loc[df_meta["F1 Score"].idxmax()]
                worst_meta = df_meta.loc[df_meta["F1 Score"].idxmin()]

                col1, col2 = st.columns(2)
                with col1:
                    st.success(
                        f"🏆 **Best Meta-Optimizer**: {best_meta['Meta Optimizer']} ({best_meta['F1 Score']:.2f}%)"
                    )
                with col2:
                    st.error(
                        f"💥 **Worst Meta-Optimizer**: {worst_meta['Meta Optimizer']} ({worst_meta['F1 Score']:.2f}%)"
                    )

        # MIPRO Analysis Section
        if len(df[df["Type"] == "MIPRO"]) > 0:
            st.header("🎯 MIPRO Analysis")

            mipro_df = df[df["Type"] == "MIPRO"]

            # MIPRO performance chart
            fig_mipro = px.bar(
                mipro_df.sort_values("F1 Score", ascending=True),
                x="F1 Score",
                y="Strategy",
                title="MIPRO Strategy Performance",
                color_discrete_sequence=["#2ca02c"],
                height=300,
            )
            st.plotly_chart(fig_mipro, use_container_width=True)

            # MIPRO summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MIPRO Strategies", len(mipro_df))
            with col2:
                st.metric("Best MIPRO Score", f"{mipro_df['F1 Score'].max():.2f}%")
            with col3:
                st.metric("Average MIPRO Score", f"{mipro_df['F1 Score'].mean():.2f}%")

        # Performance Comparison Section
        st.header("📊 Performance Comparison by Strategy Type")

        # Calculate summary statistics by type
        type_summary = (
            df.groupby("Type")
            .agg({"F1 Score": ["count", "mean", "max", "min", "std"]})
            .round(2)
        )
        type_summary.columns = ["Count", "Average", "Best", "Worst", "Std Dev"]

        st.dataframe(type_summary, use_container_width=True)

        # Box plot for distribution comparison with enhanced colors
        fig_box = px.box(
            df,
            x="Type",
            y="F1 Score",
            title="F1 Score Distribution by Strategy Type",
            color="Type",
            color_discrete_map={
                "Baseline (- Reasoning)": "#87CEEB",  # Light blue
                "Baseline (+ Reasoning)": "#1f77b4",  # Dark blue
                "Baseline": "#1f77b4",  # Fallback blue
                "Meta-Optimized": "#ff7f0e",  # Orange
                "MIPRO": "#2ca02c",  # Green
            },
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Key Insights Section
        st.header("💡 Key Performance Insights")

        # Calculate insights
        best_overall = df.loc[df["F1 Score"].idxmax()]
        worst_overall = df.loc[df["F1 Score"].idxmin()]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.success(
                f"🏆 **Overall Champion**\n{best_overall['Strategy']} ({best_overall['F1 Score']:.2f}%)"
            )

        with col2:
            st.error(
                f"💥 **Lowest Performer**\n{worst_overall['Strategy']} ({worst_overall['F1 Score']:.2f}%)"
            )

        with col3:
            performance_range = df["F1 Score"].max() - df["F1 Score"].min()
            st.warning(f"📊 **Performance Range**\n{performance_range:.2f}% spread")

        # Strategy type comparison if multiple types exist
        if len(df["Type"].unique()) > 1:
            st.subheader("📈 Strategy Type Comparison")
            type_performance = (
                df.groupby("Type")["F1 Score"].mean().sort_values(ascending=False)
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                best_type = type_performance.index[0]
                st.info(
                    f"🥇 **Best Type**: {best_type}\n({type_performance.iloc[0]:.2f}% avg)"
                )

            if len(type_performance) > 1:
                with col2:
                    worst_type = type_performance.index[-1]
                    st.warning(
                        f"🥉 **Worst Type**: {worst_type}\n({type_performance.iloc[-1]:.2f}% avg)"
                    )

                with col3:
                    type_gap = type_performance.iloc[0] - type_performance.iloc[-1]
                    st.metric("Type Performance Gap", f"{type_gap:.2f}%")

    else:
        st.info(
            "No results match the current filter criteria. Adjust the filters to see data."
        )


def display_analysis_tab() -> None:
    """Display analysis and insights tab with comprehensive two-phase experimental findings.

    This function provides comprehensive analysis including:
    - Phase 1: Reasoning field experimental findings and performance patterns
    - Phase 2: Meta-optimization results and failure analysis
    - Theoretical explanations for observed results including prompt engineering conflicts
    - Current experiment status tracking for both phases
    - Dynamic insights based on available results from both experimental phases

    Side Effects:
        - Renders Streamlit markdown content
        - Displays experiment progress dataframes for both phases
        - Shows dynamic analysis based on current results
        - Highlights critical discoveries about DSPy framework compatibility
    """
    st.header("🧠 Two-Phase Experimental Analysis")

    summary_data = load_summary_data()

    # Phase 1: Reasoning Field Results
    st.subheader("📊 Phase 1: Reasoning Field Impact")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### ✅ Phase 1 Results - CONFIRMED HYPOTHESIS
        
        #### 🎯 Reasoning Field Impact
        - **Contrastive CoT**: 42.67% → 51.33% (**+8.66% improvement**) 🏆
        - **Naive strategy**: 42.67% → 46.67% (**+4.0% improvement**)
        - **CoT & Plan & Solve**: Both show **+3.33% improvement**
        - **Self-Refine**: 43.33% → 45.33% (**+2.0% improvement**)
        
        #### 🏆 Champion Established
        **Contrastive CoT + Reasoning: 51.33%** (Performance Ceiling)
        
        #### 🔬 Key Discoveries
        - **Universal improvement**: 100% of strategies benefit from reasoning
        - **Average gain**: +4.26% across all strategies
        - **Complex strategies benefit MORE** from reasoning than simple ones
        """)

    with col2:
        st.markdown("""
        ### 🧬 Why Reasoning Fields Succeeded
        
        #### DSPy Architecture Alignment
        - **Bootstrap learning** enhanced by explicit reasoning traces
        - **Optimization signal** improved through intermediate steps
        - **Framework synergy** between DSPy expectations and reasoning output
        
        #### Contrastive Learning Dominance
        - **Negative examples** teach what NOT to extract
        - **Decision boundaries** clarified through contrasting cases
        - **Error prevention** explicitly modeled in training
        
        #### Performance Pattern
        - **Range**: +2.0% to +8.66% improvement
        - **Consistency**: 100% strategy success rate
        """)

    # Phase 2: Meta-Optimization Results
    st.subheader("📉 Phase 2: Meta-Optimization Impact")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        ### ❌ Phase 2 Results - HYPOTHESIS REFUTED
        
        #### 🚨 Meta-Optimization Failure
        - **Best baseline**: Contrastive CoT + Reasoning (51.33%)
        - **Best meta-optimized**: Contrastive CoT + Domain Expertise (49.33%)
        - **Performance regression**: **-2.0%** ❌
        - **Range**: 27.33% - 49.33% (high variance)
        
        #### 💥 Critical Failures
        - **Format enforcement**: Severe degradation (27.33%)
        - **Constitutional**: Mixed results, complexity overload
        - **Multi-combination**: Diminishing returns
        
        #### 🔍 Key Discovery
        **Meta-optimization cannot exceed reasoning field ceiling**
        """)

    with col4:
        st.markdown("""
        ### 🧠 Why Meta-Optimization Failed
        
        #### Instruction Conflict Syndrome
        ```python
        # Contrastive CoT demands:
        "Provide reasoning showing..."
        
        # Format Enforcement demands:
        "ONLY JSON object... No explanations"
        # DIRECT CONTRADICTION!
        ```
        
        #### The Reasoning Field Ceiling
        - **Tier 1**: Base + Reasoning Fields (51.33%)
        - **Tier 2**: Base + Meta-Optimization (49.33%)
        - **Tier 3**: Base Strategy Alone (42.67%)
        - **Tier 4**: Conflicting Meta-Opts (27.33%)
        
        #### Framework Compatibility Crisis
        **DSPy alignment > Prompt complexity**
        """)

    # Critical Insights Section
    st.subheader("💡 Critical Insights & Implications")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("""
        ### 🎯 Validated Optimization Principles
        
        1. **Reasoning fields are the optimization sweet spot** (+8.66% max)
        2. **DSPy architecture alignment is critical** for performance
        3. **Simple + reasoning > complex + meta-optimization**
        4. **Prompt engineering conflicts severely degrade performance**
        5. **Performance ceilings exist** - more complexity ≠ better results
        
        ### ⚠️ The Meta-Optimization Paradox
        
        **When Meta-Optimization Helps:**
        - Base strategies without reasoning
        - Simple enhancement objectives
        - Framework-compatible optimizations
        
        **When Meta-Optimization Hurts:**
        - Already optimized baselines
        - Conflicting objectives
        - Complex multi-optimization
        """)

    with col6:
        st.markdown("""
        ### 🚀 Strategic Recommendations
        
        #### For Maximum Performance
        - **Use Contrastive CoT + Reasoning** (proven 51.33%)
        - **Avoid meta-optimization** for this task type
        - **Prioritize DSPy framework alignment**
        
        #### For Research & Development
        - **Test reasoning fields first** before meta-optimization
        - **Validate framework compatibility** before enhancements
        - **Monitor for instruction conflicts**
        
        ### 🔬 Research Implications
        - **Reasoning fields = primary optimization lever**
        - **Framework-native optimization** beats external prompting
        - **Architectural alignment** as optimization principle
        - **Performance ceiling awareness** critical
        """)

    # Dynamic Results Display
    st.subheader("📊 Complete Two-Phase Experiment Results")

    if summary_data:
        # Phase 1 Results Table
        st.markdown("#### Phase 1: Reasoning Field Results")
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

        # Phase 2 Meta-Optimization Results (if available)
        meta_opt_strategies = [
            k
            for k in summary_data.keys()
            if any(
                meta_name in k
                for meta_name in [
                    "domain_expertise",
                    "specificity",
                    "error_prevention",
                    "context_anchoring",
                    "format_enforcement",
                    "constitutional",
                ]
            )
        ]

        if meta_opt_strategies:
            st.markdown("#### Phase 2: Meta-Optimization Results")
            meta_results_data = []

            for strategy in meta_opt_strategies[:10]:  # Show top 10
                score = summary_data.get(strategy, {}).get("final_score", 0)
                if score:
                    # Determine baseline comparison
                    baseline_score = 51.33 if "contrastive_cot" in strategy else 42.67
                    vs_baseline = score - baseline_score

                    meta_results_data.append(
                        {
                            "Meta-Optimized Strategy": strategy.replace(
                                "_", " "
                            ).title(),
                            "Performance": f"{score:.2f}%",
                            "vs Baseline": f"{vs_baseline:+.2f}%",
                            "Success": "✅" if vs_baseline > 0 else "❌",
                        }
                    )

            if meta_results_data:
                df_meta = pd.DataFrame(meta_results_data)
                st.dataframe(df_meta, use_container_width=True, hide_index=True)

        # Summary Statistics
        completed_improvements = [
            float(row["Improvement"].replace("+", "").replace("%", ""))
            for row in results_data
            if row["Improvement"] != "N/A"
        ]

        if completed_improvements:
            st.markdown("#### Phase 1 Summary Statistics")
            col7, col8, col9, col10 = st.columns(4)
            with col7:
                st.metric(
                    "Average Improvement",
                    f"+{sum(completed_improvements) / len(completed_improvements):.2f}%",
                )
            with col8:
                st.metric("Best Improvement", f"+{max(completed_improvements):.2f}%")
            with col9:
                st.metric("Strategies Improved", f"{len(completed_improvements)}/5")
            with col10:
                st.metric("Performance Ceiling", "51.33%")

        # Final Insights
        st.markdown("""
        ---
        ### 🎯 Final Experimental Conclusions
        
        **Phase 1 (Reasoning Fields): CONFIRMED ✅**
        - Universal improvement across all strategies
        - Established performance ceiling at 51.33%
        - Framework alignment is critical
        
        **Phase 2 (Meta-Optimization): REFUTED ❌**  
        - Failed to exceed reasoning field baseline
        - Created performance conflicts and regressions
        - Complexity penalty outweighed benefits
        
        **Key Discovery: Reasoning fields + DSPy alignment = optimization sweet spot** 🎯
        """)

    else:
        st.info(
            "No experimental data available yet. Run optimization to see two-phase analysis."
        )


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
        display_enhanced_results_tab()

    with tab2:
        display_analysis_tab()

    with tab3:
        display_model_details_tab()

    with tab4:
        display_live_demo_tab()


if __name__ == "__main__":
    main()
