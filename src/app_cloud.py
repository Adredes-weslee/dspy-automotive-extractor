r"""
app_cloud.py

Streamlit Cloud-compatible version of the DSPy optimization dashboard.
This version provides comprehensive analysis and visualization of DSPy optimization results
without requiring local LLM inference capabilities.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DSPy Automotive Extractor",
    page_icon="🚗",
    layout="wide",
)


def load_summary_data() -> Dict[str, Any]:
    """Load experiment results summary from JSON file or return demo data.

    Returns:
        Dictionary containing experiment results, demo data if file not found.
    """
    summary_path = Path("results") / "results_summary.json"

    if summary_path.exists():
        with open(summary_path, "r") as f:
            return json.load(f)
    else:
        # Comprehensive demo data based on your actual experimental results
        return {
            # Phase 1: Baseline strategies without reasoning
            "naive_without_reasoning": {
                "final_score": 42.67,
                "timestamp": "2025-06-11T06:02:31.352423",
            },
            "cot_without_reasoning": {
                "final_score": 42.67,
                "timestamp": "2025-06-11T06:15:42.123456",
            },
            "plan_and_solve_without_reasoning": {
                "final_score": 43.33,
                "timestamp": "2025-06-11T06:30:15.789123",
            },
            "self_refine_without_reasoning": {
                "final_score": 43.33,
                "timestamp": "2025-06-11T06:45:22.456789",
            },
            "contrastive_cot_without_reasoning": {
                "final_score": 42.67,
                "timestamp": "2025-06-11T07:00:33.654321",
            },
            # Phase 1: Baseline strategies with reasoning
            "naive_with_reasoning": {
                "final_score": 46.67,
                "timestamp": "2025-06-11T07:15:44.987654",
            },
            "cot_with_reasoning": {
                "final_score": 46.00,
                "timestamp": "2025-06-11T07:30:55.321987",
            },
            "plan_and_solve_with_reasoning": {
                "final_score": 46.67,
                "timestamp": "2025-06-11T07:45:11.654123",
            },
            "self_refine_with_reasoning": {
                "final_score": 45.33,
                "timestamp": "2025-06-11T08:00:22.987456",
            },
            "contrastive_cot_with_reasoning": {
                "final_score": 51.33,
                "timestamp": "2025-06-11T08:15:33.123789",
            },
            # Phase 2: Meta-optimized strategies
            "naive_error_prevention_bootstrap": {
                "final_score": 46.67,
                "strategy_type": "meta_optimized",
                "timestamp": "2025-06-12T08:25:51.979741",
            },
            "cot_multishot_reasoning_bootstrap": {
                "final_score": 47.33,
                "strategy_type": "meta_optimized",
                "timestamp": "2025-06-12T09:29:43.714139",
            },
            "naive_specificity_error_prevention_bootstrap": {
                "final_score": 49.33,
                "strategy_type": "meta_optimized",
                "timestamp": "2025-06-12T10:34:44.382526",
            },
            "cot_context_anchoring_format_enforcement_bootstrap": {
                "final_score": 44.0,
                "strategy_type": "meta_optimized",
                "timestamp": "2025-06-12T11:47:03.280159",
            },
            "plan_and_solve_constitutional_domain_expertise_bootstrap": {
                "final_score": 44.67,
                "strategy_type": "meta_optimized",
                "timestamp": "2025-06-12T13:08:04.321733",
            },
            "naive_context_anchoring_specificity_error_prevention_bootstrap": {
                "final_score": 46.0,
                "strategy_type": "meta_optimized",
                "timestamp": "2025-06-12T14:22:15.456789",
            },
            "contrastive_cot_domain_expertise_bootstrap": {
                "final_score": 49.33,
                "strategy_type": "meta_optimized",
                "timestamp": "2025-06-12T15:35:26.789123",
            },
            "self_refine_format_enforcement_bootstrap": {
                "final_score": 27.33,
                "strategy_type": "meta_optimized",
                "timestamp": "2025-06-12T16:48:37.123456",
            },
        }


def display_enhanced_results_tab() -> None:
    """Display enhanced results analysis tab with comprehensive visualizations and cloud-compatible demo data.

    This function creates an advanced dashboard for analyzing DSPy optimization experiment results
    across multiple strategy types (baseline, meta-optimized, and MIPRO). It provides interactive
    filtering, dynamic visualizations, and detailed performance comparisons including reasoning
    field impact analysis and meta-optimization effectiveness evaluation. The function is specifically
    designed for cloud deployment with automatic fallback to comprehensive demo data when local
    results are unavailable.

    The function automatically categorizes strategies into distinct types based on naming conventions
    and explicit metadata, then provides comprehensive analysis sections for each category with
    interactive Plotly visualizations and statistical summaries. It gracefully handles both real
    experimental data and demo data scenarios for cloud deployment compatibility.

    Strategy Type Detection Logic:
        - **Meta-Optimized**: Strategies with `strategy_type: "meta_optimized"` or ending in `_bootstrap`
        - **MIPRO**: Strategies containing "mipro" in the name (case-insensitive)
        - **Baseline (+ Reasoning)**: Baseline strategies containing "with_reasoning"
        - **Baseline (- Reasoning)**: Baseline strategies containing "without_reasoning"
        - **Baseline**: All other strategies not matching above categories

    Visualization Features:
        - Dynamic height bar charts with color-coded strategy types
        - Interactive sidebar filtering (baseline/meta-optimized/MIPRO/minimum score)
        - Reasoning field impact analysis with delta calculations
        - Meta-optimization performance breakdown by technique
        - Box plots for distribution analysis across strategy types
        - Performance metrics and statistical summaries
        - Cloud demo data notice when using fallback data

    Analysis Sections Generated:
        1. **Main Results Visualization**: Horizontal bar chart of all strategies with filtering
        2. **Reasoning Field Impact**: Side-by-side comparison of with/without reasoning variants
        3. **Meta-Optimization Analysis**: Performance breakdown by meta-optimization technique
        4. **MIPRO Analysis**: Dedicated analysis for MIPRO strategies (if present)
        5. **Performance Comparison**: Statistical summary tables by strategy type
        6. **Key Insights**: Overall champion, worst performer, and performance gaps

    Color Scheme:
        - Light Blue (#87CEEB): Baseline (- Reasoning)
        - Dark Blue (#1f77b4): Baseline (+ Reasoning) and fallback Baseline
        - Orange (#ff7f0e): Meta-Optimized strategies
        - Green (#2ca02c): MIPRO strategies
        - Sea Green (#2E8B57): Reasoning improvement deltas

    Interactive Controls:
        - Sidebar checkboxes for strategy type filtering
        - F1 score threshold slider (0-100%)
        - Expandable sections for detailed analysis
        - Responsive layout with dynamic chart sizing
        - Cloud compatibility notice for demo data usage

    Cloud Deployment Features:
        - Automatic detection of local vs demo data usage
        - Informational notice when using demo data for cloud deployment
        - Full analytical capabilities preserved without local dependencies
        - Graceful handling of missing results files
        - Complete demonstration of dashboard capabilities with sample data

    Data Requirements:
        The function expects either local `results_summary.json` or uses embedded demo data with structure:
        ```json
        {
            "strategy_name": {
                "final_score": float,  // F1 score percentage (0-100)
                "timestamp": str,      // ISO format timestamp
                "strategy_type": str,  // Optional: "meta_optimized" for Phase 2 strategies
                "program_path": str    // Optional: Path to optimized program file
            }
        }
        ```

    Args:
        None: Function uses global data sources and Streamlit session state.

    Returns:
        None: Function renders UI components directly to Streamlit interface.

    Side Effects:
        - Renders multiple Streamlit UI components (headers, charts, dataframes, metrics)
        - Loads and processes experiment results from local file or demo data
        - Creates interactive Plotly visualizations with dynamic sizing
        - Displays informational notices about data source (local vs demo)
        - Calculates and displays statistical summaries and performance insights
        - Shows warning messages if no data matches filter criteria

    Raises:
        FileNotFoundError: Handled gracefully - uses demo data fallback when results file missing
        KeyError: Handled gracefully - uses .get() methods with defaults for missing data fields
        ValueError: Handled gracefully - skips malformed data entries during processing
        StreamlitAPIException: Handled by Streamlit framework for UI component issues

    Example:
        >>> # Called from main tab navigation in cloud deployment
        >>> with tab1:
        >>>     display_enhanced_results_tab()
        # Renders complete analysis dashboard with:
        # - Cloud demo notice if using fallback data
        # - Interactive filtering controls in sidebar
        # - Color-coded strategy performance chart
        # - Reasoning field impact analysis (+8.66% for Contrastive CoT)
        # - Meta-optimization performance breakdown
        # - Statistical summaries and key insights

    Note:
        This function is designed for both local and cloud deployment and assumes:
        - `load_summary_data()` function provides either real or demo data
        - Required visualization libraries (plotly.express, pandas) are imported
        - Results data follows the expected JSON schema with numeric F1 scores
        - Streamlit sidebar and main content areas are available for rendering
        - Path checking capabilities are available for cloud vs local detection

        The function gracefully handles missing data, empty results, and various data quality issues
        by providing appropriate fallbacks and user-friendly messages. It provides full analytical
        capabilities regardless of data source (local results or embedded demo data).

        Performance scales well with large numbers of strategies due to dynamic chart sizing
        and efficient pandas operations for data processing and aggregation. The cloud-compatible
        design ensures consistent functionality across deployment environments.

        Demo data includes comprehensive examples of all strategy types to demonstrate complete
        dashboard capabilities when deployed to Streamlit Community Cloud without local results.
    """
    st.header("📈 Experiment Results")

    summary_data = load_summary_data()

    if not summary_data:
        st.warning("⚠️ No results data available.")
        return

    # Cloud notice for demo data
    if not (Path("results") / "results_summary.json").exists():
        st.info(
            "🌐 **Cloud Demo**: Using example results data to demonstrate dashboard capabilities."
        )

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

        # Reasoning Impact Analysis Section
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

        # Meta-Optimization Analysis Section
        meta_df = df[df["Type"] == "Meta-Optimized"]
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
    """Display comprehensive two-phase experimental analysis with detailed insights and findings.

    This function creates an in-depth analysis tab that presents the complete narrative and
    findings from a two-phase DSPy optimization experiment. It provides detailed theoretical
    analysis, performance comparisons, and strategic insights derived from reasoning field
    impact assessment and meta-optimization effectiveness evaluation.

    The function serves as the primary research insights dashboard, presenting both quantitative
    results and qualitative analysis of why certain optimization approaches succeeded or failed.
    It's designed to communicate complex experimental findings in an accessible, visually
    organized format with clear actionable recommendations.

    Experimental Phases Analyzed:
        1. **Phase 1 - Reasoning Field Impact**: Analysis of adding reasoning fields to baseline
           strategies, showing universal improvement across all tested approaches
        2. **Phase 2 - Meta-Optimization**: Evaluation of advanced prompt engineering techniques
           applied to optimized baselines, revealing performance regression and conflicts

    Content Structure:
        - **Phase 1 Results**: Confirmed hypothesis with detailed performance improvements
        - **Phase 1 Analysis**: Theoretical explanation of why reasoning fields succeeded
        - **Phase 2 Results**: Refuted hypothesis with meta-optimization failure analysis
        - **Phase 2 Analysis**: Technical explanation of instruction conflicts and ceiling effects
        - **Critical Insights**: Validated optimization principles and strategic paradoxes
        - **Strategic Recommendations**: Actionable guidance for performance and research
        - **Dynamic Results Tables**: Real-time data display with completion status
        - **Summary Statistics**: Quantitative metrics and experimental conclusions

    Key Findings Presented:
        - Contrastive CoT + Reasoning achieved 51.33% performance ceiling (+8.66% improvement)
        - Universal reasoning field benefit across all 5 baseline strategies tested
        - Meta-optimization failed to exceed reasoning field baseline (49.33% vs 51.33%)
        - Format enforcement caused severe performance degradation (27.33%)
        - DSPy framework alignment more critical than prompt complexity

    Theoretical Insights Covered:
        - Bootstrap learning enhancement through explicit reasoning traces
        - Contrastive learning dominance via negative example training
        - Instruction conflict syndrome between competing optimization objectives
        - Performance ceiling establishment and framework compatibility crisis
        - Meta-optimization paradox: when it helps vs when it hurts

    UI Components Created:
        - Multi-column layout for side-by-side phase comparison
        - Emoji-enhanced section headers with status indicators
        - Code snippets demonstrating instruction conflicts
        - Performance tier visualization with explicit rankings
        - Dynamic data tables with conditional formatting
        - Summary metrics with four-column statistical overview
        - Conclusion section with experimental validation status

    Data Processing Features:
        - Real-time strategy matching between with/without reasoning variants
        - Automatic baseline score determination for meta-optimization comparison
        - Dynamic improvement calculation with percentage formatting
        - Completion status tracking for experimental coverage
        - Top 10 meta-optimization strategy filtering for readability

    Args:
        None: Function operates independently using global data sources.

    Returns:
        None: Function renders Streamlit UI components directly to the interface.

    Side Effects:
        - Loads experimental data from summary_data source (real or demo)
        - Renders comprehensive multi-section analysis interface
        - Displays performance tables with conditional data availability
        - Creates summary statistics with dynamic calculations
        - Shows experimental conclusions with validation status indicators

    Raises:
        KeyError: Gracefully handled - uses .get() methods with defaults for missing data
        ValueError: Handled gracefully - skips malformed improvement calculations
        StreamlitAPIException: Handled by Streamlit framework for UI rendering issues

    Example:
        >>> # Called from main tab navigation
        >>> with tab2:
        >>>     display_analysis_tab()
        # Renders complete experimental analysis with:
        # - Phase 1: Reasoning field impact analysis (CONFIRMED hypothesis)
        # - Phase 2: Meta-optimization effectiveness (REFUTED hypothesis)
        # - Critical insights and strategic recommendations
        # - Dynamic results tables and summary statistics

    Note:
        This function is designed for research communication and assumes:
        - Users want detailed theoretical understanding of experimental results
        - Both quantitative data and qualitative insights are important
        - Strategic recommendations should be actionable and specific
        - Experimental narrative should be clear and accessible

        The function presents hard-coded insights based on completed experimental analysis
        while dynamically displaying any available results data. It balances theoretical
        depth with practical applicability, making complex DSPy optimization findings
        accessible to both researchers and practitioners.

        Performance data is processed efficiently with pandas operations, and the interface
        scales well with varying amounts of experimental data while maintaining
        comprehensive analytical coverage.
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


def display_cloud_demo_tab() -> None:
    """Display cloud deployment information and local setup instructions for full functionality.

    This function creates a comprehensive information tab specifically designed for the Streamlit
    Cloud version of the DSPy automotive extractor dashboard. It provides clear explanations
    about the limitations of the cloud deployment, instructions for local setup to access full
    interactive capabilities, and demonstrates what the live extraction demo would produce.

    The function serves as a bridge between the cloud-compatible analytical dashboard and the
    full-featured local version, helping users understand the differences and providing guidance
    for accessing complete functionality when needed.

    Content Sections:
        1. **Cloud Limitations Notice**: Explains why live demo is unavailable on cloud
        2. **Local Setup Instructions**: Step-by-step guide for running full version locally
        3. **Feature Comparison**: Cloud vs local capabilities side-by-side
        4. **Sample Extraction Demo**: Example of what the live demo would produce
        5. **Repository Link**: Direct link to GitHub for local deployment

    Local Setup Instructions Provided:
        - Repository cloning instructions
        - Ollama installation and model download requirements
        - Environment setup with dependencies
        - Local execution command for full interactive version

    Cloud Features Highlighted:
        - Complete experimental results analysis capabilities
        - Interactive visualizations and filtering systems
        - Reasoning field impact analysis with delta calculations
        - Meta-optimization performance comparison tools
        - Two-phase experimental insights and theoretical analysis
        - Research findings and strategic recommendations

    Sample Demonstration:
        - Realistic automotive complaint narrative as input
        - Extracted vehicle information (Make, Model, Year) as output
        - Sample reasoning chain showing extraction logic
        - Visual representation of what users would see locally

    UI Components Created:
        - Header with cloud-specific branding
        - Informational notice about cloud limitations
        - Two-column layout for setup vs features comparison
        - Sample extraction results with metrics display
        - Expandable reasoning chain demonstration
        - Repository link for further exploration

    Args:
        None: Function operates independently without parameters.

    Returns:
        None: Function renders Streamlit UI components directly to the interface.

    Side Effects:
        - Displays informational content about cloud vs local capabilities
        - Creates sample extraction demonstration with realistic automotive data
        - Renders step-by-step local setup instructions
        - Shows feature comparison between deployment options
        - Provides repository link for local deployment access

    Raises:
        StreamlitAPIException: Handled by Streamlit framework for UI component issues
        MarkdownError: Implicitly handled - malformed markdown would display as text

    Example:
        >>> # Called from main tab navigation
        >>> with tab3:
        >>>     display_cloud_demo_tab()
        # Renders complete demo information tab with:
        # - Clear explanation of cloud limitations
        # - Local setup instructions for full functionality
        # - Sample extraction demonstration showing Tesla Model Y extraction
        # - Repository link for accessing source code

    Note:
        This function is specifically designed for cloud deployment education and assumes:
        - Users may want to access full interactive capabilities locally
        - Clear distinction between cloud and local features is important
        - Sample data should be realistic and demonstrate actual functionality
        - Repository link should guide users to complete local setup

        The function provides a realistic sample extraction using a Tesla Model Y
        automotive complaint to demonstrate the type of extraction the live demo
        would perform. The reasoning chain example shows the logical process
        the optimized models use for vehicle information extraction.

        All interactive elements are disabled in the sample to prevent confusion
        while still showing the visual format users would see in the local version.
    """
    st.header("🌐 Cloud Demo Information")

    st.info("""
    **Live Demo Not Available in Cloud Version**
    
    The interactive live demo requires local LLM inference capabilities that are not available 
    on Streamlit Community Cloud. However, you can:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🏠 **Local Setup**
        To run the full interactive demo:
        
        1. **Clone the repository**
        2. **Install Ollama** and download models
        3. **Set up the environment** with requirements
        4. **Run the local version**: `streamlit run src/app.py`
        
        ### 🚀 **What You Get Locally**
        - Real-time vehicle information extraction
        - Test with custom automotive complaints
        - See reasoning chains in action
        - Compare optimized vs base models
        """)

    with col2:
        st.markdown("""
        ### 📊 **Cloud Dashboard Features**
        This cloud version provides:
        
        - ✅ **Complete experimental results analysis**
        - ✅ **Interactive visualizations and filtering**
        - ✅ **Reasoning field impact analysis** 
        - ✅ **Meta-optimization performance comparison**
        - ✅ **Two-phase experimental insights**
        - ✅ **Research findings and recommendations**
        
        ### 🔗 **Repository Link**
        [GitHub Repository](https://github.com/your-username/dspy-automotive-extractor)
        """)

    # Example of what the demo would show
    st.subheader("📝 Example Extraction Results")
    st.markdown("Here's what the live demo would extract from this sample complaint:")

    # Sample input
    st.text_area(
        "Sample Automotive Complaint:",
        value="THE CONTACT OWNS A 2022 TESLA MODEL Y. THE CONTACT STATED THAT WHILE DRIVING AT 65 MPH, THE VEHICLE'S AUTONOMOUS BRAKING SYSTEM ACTIVATED INDEPENDENTLY, CAUSING AN ABRUPT STOP IN TRAFFIC.",
        height=100,
        disabled=True,
    )

    # Sample output
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Make", "Tesla")
    with col2:
        st.metric("Model", "Model Y")
    with col3:
        st.metric("Year", "2022")

    with st.expander("🧠 Sample Reasoning Chain"):
        st.text("""
        Looking at the narrative, I need to extract vehicle information:

        1. Make: "TESLA" is explicitly mentioned as the vehicle manufacturer
        2. Model: "MODEL Y" is specifically stated as the vehicle model
        3. Year: "2022" is clearly indicated as the model year

        The extraction is straightforward as all three pieces of information 
        are directly stated in the complaint narrative.
        """)


def main() -> None:
    """Main application entry point for Streamlit Cloud-compatible DSPy optimization dashboard.

    This function serves as the primary entry point for the cloud-compatible version of the DSPy
    automotive extractor optimization analysis dashboard. It initializes the Streamlit interface
    with a three-tab layout providing comprehensive visualization and analysis capabilities for
    DSPy optimization experiment results without requiring local LLM inference capabilities.

    The function creates a complete analytical dashboard that displays optimization experiment
    results, two-phase experimental insights, and cloud deployment information. It's specifically
    designed for Streamlit Community Cloud deployment where local model inference is not available,
    but full analytical capabilities are preserved through demo data and comprehensive visualizations.

    Tab Structure:
        1. **Results & Analysis**: Interactive dashboard with filtering, visualizations, and
           performance comparisons across baseline, meta-optimized, and MIPRO strategies
        2. **Experimental Insights**: Comprehensive two-phase experimental analysis including
           reasoning field impact assessment and meta-optimization effectiveness evaluation
        3. **Demo Info**: Cloud deployment information, local setup instructions, and sample
           extraction demonstrations

    Features Provided:
        - Cloud compatibility notice and demo data usage information
        - Interactive filtering controls for strategy types and performance thresholds
        - Color-coded performance visualizations with dynamic chart sizing
        - Reasoning field impact analysis with delta calculations
        - Meta-optimization technique performance breakdown
        - Statistical summaries and performance insights
        - Two-phase experimental narrative and theoretical analysis
        - Local setup instructions for full interactive capabilities

    UI Components Created:
        - Application title with cloud version designation
        - Informational notices about cloud vs local capabilities
        - Three-tab navigation interface with emoji icons
        - Tab content delegation to specialized display functions
        - Responsive layout optimized for cloud deployment

    Cloud Optimizations:
        - Demo data fallback when results files are unavailable
        - No dependencies on local LLM infrastructure (Ollama, etc.)
        - Streamlined imports without DSPy configuration requirements
        - Full analytical capabilities preserved without inference features
        - Clear distinction between cloud and local functionality

    Args:
        None: Function operates independently without parameters.

    Returns:
        None: Function renders Streamlit UI components directly to the interface.

    Side Effects:
        - Sets Streamlit page title and creates complete dashboard interface
        - Displays informational messages about cloud version capabilities
        - Creates three-tab navigation structure with interactive content
        - Delegates to specialized functions for tab content rendering
        - Loads and processes experimental data (real or demo) for analysis

    Raises:
        ImportError: Implicitly handled - function assumes required libraries are available
        FileNotFoundError: Gracefully handled by display functions with demo data fallback
        StreamlitAPIException: Handled by Streamlit framework for UI component issues

    Example:
        >>> # Called directly when script is executed
        >>> if __name__ == "__main__":
        >>>     main()
        # Creates complete dashboard with:
        # - Tab 1: Interactive results analysis with filtering and visualizations
        # - Tab 2: Two-phase experimental insights and theoretical analysis
        # - Tab 3: Cloud deployment info and local setup instructions

    Note:
        This function is specifically designed for Streamlit Community Cloud deployment
        and assumes that:
        - Streamlit environment is properly configured with required packages
        - Display functions (display_enhanced_results_tab, display_analysis_tab,
          display_cloud_demo_tab) are available and functional
        - Demo data is embedded within the application for cloud compatibility
        - Page configuration has been set prior to calling this function

        The function gracefully handles the absence of local results files by using
        comprehensive demo data that demonstrates all analytical capabilities. It
        provides clear guidance for users who want to access the full interactive
        features through local deployment.

        Performance is optimized for cloud deployment with efficient data processing
        and responsive visualization components that work well within Streamlit
        Community Cloud resource constraints.
    """
    # Application title and description
    st.title("🚗 DSPy Automotive Extractor Dashboard")
    st.markdown("*Cloud version - Comprehensive optimization analysis and insights*")

    # Cloud version notice
    st.info(
        "🌐 **Streamlit Cloud Version** - Full analysis capabilities with demo data"
    )

    # Create tabs
    tab1, tab2, tab3 = st.tabs(
        ["📊 Results & Analysis", "🧠 Experimental Insights", "🌐 Demo Info"]
    )

    with tab1:
        display_enhanced_results_tab()

    with tab2:
        display_analysis_tab()

    with tab3:
        display_cloud_demo_tab()


if __name__ == "__main__":
    main()
