r"""
_02_define_schema.py

This script defines the data structures and prompting strategies for our LLM pipeline. It includes:
1.  A Pydantic model (`VehicleInfo`) that defines the strict, typed schema for
    the structured data we want to extract.
2.  A base DSPy Signature (`ExtractionSignature`) that defines the high-level
    input/output contract for the LLM.
3.  The Strategy Pattern implementation for Prompting Techniques. This allows us
    to easily switch between different prompt instructions (e.g., CoT, Self-Refine)
    by defining each as a separate "strategy" class. This makes the system
    modular and easy to extend with new techniques.

Usage:
    .\.venv\Scripts\python.exe src\_02_define_schema.py

Example:
    >>> from _02_define_schema import get_strategy, ExtractionSignature
    >>> strategy = get_strategy("chain_of_thought")
    >>> ExtractionSignature.__doc__ = strategy.get_docstring()
"""

from abc import ABC, abstractmethod

import dspy
from pydantic import BaseModel, Field

# --- Pydantic Output Model ---
# This defines the structured data we want the LLM to return.
# Using Pydantic ensures the output is type-checked and valid.


class VehicleInfo(BaseModel):
    """
    Structured representation of vehicle information extracted from automotive complaints or narratives.

    This Pydantic model defines the exact schema for vehicle data that the LLM should extract.
    It ensures type safety and provides clear field descriptions for the extraction process.

    Attributes:
        make (str): Vehicle manufacturer name (e.g., 'Tesla', 'Toyota', 'Ford').
                   Use 'UNKNOWN' if not found in the text.
        model (str): Vehicle model name (e.g., 'Model 3', 'Camry', 'F-150').
                    Use 'UNKNOWN' if not found in the text.
        year (str): Vehicle year as 4-digit string (e.g., '2023', '2024').
                   Use 'UNKNOWN' if not found in the text.

    Example:
        >>> vehicle = VehicleInfo(make="Tesla", model="Model Y", year="2023")
        >>> print(vehicle.make)
        Tesla
    """

    make: str = Field(
        description="Vehicle manufacturer name. Examples: 'Tesla', 'Toyota', 'Ford', 'BMW', 'Mercedes-Benz', 'Chevrolet', 'KIA', 'Rivian', 'Subaru'. Extract the exact brand name as mentioned in the text. If not found, use 'UNKNOWN'."
    )
    model: str = Field(
        description="Vehicle model name. Examples: 'Model 3', 'Model Y', 'Camry', 'F-150', 'Mustang', 'Accord', 'Civic'. Extract the specific model as mentioned. If not found, use 'UNKNOWN'."
    )
    year: str = Field(
        description="Vehicle year as 4-digit string. Examples: '2019', '2020', '2021', '2022', '2023', '2024', '2025'. Look for phrases like '2023 Tesla', 'a 2021 model', etc. If not found, use 'UNKNOWN'."
    )


# --- Base DSPy Signature ---
# This defines the input and output fields for our LLM call.


class ExtractionSignature(dspy.Signature):
    """
    DSPy Signature for extracting structured vehicle information from unstructured narratives.

    This signature defines the input/output contract for the LLM extraction task.
    The specific instructions for extraction are dynamically set by different
    prompting strategies to enable experimentation with various approaches.

    Attributes:
        narrative (str): Input field containing the unstructured vehicle complaint text.
        reasoning (str): Output field containing the step-by-step thought process.
        vehicle_info (VehicleInfo): Output field containing the extracted structured data.

    Note:
        The docstring of this class is dynamically updated by prompting strategies
        to provide specific extraction instructions to the LLM.
    """

    narrative: str = dspy.InputField(
        desc="A detailed, unstructured description of a vehicle complaint."
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning process showing how you identified the vehicle information. Explain your thought process, what clues you found, and any uncertainties."
    )
    vehicle_info: VehicleInfo = dspy.OutputField(
        desc="The structured vehicle information."
    )


# --- Strategy Pattern for Prompting Techniques ---


class PromptStrategy(ABC):
    """
    Abstract base class for prompting strategies in the vehicle extraction pipeline.

    This class defines the interface for all prompting strategies, enabling
    different approaches to be swapped in and out easily. Each concrete strategy
    provides specific instructions for how the LLM should extract vehicle information.

    The strategy pattern allows for experimentation with various prompting techniques
    like Chain of Thought, Self-Refine, Plan and Solve, etc.
    """

    @abstractmethod
    def get_docstring(self) -> str:
        """
        Returns the instruction docstring to be used for the DSPy Signature.

        Returns:
            str: A detailed instruction string that will be used as the signature's
                docstring, providing specific guidance to the LLM on how to perform
                the extraction task.
        """
        pass


class NaivePrompt(PromptStrategy):
    """
    A simple, direct instruction strategy for vehicle information extraction.

    This strategy provides straightforward extraction instructions with clear examples.
    It serves as a baseline approach without complex reasoning steps, making it
    ideal for comparison with more sophisticated prompting techniques.
    """

    def get_docstring(self) -> str:
        """
        Returns simple, direct extraction instructions with examples.

        Returns:
            str: Basic extraction instructions with clear input/output examples.
        """
        return """Extract the vehicle make, model, and year from the automotive complaint text.

Examples:
- "2023 Tesla Model Y" → Make: Tesla, Model: Model Y, Year: 2023
- "KIA NIRO 2023MY" → Make: KIA, Model: NIRO, Year: 2023
- "Ford F-150 from 2021" → Make: Ford, Model: F-150, Year: 2021

First, provide your reasoning explaining what text clues you found for each field and any uncertainties you encountered. Then extract the structured data.

Use "UNKNOWN" if any information is not clearly mentioned."""


class ChainOfThought(PromptStrategy):
    """
    Chain of Thought prompting strategy for step-by-step reasoning.

    This strategy instructs the model to break down the extraction task into
    sequential steps, encouraging explicit reasoning about each piece of information.
    This approach often leads to better accuracy by making the reasoning process
    more transparent and systematic.
    """

    def get_docstring(self) -> str:
        """
        Returns step-by-step extraction instructions encouraging explicit reasoning.

        Returns:
            str: Instructions that guide the model through a systematic extraction process.
        """
        return """Let's extract vehicle information step by step:

1. MAKE: Search for vehicle manufacturer names like Tesla, Toyota, Ford, BMW, etc.
2. MODEL: Look for specific model names like Model 3, Camry, F-150, etc.
3. YEAR: Find the model year, often mentioned as a 4-digit number (2019-2025).

First, provide your reasoning showing your thought process, then extract the structured data.

In your reasoning, explain:
- What specific text clues you found for each field
- Any ambiguities or uncertainties you encountered
- Why you made your final decisions

Be careful to distinguish between:
- Make vs Model (e.g., "Tesla Model 3" → Make="Tesla", Model="Model 3")
- Year vs other numbers (mileage, speed, etc.)

If any information is not clearly stated, use "UNKNOWN" for that field."""


class PlanAndSolve(PromptStrategy):
    """
    Plan-and-Solve prompting strategy that separates planning from execution.

    This strategy instructs the model to first devise a comprehensive plan for
    the extraction task, then execute that plan systematically. This two-phase
    approach can improve performance by encouraging more deliberate and organized
    reasoning processes.
    """

    def get_docstring(self) -> str:
        """
        Returns instructions for planning the extraction approach before execution.

        Returns:
            str: Two-phase instructions covering planning and execution steps.
        """
        return """Plan: I will extract vehicle make, model, and year from this automotive complaint text.

Step 1: Identify the vehicle manufacturer (make) - look for brand names
Step 2: Identify the specific model - look for model names after the make
Step 3: Identify the year - look for 4-digit years, often before or after the make/model

Execution: Now I'll carefully read the text and extract each piece of information, using "UNKNOWN" if any detail is not clearly mentioned.

Provide your reasoning showing:
- How you planned your approach
- What specific evidence you found during execution
- Any challenges or ambiguities you encountered
- How your plan helped you reach your final decisions

Then provide the extracted structured data."""


class SelfRefine(PromptStrategy):
    """
    Self-Refine prompting strategy for iterative improvement.

    This strategy instructs the model to generate an initial extraction, critique
    its own work, and then refine the results. This self-correction approach can
    lead to higher accuracy by catching and correcting initial mistakes or
    oversights in the extraction process.
    """

    def get_docstring(self) -> str:
        """
        Returns instructions for draft-critique-refine extraction process.

        Returns:
            str: Three-phase instructions for iterative self-improvement.
        """
        return """Step 1 - DRAFT: First, extract what you think is the vehicle's make, model, and year from the text.

Step 2 - CRITIQUE: Review your draft extraction:
- Is the make actually a vehicle manufacturer (Tesla, Ford, BMW, etc.)?
- Is the model a specific vehicle model (Model 3, F-150, Camry, etc.)?
- Is the year a valid 4-digit vehicle year (2015-2025)?
- Did I confuse any numbers (mileage, speed) with the model year?

Step 3 - REFINE: Based on your critique, provide the final, corrected extraction. Use "UNKNOWN" for any field you cannot confidently determine.

Show your complete reasoning process including:
- Your initial draft and the evidence you found
- Your self-critique and what issues you identified
- Your final refinement and what you changed and why

Then provide the final extracted structured data."""


class ContrastiveCoT(PromptStrategy):
    """
    Contrastive Chain of Thought strategy using positive and negative examples.

    This strategy provides both correct and incorrect reasoning examples to help
    the model understand what to avoid. By showing both good and bad approaches,
    it helps the model distinguish between relevant and irrelevant information
    more effectively.
    """

    def get_docstring(self) -> str:
        """
        Returns instructions with contrasting good and bad reasoning examples.

        Returns:
            str: Instructions featuring both positive and negative examples
                 to guide proper reasoning.
        """
        return """To extract vehicle information correctly, you must distinguish between relevant and irrelevant details.

GOOD REASONING EXAMPLE:
Text: "The owner of a 2023MY Kia Niro contacted customer care..."
Analysis: "2023MY" indicates model year 2023, "Kia" is the manufacturer, "Niro" is the model.
Result: Make=Kia, Model=Niro, Year=2023

BAD REASONING EXAMPLE:
Text: "Vehicle was traveling at 65 mph with 50,000 miles on it"
Wrong Analysis: "The year must be 65 because that's a number I see"
Correct Analysis: "65 mph is speed, 50,000 is mileage, no model year mentioned"
Result: Make=UNKNOWN, Model=UNKNOWN, Year=UNKNOWN

Now analyze the following text using good reasoning principles.

Provide your reasoning showing:
- How you applied the good reasoning principles
- What relevant vs irrelevant details you identified
- How you avoided the bad reasoning patterns shown above
- Your step-by-step analysis leading to your conclusions

Then provide the extracted structured data."""


# Factory to get a strategy by name
PROMPT_STRATEGIES = {
    "naive": NaivePrompt(),
    "cot": ChainOfThought(),
    "plan_and_solve": PlanAndSolve(),
    "self_refine": SelfRefine(),
    "contrastive_cot": ContrastiveCoT(),
}


def get_strategy(name: str) -> PromptStrategy:
    """
    Retrieves a prompt strategy instance by its name.

    This factory function provides access to different prompting strategies
    by name, enabling easy experimentation and comparison between approaches.

    Args:
        name (str): The name of the strategy to retrieve. Must be one of:
                   'naive', 'cot', 'plan_and_solve', 'self_refine', 'contrastive_cot'.
                   Case-insensitive.

    Returns:
        PromptStrategy: An instance of the requested prompting strategy.

    Raises:
        ValueError: If the strategy name is not recognized.

    Example:
        >>> strategy = get_strategy("chain_of_thought")
        >>> docstring = strategy.get_docstring()
        >>> print(docstring[:50])
        Let's extract vehicle information step by step:
    """
    strategy = PROMPT_STRATEGIES.get(name.lower())
    if not strategy:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {list(PROMPT_STRATEGIES.keys())}"
        )
    return strategy


if __name__ == "__main__":
    # This block allows you to run this script directly to test the strategies
    print("Testing Prompting Strategies...")
    for name in PROMPT_STRATEGIES:
        strategy = get_strategy(name)
        print(f"\n--- Strategy: {name} ---")
        print(strategy.get_docstring())

    # Example of how to dynamically set the docstring for the signature
    selected_strategy = get_strategy("cot")
    ExtractionSignature.__doc__ = selected_strategy.get_docstring()
    print("\n--- Dynamically updated signature docstring ---")
    print(ExtractionSignature.__doc__)
