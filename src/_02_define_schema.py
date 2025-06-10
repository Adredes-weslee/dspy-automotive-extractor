"""
_02_define_schema.py

This script defines the data structures for our LLM pipeline. It includes:
1.  A Pydantic model (`VehicleInfo`) that defines the strict, typed schema for
    the structured data we want to extract.
2.  A base DSPy Signature (`ExtractionSignature`) that defines the high-level
    input/output contract for the LLM.
3.  The Strategy Pattern implementation for Prompting Techniques. This allows us
    to easily switch between different prompt instructions (e.g., CoT, Self-Refine)
    by defining each as a separate "strategy" class. This makes the system
    modular and easy to extend with new techniques.
"""
import dspy
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

# --- Pydantic Output Model ---
# This defines the structured data we want the LLM to return.
# Using Pydantic ensures the output is type-checked and valid.

class VehicleInfo(BaseModel):
    """Structured representation of vehicle information extracted from automotive complaints or narratives."""
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
    Extracts structured vehicle information from an unstructured narrative.
    The specific instructions for extraction will be dynamically set by a strategy.
    """
    narrative: str = dspy.InputField(desc="A detailed, unstructured description of a vehicle complaint.")
    vehicle_info: VehicleInfo = dspy.OutputField(desc="The structured vehicle information.")


# --- Strategy Pattern for Prompting Techniques ---

class PromptStrategy(ABC):
    """Abstract base class for a prompting strategy."""
    @abstractmethod
    def get_docstring(self) -> str:
        """Returns the docstring to be used for the DSPy Signature."""
        pass

class NaivePrompt(PromptStrategy):
    """A simple, direct instruction for extraction with clear examples."""
    def get_docstring(self) -> str:
        return """Extract the vehicle make, model, and year from the automotive complaint text.

Examples:
- "2023 Tesla Model Y" → Make: Tesla, Model: Model Y, Year: 2023
- "KIA NIRO 2023MY" → Make: KIA, Model: NIRO, Year: 2023
- "Ford F-150 from 2021" → Make: Ford, Model: F-150, Year: 2021

Use "UNKNOWN" if any information is not clearly mentioned."""

class ChainOfThought(PromptStrategy):
    """Instructs the model to reason step-by-step."""
    def get_docstring(self) -> str:
        return """Let's extract vehicle information step by step:

1. MAKE: Search for vehicle manufacturer names like Tesla, Toyota, Ford, BMW, etc.
2. MODEL: Look for specific model names like Model 3, Camry, F-150, etc.
3. YEAR: Find the model year, often mentioned as a 4-digit number (2019-2025).

Be careful to distinguish between:
- Make vs Model (e.g., "Tesla Model 3" → Make="Tesla", Model="Model 3")
- Year vs other numbers (mileage, speed, etc.)

If any information is not clearly stated, use "UNKNOWN" for that field."""

class PlanAndSolve(PromptStrategy):
    """Instructs the model to first devise a plan and then execute it."""
    def get_docstring(self) -> str:
        return """Plan: I will extract vehicle make, model, and year from this automotive complaint text.

Step 1: Identify the vehicle manufacturer (make) - look for brand names
Step 2: Identify the specific model - look for model names after the make
Step 3: Identify the year - look for 4-digit years, often before or after the make/model

Execution: Now I'll carefully read the text and extract each piece of information, using "UNKNOWN" if any detail is not clearly mentioned."""

class SelfRefine(PromptStrategy):
    """Instructs the model to generate a draft and then critique it."""
    def get_docstring(self) -> str:
        return """Step 1 - DRAFT: First, extract what you think is the vehicle's make, model, and year from the text.

Step 2 - CRITIQUE: Review your draft extraction:
- Is the make actually a vehicle manufacturer (Tesla, Ford, BMW, etc.)?
- Is the model a specific vehicle model (Model 3, F-150, Camry, etc.)?
- Is the year a valid 4-digit vehicle year (2015-2025)?
- Did I confuse any numbers (mileage, speed) with the model year?

Step 3 - REFINE: Based on your critique, provide the final, corrected extraction. Use "UNKNOWN" for any field you cannot confidently determine."""

class ContrastiveCoT(PromptStrategy):
    """Provides both good and bad examples of reasoning."""
    def get_docstring(self) -> str:
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

Now analyze the following text using good reasoning principles."""

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

    Args:
        name (str): The name of the strategy.

    Returns:
        PromptStrategy: An instance of the requested strategy.
    """
    strategy = PROMPT_STRATEGIES.get(name.lower())
    if not strategy:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(PROMPT_STRATEGIES.keys())}")
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
