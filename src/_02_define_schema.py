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
    """Pydantic model for the structured vehicle information."""
    make: str = Field(description="The manufacturer of the vehicle, e.g., 'TESLA', 'FORD'. Should be in all caps.")
    model: str = Field(description="The model of the vehicle, e.g., 'MODEL Y', 'F-150'. Should be in all caps.")
    year: int = Field(description="The model year of the vehicle, e.g., 2022.")

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
    """A simple, direct instruction for extraction."""
    def get_docstring(self) -> str:
        return "Extract the vehicle make, model, and year from the text."

class ChainOfThought(PromptStrategy):
    """Instructs the model to reason step-by-step."""
    def get_docstring(self) -> str:
        return "Let's think step by step. First, identify the vehicle's make. Second, identify its model. Third, find the model year. Finally, provide the structured output."

class PlanAndSolve(PromptStrategy):
    """Instructs the model to first devise a plan and then execute it."""
    def get_docstring(self) -> str:
        return "First, devise a plan to extract the vehicle's make, model, and year. Then, execute the plan, detailing each step of the extraction to arrive at the final answer."

class SelfRefine(PromptStrategy):
    """Instructs the model to generate a draft and then critique it."""
    def get_docstring(self) -> str:
        return "Generate a draft extraction of the vehicle's make, model, and year. Then, critique your draft for accuracy and completeness. Finally, based on your critique, provide a final, refined structured answer."

class ContrastiveCoT(PromptStrategy):
    """Provides both a good and a bad example of reasoning."""
    def get_docstring(self) -> str:
        return "To extract the vehicle's make, model, and year, you must reason correctly. A good example of reasoning is: 'The text mentions a 2022 Tesla Model Y. Therefore, the make is Tesla, the model is Model Y, and the year is 2022.' A bad example is: 'The text mentions a steering wheel, so the make is car.' Now, analyze the following text."

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
