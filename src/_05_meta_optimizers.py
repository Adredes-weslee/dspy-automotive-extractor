"""
_05_meta_optimizers.py

This module implements meta-optimization techniques that enhance base prompting strategies
with established best practices for prompt engineering. Each meta-optimizer applies
specific improvement patterns to base instructions based on research-backed techniques.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from _02_define_schema import PromptStrategy


class MetaOptimizer(ABC):
    """Abstract base class for meta-optimization techniques."""

    @abstractmethod
    def get_name(self) -> str:
        """Returns the name of this meta-optimizer."""
        pass

    @abstractmethod
    def optimize_instruction(self, base_instruction: str, strategy_name: str) -> str:
        """
        Enhances a base instruction with meta-optimization techniques.

        Args:
            base_instruction: The original strategy instruction
            strategy_name: Name of the base strategy (for context)

        Returns:
            Enhanced instruction string
        """
        pass


class SpecificityEnhancer(MetaOptimizer):
    """
    Enhances instructions with specific requirements and constraints.

    Based on research showing that vague instructions lead to inconsistent outputs.
    Makes abstract requirements concrete and measurable.
    """

    def get_name(self) -> str:
        return "specificity"

    def optimize_instruction(self, base_instruction: str, strategy_name: str) -> str:
        return f"""
{base_instruction}

SPECIFIC EXTRACTION REQUIREMENTS:
- Extract EXACTLY 3 fields: make, model, year
- Make: Vehicle manufacturer in ALL CAPS (e.g., 'TESLA', 'FORD', 'HONDA')
- Model: Vehicle model in ALL CAPS (e.g., 'MODEL Y', 'F-150', 'CIVIC')
- Year: 4-digit integer only (e.g., 2022, 2019, 2021)
- Output must be valid JSON format
- Do not include explanatory text outside the JSON structure
- Field names must be exactly: "make", "model", "year"
"""


class ErrorPreventionWrapper(MetaOptimizer):
    """
    Adds error prevention guidelines based on observed failure patterns.

    Implements Constitutional AI principles by explicitly stating what NOT to do.
    Addresses common extraction errors through negative constraints.
    """

    def get_name(self) -> str:
        return "error_prevention"

    def optimize_instruction(self, base_instruction: str, strategy_name: str) -> str:
        return f"""
{base_instruction}

CRITICAL ERROR PREVENTION RULES:
- Do NOT confuse trim levels (LX, EX, Sport, Limited) with model names
- Do NOT extract partial years (e.g., '22' instead of 2022)
- Do NOT include sub-models or variants as separate models
- Do NOT extract generic terms like 'car', 'truck', 'vehicle' as make/model
- Do NOT confuse Honda with Acura (they are separate makes)
- Do NOT extract speed, mileage, or other numbers as years
- Do NOT include quotation marks in the final field values
- If information is unclear or missing, use 'UNKNOWN' for that field
- Do NOT guess or infer information not explicitly stated
"""


class ContextAnchoringOptimizer(MetaOptimizer):
    """
    Provides domain-specific context and expertise framing.

    Uses role-playing technique to improve domain understanding.
    Anchors the task in real-world context for better performance.
    """

    def get_name(self) -> str:
        return "context_anchoring"

    def optimize_instruction(self, base_instruction: str, strategy_name: str) -> str:
        return f"""
You are an expert automotive data analyst specializing in NHTSA vehicle complaint analysis.
Your task is to extract precise vehicle identification information from consumer complaints.

{base_instruction}

DOMAIN EXPERTISE CONTEXT:
- You are analyzing official NHTSA (National Highway Traffic Safety Administration) complaints
- These complaints often contain detailed narratives about vehicle incidents
- Vehicle identification is crucial for safety recall analysis and defect investigations
- Accuracy in make/model/year extraction directly impacts public safety outcomes
- You have extensive knowledge of automotive manufacturers and their model lineups
- You understand the difference between makes, models, trim levels, and model years
"""


class OutputFormatEnforcer(MetaOptimizer):
    """
    Enforces strict output format requirements using imperative language.

    Based on instruction-following research showing that explicit format
    constraints improve consistency and parsability.
    """

    def get_name(self) -> str:
        return "format_enforcement"

    def optimize_instruction(self, base_instruction: str, strategy_name: str) -> str:
        return f"""
{base_instruction}

MANDATORY OUTPUT FORMAT - YOU MUST FOLLOW THIS EXACTLY:
You MUST respond with ONLY a JSON object in this EXACT format:
{{
  "make": "MANUFACTURER_NAME",
  "model": "MODEL_NAME", 
  "year": 2022
}}

FORMAT REQUIREMENTS - NO EXCEPTIONS:
- Use double quotes for JSON keys and string values
- Make and model values MUST be in ALL CAPITALS
- Year MUST be a 4-digit integer (no quotes)
- No additional text, explanations, or commentary
- No markdown formatting or code blocks
- Response must be valid JSON that can be parsed directly
- Do not wrap in ```json``` blocks
"""


class MultiShotReasoningEnhancer(MetaOptimizer):
    """
    Adds structured reasoning steps and examples.

    Implements few-shot learning with explicit reasoning chains.
    Shows the model how to think through the extraction process.
    """

    def get_name(self) -> str:
        return "multishot_reasoning"

    def optimize_instruction(self, base_instruction: str, strategy_name: str) -> str:
        return f"""
{base_instruction}

STRUCTURED REASONING PROCESS:
1. SCAN the text for manufacturer keywords (Tesla, Ford, Honda, Toyota, etc.)
2. IDENTIFY model indicators (usually follows make, or mentioned separately)
3. LOCATE year information (often at beginning or with specific model reference)
4. VALIDATE extracted information for consistency and completeness
5. FORMAT the output according to the required JSON structure

EXAMPLE EXTRACTION PATTERNS:
- "2022 Tesla Model Y" → make: "TESLA", model: "MODEL Y", year: 2022
- "Ford F-150 truck from 2021" → make: "FORD", model: "F-150", year: 2021
- "My Honda Civic (2020 model)" → make: "HONDA", model: "CIVIC", year: 2020
- "65 mph in my car" → make: "UNKNOWN", model: "UNKNOWN", year: "UNKNOWN"

REASONING VALIDATION:
- Does the make match a known automotive manufacturer?
- Is the model a real vehicle model from that manufacturer?
- Is the year realistic (typically 1990-2025)?
"""


class ConstitutionalConstraints(MetaOptimizer):
    """
    Implements Constitutional AI principles with explicit behavioral constraints.

    Creates a hierarchy of rules that the model must follow, with
    emphasis on accuracy and consistency over speed.
    """

    def get_name(self) -> str:
        return "constitutional"

    def optimize_instruction(self, base_instruction: str, strategy_name: str) -> str:
        return f"""
{base_instruction}

CONSTITUTIONAL CONSTRAINTS - HIERARCHICAL RULES:

RULE 1 (HIGHEST PRIORITY): Accuracy over Speed
- Take time to carefully analyze the text
- Double-check your extraction before outputting
- Prefer 'UNKNOWN' over incorrect guesses

RULE 2: Consistency over Creativity  
- Follow the exact same extraction pattern for every input
- Use standardized capitalization (ALL CAPS for make/model)
- Maintain consistent field naming

RULE 3: Explicitness over Inference
- Only extract information explicitly stated in the text
- Do not infer missing information from context
- Do not assume default values

RULE 4: Structure over Flexibility
- Always output valid JSON
- Always include all three fields (make, model, year)
- Never deviate from the required format
"""


class DomainExpertiseInjector(MetaOptimizer):
    """
    Injects specific automotive domain knowledge and common patterns.

    Provides the model with expert knowledge about automotive naming
    conventions and common extraction challenges.
    """

    def get_name(self) -> str:
        return "domain_expertise"

    def optimize_instruction(self, base_instruction: str, strategy_name: str) -> str:
        return f"""
{base_instruction}

AUTOMOTIVE DOMAIN EXPERTISE:

COMMON MANUFACTURER VARIATIONS:
- GM → CHEVROLET, GMC, CADILLAC, BUICK
- FCA → CHRYSLER, DODGE, JEEP, RAM
- Honda Motor Company → HONDA, ACURA
- Toyota Motor Corporation → TOYOTA, LEXUS
- Hyundai Motor Group → HYUNDAI, KIA, GENESIS

MODEL NAMING PATTERNS:
- Alphanumeric: F-150, Model 3, CX-5, Q7
- Word-based: Camry, Accord, Explorer, Pilot
- Compound: Grand Cherokee, Model Y, Santa Fe

YEAR EXTRACTION HINTS:
- Often appears at beginning: "2022 Honda Civic"
- Sometimes in parentheses: "Honda Civic (2022)"
- May be abbreviated: "'22 Civic" → 2022
- Model year vs calendar year: usually model year in complaints

TRIM LEVELS TO IGNORE (NOT MODELS):
- Base, LX, EX, EX-L, Touring, Sport, Limited
- S, SE, SEL, SXT, R/T, SRT, Hellcat
- LE, XLE, XSE, TRD, Limited, Platinum
"""


# Factory for meta-optimizers
META_OPTIMIZERS = {
    "specificity": SpecificityEnhancer(),
    "error_prevention": ErrorPreventionWrapper(),
    "context_anchoring": ContextAnchoringOptimizer(),
    "format_enforcement": OutputFormatEnforcer(),
    "multishot_reasoning": MultiShotReasoningEnhancer(),
    "constitutional": ConstitutionalConstraints(),
    "domain_expertise": DomainExpertiseInjector(),
}


class MetaOptimizedStrategy(PromptStrategy):
    """A strategy that combines a base strategy with meta-optimizations."""

    def __init__(
        self, base_strategy: PromptStrategy, meta_optimizers: List[MetaOptimizer]
    ):
        self.base_strategy = base_strategy
        self.meta_optimizers = meta_optimizers
        self.strategy_name = self._generate_name()

    def _generate_name(self) -> str:
        """Generate a name for this meta-optimized strategy."""
        base_name = (
            self.base_strategy.__class__.__name__.lower()
            .replace("prompt", "")
            .replace("strategy", "")
        )
        meta_names = [opt.get_name() for opt in self.meta_optimizers]
        return f"{base_name}_{'_'.join(meta_names)}"

    def get_docstring(self) -> str:
        """Apply all meta-optimizations to the base strategy."""
        instruction = self.base_strategy.get_docstring()

        for optimizer in self.meta_optimizers:
            instruction = optimizer.optimize_instruction(
                instruction, self.strategy_name
            )

        return instruction


def create_meta_optimized_strategies() -> Dict[str, PromptStrategy]:
    """Create a comprehensive set of meta-optimized strategies based on research best practices."""
    from _02_define_schema import PROMPT_STRATEGIES

    meta_strategies = {}

    # Single meta-optimizations for each base strategy
    for base_name, base_strategy in PROMPT_STRATEGIES.items():
        for meta_name, meta_optimizer in META_OPTIMIZERS.items():
            strategy_name = f"{base_name}_{meta_name}"
            meta_strategies[strategy_name] = MetaOptimizedStrategy(
                base_strategy, [meta_optimizer]
            )

    # Research-backed combinations (most promising based on prompt engineering literature)
    high_impact_combinations = [
        # Core combination: Specificity + Error Prevention
        ("specificity", "error_prevention"),
        # Context + Format (role-playing with structure)
        ("context_anchoring", "format_enforcement"),
        # Constitutional + Domain (rule-based with expertise)
        ("constitutional", "domain_expertise"),
        # Complete reasoning enhancement
        ("specificity", "multishot_reasoning"),
        # Error prevention with format enforcement
        ("error_prevention", "format_enforcement"),
        # Triple combinations for maximum enhancement
        ("context_anchoring", "specificity", "error_prevention"),
        ("constitutional", "domain_expertise", "format_enforcement"),
        ("specificity", "error_prevention", "multishot_reasoning"),
        # Ultimate combination (use sparingly - may be too verbose)
        ("context_anchoring", "specificity", "error_prevention", "format_enforcement"),
    ]

    for base_name, base_strategy in PROMPT_STRATEGIES.items():
        for combo in high_impact_combinations:
            optimizers = [META_OPTIMIZERS[opt_name] for opt_name in combo]
            strategy_name = f"{base_name}_{'_'.join(combo)}"
            meta_strategies[strategy_name] = MetaOptimizedStrategy(
                base_strategy, optimizers
            )

    return meta_strategies


def get_priority_meta_strategies() -> List[str]:
    """
    Returns a curated list of the most promising meta-optimized strategies.

    Based on prompt engineering research and expected performance gains.
    Use this for focused experimentation when you don't want to run all combinations.
    """
    return [
        # Best single meta-optimizations
        "naive_specificity",
        "naive_error_prevention",
        "cot_multishot_reasoning",
        "contrastive_cot_domain_expertise",
        # Best dual combinations
        "naive_specificity_error_prevention",
        "cot_context_anchoring_format_enforcement",
        "plan_and_solve_constitutional_domain_expertise",
        # Best triple combinations
        "naive_context_anchoring_specificity_error_prevention",
        "contrastive_cot_constitutional_domain_expertise_format_enforcement",
        # Ultimate enhanced strategy
        "cot_context_anchoring_specificity_error_prevention_format_enforcement",
    ]


if __name__ == "__main__":
    # Test the meta-optimizers
    from _02_define_schema import ChainOfThought

    print("=== Meta-Optimizer Testing ===\n")

    base_strategy = ChainOfThought()
    print("Base Strategy:")
    print(base_strategy.get_docstring())
    print("\n" + "=" * 80 + "\n")

    # Test comprehensive meta-optimization
    enhanced_strategy = MetaOptimizedStrategy(
        base_strategy,
        [
            SpecificityEnhancer(),
            ErrorPreventionWrapper(),
            ContextAnchoringOptimizer(),
            OutputFormatEnforcer(),
        ],
    )

    print("Meta-Optimized Strategy:")
    print(enhanced_strategy.get_docstring())

    print("\n" + "=" * 80 + "\n")
    print("Available Meta-Optimized Strategies:")
    meta_strategies = create_meta_optimized_strategies()
    print(f"Total strategies: {len(meta_strategies)}")

    print("\nPriority strategies for focused testing:")
    for name in get_priority_meta_strategies():
        print(f"  - {name}")

    print("\nAll available strategies:")
    for name in sorted(meta_strategies.keys()):
        print(f"  - {name}")
