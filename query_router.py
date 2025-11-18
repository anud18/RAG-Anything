#!/usr/bin/env python
"""
Query Router for Adaptive RAG

This module implements an intelligent query complexity classifier based on the
Adaptive-RAG paper. It analyzes user queries and routes them to optimal retrieval
strategies based on their complexity.

Key Features:
- LLM-based query complexity classification (simple/moderate/complex)
- Dynamic mode selection based on query characteristics
- Configurable retrieval strategies
- Support for custom classification criteria

Usage:
    from query_router import QueryRouter

    router = QueryRouter(llm_func=your_llm_function)
    result = await router.classify_query("給我PCD急救人員名單")
    modes = router.select_modes(result['complexity'])
"""

import logging
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
import asyncio


@dataclass
class QueryComplexity:
    """Data class representing query complexity analysis result"""
    complexity: str  # "simple", "moderate", or "complex"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    recommended_modes: List[str]
    raw_classification: str = ""


@dataclass
class RouterConfig:
    """Configuration for Query Router"""
    # Mode strategies for each complexity level
    simple_modes: List[str] = field(default_factory=lambda: ["naive", "local"])
    moderate_modes: List[str] = field(default_factory=lambda: ["hybrid", "local"])
    complex_modes: List[str] = field(default_factory=lambda: ["mix", "global", "hybrid"])

    # Valid mode names
    valid_modes: List[str] = field(default_factory=lambda: ["local", "global", "hybrid", "naive", "mix"])

    # Classification confidence threshold
    confidence_threshold: float = 0.3

    # Custom classification prompt (optional)
    custom_prompt_template: Optional[str] = None


class QueryRouter:
    """
    Intelligent query router for Adaptive RAG.

    Classifies queries into complexity levels and selects optimal retrieval modes.
    Based on: "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
    Models through Question Complexity"
    """

    DEFAULT_CLASSIFICATION_PROMPT = """Analyze the following user query and classify its complexity level for RAG retrieval.

User Query: {query}

Classification Criteria:

1. SIMPLE queries:
   - Direct factual questions (Who, What, When, Where)
   - Single entity lookup
   - Straightforward information retrieval
   - Example: "給我PCD急救人員名單", "What is the capital of France?"
   - Recommended: Naive or Local mode (simple vector search)

2. MODERATE queries:
   - Requires combining multiple facts
   - Comparison or analysis of 2-3 entities
   - Requires some reasoning
   - Example: "比較A和B的差異", "How does X affect Y?"
   - Recommended: Hybrid mode (combines local and global)

3. COMPLEX queries:
   - Multi-hop reasoning required
   - Requires synthesizing information across multiple sources
   - Abstract concepts or relationships
   - Analytical or summarization tasks
   - Example: "分析整體趨勢", "What are the implications of X on Y and Z?"
   - Recommended: Mix or Global mode (knowledge graph + vector search)

Please analyze the query and provide your classification in this EXACT format:
Complexity: [simple/moderate/complex]
Confidence: [0.0-1.0]
Reasoning: [Brief explanation of why you classified it this way]
Recommended Modes: [comma-separated list of modes]

Important: Be precise and consistent with the format."""

    DEFAULT_SYSTEM_PROMPT = "You are an expert query analyzer. Classify queries accurately and consistently."

    def __init__(
        self,
        llm_func: Callable,
        config: Optional[RouterConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Query Router.

        Args:
            llm_func: Async LLM function that takes (prompt, system_prompt) and returns string
            config: Router configuration (uses defaults if not provided)
            logger: Logger instance (creates new if not provided)
        """
        self.llm_func = llm_func
        self.config = config or RouterConfig()
        self.logger = logger or logging.getLogger(__name__)

    async def classify_query(self, query: str) -> QueryComplexity:
        """
        Classify query complexity using LLM.

        Args:
            query: User query string

        Returns:
            QueryComplexity object with classification results

        Example:
            result = await router.classify_query("給我PCD急救人員名單")
            print(f"Complexity: {result.complexity}")
            print(f"Confidence: {result.confidence}")
            print(f"Modes: {result.recommended_modes}")
        """
        # Use custom prompt template if provided
        prompt_template = (
            self.config.custom_prompt_template
            if self.config.custom_prompt_template
            else self.DEFAULT_CLASSIFICATION_PROMPT
        )

        classification_prompt = prompt_template.format(query=query)

        try:
            classification = await self.llm_func(
                classification_prompt,
                system_prompt=self.DEFAULT_SYSTEM_PROMPT
            )

            # Parse the classification result
            complexity = "moderate"  # default
            confidence = 0.5
            reasoning = ""
            recommended_modes = ["hybrid"]

            lines = classification.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Complexity:"):
                    complexity = line.split(":", 1)[1].strip().lower()
                elif line.startswith("Confidence:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        confidence = 0.5
                elif line.startswith("Reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.startswith("Recommended Modes:"):
                    modes_str = line.split(":", 1)[1].strip()
                    recommended_modes = [m.strip().lower() for m in modes_str.split(",")]

            return QueryComplexity(
                complexity=complexity,
                confidence=confidence,
                reasoning=reasoning,
                recommended_modes=recommended_modes,
                raw_classification=classification
            )

        except Exception as e:
            self.logger.error(f"Error in query classification: {str(e)}")
            # Fallback to moderate complexity
            return QueryComplexity(
                complexity="moderate",
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}. Defaulting to moderate.",
                recommended_modes=["hybrid"],
                raw_classification=""
            )

    def select_modes(
        self,
        complexity: str,
        recommended_modes: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select retrieval modes based on complexity.

        Args:
            complexity: Query complexity ("simple", "moderate", or "complex")
            recommended_modes: Optional list of modes from LLM (overrides defaults)

        Returns:
            List of selected mode names

        Example:
            modes = router.select_modes("simple")
            # Returns: ["naive", "local"]
        """
        # Use recommended modes if available and valid
        if recommended_modes:
            filtered = [m for m in recommended_modes if m in self.config.valid_modes]
            if filtered:
                return filtered

        # Fallback to configured strategies
        mode_strategy = {
            "simple": self.config.simple_modes,
            "moderate": self.config.moderate_modes,
            "complex": self.config.complex_modes
        }

        return mode_strategy.get(complexity, self.config.moderate_modes)

    async def route_query(self, query: str) -> Dict[str, Any]:
        """
        Complete routing pipeline: classify and select modes.

        Args:
            query: User query string

        Returns:
            Dictionary containing:
                - complexity: QueryComplexity object
                - selected_modes: List of mode names
                - efficiency_gain: Percentage of modes saved

        Example:
            result = await router.route_query("給我PCD急救人員名單")
            print(f"Use modes: {result['selected_modes']}")
            print(f"Saved: {result['efficiency_gain']:.0f}% API calls")
        """
        # Classify query
        classification = await self.classify_query(query)

        # Select modes
        selected_modes = self.select_modes(
            classification.complexity,
            classification.recommended_modes
        )

        # Calculate efficiency gain (assuming 5 total modes)
        total_modes = 5
        efficiency_gain = ((total_modes - len(selected_modes)) / total_modes) * 100

        return {
            "classification": classification,
            "selected_modes": selected_modes,
            "efficiency_gain": efficiency_gain
        }

    def update_config(self, **kwargs):
        """
        Update router configuration.

        Args:
            **kwargs: Configuration parameters to update
                - simple_modes: List[str]
                - moderate_modes: List[str]
                - complex_modes: List[str]
                - confidence_threshold: float

        Example:
            router.update_config(
                simple_modes=["naive"],
                complex_modes=["mix", "global"]
            )
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config parameter: {key}")


# Standalone utility functions for backward compatibility
async def classify_query_complexity(
    query: str,
    llm_func: Callable,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Standalone function to classify query complexity.

    Args:
        query: User query string
        llm_func: LLM function
        logger: Optional logger

    Returns:
        Dictionary with classification results
    """
    router = QueryRouter(llm_func=llm_func, logger=logger)
    result = await router.classify_query(query)

    return {
        "complexity": result.complexity,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "recommended_modes": result.recommended_modes,
        "raw_classification": result.raw_classification
    }


def select_modes_by_complexity(
    complexity: str,
    recommended_modes: Optional[List[str]] = None,
    config: Optional[RouterConfig] = None
) -> List[str]:
    """
    Standalone function to select modes based on complexity.

    Args:
        complexity: Query complexity level
        recommended_modes: Optional recommended modes
        config: Optional router configuration

    Returns:
        List of selected mode names
    """
    # Create a dummy LLM function (not used for mode selection)
    async def dummy_llm(prompt, system_prompt=None):
        return ""

    router = QueryRouter(llm_func=dummy_llm, config=config)
    return router.select_modes(complexity, recommended_modes)


# Example usage
async def example_usage():
    """Example demonstrating how to use QueryRouter"""

    # Mock LLM function for demonstration
    async def mock_llm_func(prompt: str, system_prompt: str = None) -> str:
        return """Complexity: simple
Confidence: 0.95
Reasoning: This is a direct factual question asking for a specific list of names.
Recommended Modes: naive, local"""

    # Initialize router
    router = QueryRouter(llm_func=mock_llm_func)

    # Example 1: Full routing pipeline
    query = "給我PCD急救人員名單"
    result = await router.route_query(query)

    print("="*60)
    print(f"Query: {query}")
    print("="*60)
    print(f"Complexity: {result['classification'].complexity}")
    print(f"Confidence: {result['classification'].confidence}")
    print(f"Reasoning: {result['classification'].reasoning}")
    print(f"Selected Modes: {', '.join(result['selected_modes'])}")
    print(f"Efficiency Gain: {result['efficiency_gain']:.0f}%")
    print()

    # Example 2: Custom configuration
    custom_config = RouterConfig(
        simple_modes=["naive"],
        moderate_modes=["hybrid"],
        complex_modes=["mix", "global"]
    )

    custom_router = QueryRouter(llm_func=mock_llm_func, config=custom_config)
    custom_result = await custom_router.route_query(query)

    print("With Custom Config:")
    print(f"Selected Modes: {', '.join(custom_result['selected_modes'])}")
    print()

    # Example 3: Update configuration dynamically
    router.update_config(simple_modes=["naive"])
    updated_result = await router.route_query(query)

    print("After Config Update:")
    print(f"Selected Modes: {', '.join(updated_result['selected_modes'])}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
