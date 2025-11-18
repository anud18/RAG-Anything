"""
Adaptive Query Router for RAGAnything

Implements Adaptive-RAG approach that routes queries to appropriate retrieval modes
based on query complexity and characteristics.

Based on the paper: "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity"
"""

from typing import Dict, Any, Callable, Optional
from enum import Enum
import json


class QueryComplexity(Enum):
    """Query complexity levels for adaptive routing"""
    SIMPLE = "simple"           # Simple factual questions
    MODERATE = "moderate"       # Questions requiring some analysis
    COMPLEX = "complex"         # Complex analytical or multi-hop questions


class QueryType(Enum):
    """Query type categories"""
    FACTUAL = "factual"                     # Specific facts lookup
    ANALYTICAL = "analytical"               # Requires analysis/reasoning
    SUMMARIZATION = "summarization"         # Requires summarization
    COMPARISON = "comparison"               # Comparing multiple items
    MULTI_HOP = "multi_hop"                # Requires multiple reasoning steps


# Prompt templates for query analysis
QUERY_ROUTER_PROMPTS = {
    "QUERY_ANALYSIS_SYSTEM": """You are a query analysis expert. Your task is to analyze user queries and classify them based on complexity and type to help route them to the appropriate retrieval strategy.""",

    "QUERY_ANALYSIS_PROMPT": """Analyze the following user query and provide a classification:

User Query: "{query}"

Please classify this query based on:

1. **Complexity Level:**
   - SIMPLE: Direct factual questions that can be answered with specific facts from the document
     Examples: "What is X?", "When did Y happen?", "Who is Z?"

   - MODERATE: Questions requiring some analysis or synthesis of information
     Examples: "How does X work?", "What are the differences between X and Y?", "Explain the relationship between X and Y"

   - COMPLEX: Questions requiring deep analysis, multiple reasoning steps, or comprehensive understanding
     Examples: "Analyze the impact of X on Y", "What are the implications of X?", "Synthesize information about X from multiple perspectives"

2. **Query Type:**
   - FACTUAL: Looking up specific facts or definitions
   - ANALYTICAL: Requires analysis or reasoning over information
   - SUMMARIZATION: Requires summarizing or condensing information
   - COMPARISON: Comparing multiple entities or concepts
   - MULTI_HOP: Requires connecting multiple pieces of information through reasoning

3. **Recommended Retrieval Mode:**
   - naive: Basic vector search (best for simple factual queries)
   - local: Context-dependent retrieval (good for questions about specific topics)
   - hybrid: Combines multiple approaches (good for moderate complexity)
   - global: Uses global knowledge structure (best for summarization and overview)
   - mix: Integrates graph and vector retrieval (best for complex multi-hop reasoning)

Provide your analysis in the following JSON format:
{{
    "complexity": "simple|moderate|complex",
    "query_type": "factual|analytical|summarization|comparison|multi_hop",
    "recommended_mode": "naive|local|hybrid|global|mix",
    "reasoning": "Brief explanation of why you chose this classification (1-2 sentences)",
    "confidence": 0.0-1.0
}}

Be concise and accurate in your analysis."""
}


class AdaptiveQueryRouter:
    """
    Adaptive query router that analyzes queries and recommends appropriate retrieval modes

    This implements the routing strategy from Adaptive-RAG, where queries are classified
    by complexity and routed to different retrieval strategies accordingly.
    """

    def __init__(self, llm_model_func: Callable):
        """
        Initialize the adaptive query router

        Args:
            llm_model_func: LLM function for query analysis
        """
        self.llm_model_func = llm_model_func

        # Default mode mapping based on complexity
        self.complexity_mode_map = {
            QueryComplexity.SIMPLE: "naive",      # Simple queries → naive/vector search
            QueryComplexity.MODERATE: "hybrid",   # Moderate queries → hybrid approach
            QueryComplexity.COMPLEX: "mix",       # Complex queries → graph + vector (mix)
        }

        # Alternative mappings for specific query types
        self.type_mode_override = {
            QueryType.FACTUAL: "local",           # Factual → local context
            QueryType.SUMMARIZATION: "global",    # Summarization → global knowledge
            QueryType.MULTI_HOP: "mix",          # Multi-hop → graph-based
        }

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and recommend retrieval mode

        Args:
            query: User query string

        Returns:
            Dict containing:
                - complexity: QueryComplexity enum value
                - query_type: QueryType enum value
                - recommended_mode: str (retrieval mode)
                - reasoning: str (explanation)
                - confidence: float (0-1)
        """
        try:
            # Prepare prompt for query analysis
            analysis_prompt = QUERY_ROUTER_PROMPTS["QUERY_ANALYSIS_PROMPT"].format(
                query=query
            )

            # Call LLM for analysis
            response = await self.llm_model_func(
                analysis_prompt,
                system_prompt=QUERY_ROUTER_PROMPTS["QUERY_ANALYSIS_SYSTEM"]
            )

            # Parse JSON response
            # Try to extract JSON from response (handle markdown code blocks)
            response_text = response.strip()
            if "```json" in response_text:
                # Extract JSON from markdown code block
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                # Extract from generic code block
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            analysis = json.loads(response_text)

            # Validate and normalize the response
            complexity_str = analysis.get("complexity", "moderate").lower()
            query_type_str = analysis.get("query_type", "factual").lower()

            # Map to enums
            complexity = QueryComplexity(complexity_str)
            query_type = QueryType(query_type_str)

            # Get recommended mode (use LLM's recommendation or apply our mapping)
            recommended_mode = analysis.get("recommended_mode",
                                           self._get_mode_for_query(complexity, query_type))

            return {
                "complexity": complexity,
                "query_type": query_type,
                "recommended_mode": recommended_mode,
                "reasoning": analysis.get("reasoning", "No reasoning provided"),
                "confidence": float(analysis.get("confidence", 0.8)),
                "raw_analysis": analysis
            }

        except Exception as e:
            # Fallback to default if analysis fails
            return {
                "complexity": QueryComplexity.MODERATE,
                "query_type": QueryType.FACTUAL,
                "recommended_mode": "hybrid",
                "reasoning": f"Failed to analyze query, using default mode. Error: {str(e)}",
                "confidence": 0.5,
                "error": str(e)
            }

    def _get_mode_for_query(self, complexity: QueryComplexity, query_type: QueryType) -> str:
        """
        Get retrieval mode based on complexity and query type

        Args:
            complexity: Query complexity level
            query_type: Query type category

        Returns:
            str: Recommended retrieval mode
        """
        # Check if query type has a specific override
        if query_type in self.type_mode_override:
            return self.type_mode_override[query_type]

        # Otherwise use complexity-based mapping
        return self.complexity_mode_map.get(complexity, "hybrid")

    def update_mode_mapping(self,
                           complexity_map: Optional[Dict[QueryComplexity, str]] = None,
                           type_override: Optional[Dict[QueryType, str]] = None):
        """
        Update the mode mapping configuration

        Args:
            complexity_map: New complexity to mode mapping
            type_override: New query type to mode override mapping
        """
        if complexity_map:
            self.complexity_mode_map.update(complexity_map)
        if type_override:
            self.type_mode_override.update(type_override)


async def analyze_and_route_query(
    query: str,
    llm_model_func: Callable,
    return_analysis: bool = False
) -> str | Dict[str, Any]:
    """
    Convenience function to analyze a query and get recommended mode

    Args:
        query: User query
        llm_model_func: LLM function for analysis
        return_analysis: If True, return full analysis; if False, return only mode

    Returns:
        str: Recommended mode (if return_analysis=False)
        Dict: Full analysis (if return_analysis=True)
    """
    router = AdaptiveQueryRouter(llm_model_func)
    analysis = await router.analyze_query(query)

    if return_analysis:
        return analysis
    else:
        return analysis["recommended_mode"]
