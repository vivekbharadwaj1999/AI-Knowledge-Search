"""
Extended analysis functions for VivBot RAG system.
Adds faithfulness, retrieval quality, and counterfactual analysis.
"""

from typing import List, Dict, Any, Optional
from app.config import EmbeddingClient, LLMClient
from app.vector_store import similarity_search
from app.qa import build_prompt, calculate_all_similarities
from app.faithfulness import (
    calculate_faithfulness_metrics,
    calculate_retrieval_quality_metrics,
    calculate_counterfactual_metrics
)
import random


def analyze_with_faithfulness(
    question: str,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    similarity_method: str = "cosine",
    embedding_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add faithfulness and retrieval quality metrics to existing analysis.
    
    Args:
        question: The question asked
        answer: The generated answer
        retrieved_chunks: Retrieved chunks with text, doc_name, score
        similarity_method: Similarity method used
        embedding_model: Embedding model used
    
    Returns:
        Dict with faithfulness and retrieval quality metrics
    """
    
    # Calculate faithfulness metrics
    faithfulness = calculate_faithfulness_metrics(
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        question=question
    )
    
    # Calculate retrieval quality metrics
    retrieval_quality = calculate_retrieval_quality_metrics(
        retrieved_chunks=retrieved_chunks,
        question=question
    )
    
    return {
        "faithfulness": faithfulness,
        "retrieval_quality": retrieval_quality,
        "similarity_method": similarity_method,
        "embedding_model": embedding_model
    }


def run_counterfactual_analysis(
    question: str,
    original_chunks: List[Dict[str, Any]],
    counterfactual_type: str,
    k: int = 7,
    doc_name: Optional[str] = None,
    model: Optional[str] = None,
    similarity: str = "cosine",
    embedding_model: Optional[str] = None,
    temperature: Optional[float] = None,
    username: Optional[str] = None,
    is_guest: bool = False,
    original_answer: Optional[str] = None  # NEW: Accept pre-generated answer
) -> Dict[str, Any]:
    """
    Run counterfactual retrieval analysis.
    
    Counterfactual types:
    - "remove_top": Remove the top-ranked chunk
    - "remove_top_3": Remove top 3 chunks
    - "random": Use random chunks instead
    - "lexical_only": Use only lexical (hybrid with high lexical weight)
    - "reverse_order": Reverse the ranking order
    
    Args:
        question: The question
        original_chunks: Original retrieved chunks
        counterfactual_type: Type of counterfactual experiment
        k: Number of chunks
        doc_name: Optional document filter
        model: LLM model
        similarity: Similarity method
        embedding_model: Embedding model
        temperature: Temperature for generation
        username: Username for access control
        is_guest: Whether user is guest
        original_answer: Optional pre-generated answer (e.g., from Critique final round or Compare specific model)
    
    Returns:
        Counterfactual analysis results
    """
    
    # Generate counterfactual chunks based on type
    if counterfactual_type == "remove_top":
        counterfactual_chunks = original_chunks[1:] if len(original_chunks) > 1 else []
    
    elif counterfactual_type == "remove_top_3":
        counterfactual_chunks = original_chunks[3:] if len(original_chunks) > 3 else []
    
    elif counterfactual_type == "reverse_order":
        counterfactual_chunks = list(reversed(original_chunks))
    
    elif counterfactual_type == "random":
        # This would require re-querying with random selection
        # For now, shuffle the existing chunks
        counterfactual_chunks = original_chunks.copy()
        random.shuffle(counterfactual_chunks)
    
    elif counterfactual_type == "lexical_only":
        # Re-rank using only keyword overlap
        # For simplification, we'll use the hybrid scores but weight them differently
        # In a full implementation, you'd re-query with different similarity weights
        counterfactual_chunks = original_chunks.copy()
        # This is a placeholder - you'd want to re-score with lexical emphasis
    
    else:
        raise ValueError(f"Unknown counterfactual type: {counterfactual_type}")
    
    # Generate answer with original chunks (or use provided answer)
    llm = LLMClient()
    
    if original_answer is None:
        # Generate original answer if not provided
        original_context = [f"[Source: {c.get('doc_name', 'Unknown')}] {c.get('text', '')}" for c in original_chunks]
        original_prompt = build_prompt(question, original_context)
        original_answer = llm.complete(original_prompt, model=model, temperature=temperature)
    
    # Generate answer with counterfactual chunks
    if counterfactual_chunks:
        cf_context = [f"[Source: {c.get('doc_name', 'Unknown')}] {c.get('text', '')}" for c in counterfactual_chunks]
        cf_prompt = build_prompt(question, cf_context)
        cf_answer = llm.complete(cf_prompt, model=model, temperature=temperature)
    else:
        cf_answer = "Unable to answer - no chunks available."
    
    # Calculate counterfactual metrics
    cf_metrics = calculate_counterfactual_metrics(
        original_answer=original_answer,
        counterfactual_answer=cf_answer,
        original_chunks=original_chunks,
        counterfactual_chunks=counterfactual_chunks,
        counterfactual_type=counterfactual_type
    )
    
    return {
        "counterfactual_type": counterfactual_type,
        "original_answer": original_answer,
        "counterfactual_answer": cf_answer,
        "original_chunks_count": len(original_chunks),
        "counterfactual_chunks_count": len(counterfactual_chunks),
        "metrics": cf_metrics,
        "original_chunks": [
            {
                "text": c.get("text", ""),
                "doc_name": c.get("doc_name", "Unknown"),
                "score": c.get("score", 0)
            }
            for c in original_chunks
        ],
        "chunks_used": [
            {
                "text": c.get("text", ""),
                "doc_name": c.get("doc_name", "Unknown"),
                "score": c.get("score", 0),
                "rank": c.get("rank", idx + 1)
            }
            for idx, c in enumerate(counterfactual_chunks)
        ]
    }


def add_extended_metrics_to_analysis(
    analysis_result: Dict[str, Any],
    include_faithfulness: bool = True,
    include_retrieval_quality: bool = True
) -> Dict[str, Any]:
    """
    Add extended metrics to existing analysis results.
    
    This function takes the output from analyze_ask_with_all_methods,
    analyze_compare_with_all_methods, or analyze_critique_with_all_methods
    and enriches it with faithfulness and retrieval quality metrics.
    
    Args:
        analysis_result: Existing analysis result dict
        include_faithfulness: Whether to add faithfulness metrics
        include_retrieval_quality: Whether to add retrieval quality metrics
    
    Returns:
        Enhanced analysis result with extended metrics
    """
    
    operation = analysis_result.get("operation", "ask")
    results_by_method = analysis_result.get("results_by_method", {})
    
    # Add extended metrics for each method
    enhanced_results = {}
    
    for method, method_data in results_by_method.items():
        enhanced_method = method_data.copy()
        
        # Get answer and chunks
        if operation == "ask":
            answer = method_data.get("answer", "")
            chunks = method_data.get("sources", [])
        elif operation == "critique":
            critique_result = method_data.get("critique_result", {})
            answer = critique_result.get("answer", "")
            chunks = method_data.get("sources", [])
        elif operation == "compare":
            # For compare, we have multiple models
            # We'll add metrics for each model
            enhanced_method["extended_metrics_by_model"] = {}
            for model, model_data in method_data.get("answers_by_model", {}).items():
                answer = model_data.get("answer", "")
                chunks = method_data.get("sources", [])
                
                extended = {}
                if include_faithfulness:
                    extended["faithfulness"] = calculate_faithfulness_metrics(
                        answer=answer,
                        retrieved_chunks=chunks,
                        question=analysis_result.get("input", {}).get("question", "")
                    )
                if include_retrieval_quality:
                    extended["retrieval_quality"] = calculate_retrieval_quality_metrics(
                        retrieved_chunks=chunks,
                        question=analysis_result.get("input", {}).get("question", "")
                    )
                
                enhanced_method["extended_metrics_by_model"][model] = extended
            
            enhanced_results[method] = enhanced_method
            continue
        else:
            enhanced_results[method] = enhanced_method
            continue
        
        # Add extended metrics
        extended = {}
        
        if include_faithfulness:
            extended["faithfulness"] = calculate_faithfulness_metrics(
                answer=answer,
                retrieved_chunks=chunks,
                question=analysis_result.get("input", {}).get("question", "")
            )
        
        if include_retrieval_quality:
            extended["retrieval_quality"] = calculate_retrieval_quality_metrics(
                retrieved_chunks=chunks,
                question=analysis_result.get("input", {}).get("question", "")
            )
        
        enhanced_method["extended_metrics"] = extended
        enhanced_results[method] = enhanced_method
    
    # Create enhanced result
    enhanced_analysis = analysis_result.copy()
    enhanced_analysis["results_by_method"] = enhanced_results
    enhanced_analysis["has_extended_metrics"] = True
    
    return enhanced_analysis
