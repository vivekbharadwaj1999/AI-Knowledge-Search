"""
Batch Evaluation Harness for VivBot RAG System

This module allows running controlled experiments across:
- Multiple questions
- Different similarity methods
- Different embedding models
- Different Top-K values
- Different LLM models
"""

import json
import csv
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from app.config import EmbeddingClient, LLMClient, AVAILABLE_EMBEDDING_MODELS
from app.qa import answer_question, calculate_all_similarities
from app.vector_store import similarity_search
from app.faithfulness import calculate_faithfulness_metrics


class BatchEvaluator:
    """Runs batch experiments and exports results."""
    
    def __init__(self, username: str = "default", is_guest: bool = False):
        self.username = username
        self.is_guest = is_guest
        self.results = []
    
    def run_batch_experiment(
        self,
        questions: List[str],
        operations: List[Dict[str, Any]],
        similarity_methods: Optional[List[str]] = None,
        embedding_models: Optional[List[str]] = None,
        top_k_values: Optional[List[int]] = None,
        doc_name: Optional[str] = None,
        normalize_vectors: bool = True,
        temperature: Optional[float] = None,
        include_faithfulness: bool = True
    ) -> Dict[str, Any]:
        """
        Run a batch experiment with specified configurations.
        
        Args:
            questions: List of questions to evaluate
            operations: List of operation configs (ask/compare/critique)
            similarity_methods: List of similarity methods to test (default: all 5)
            embedding_models: List of embedding models to test (default: current)
            top_k_values: List of Top-K values to test (default: [5, 7, 10])
            doc_name: Optional document to filter by
            normalize_vectors: Whether to normalize vectors
            temperature: Temperature for LLM
            include_faithfulness: Whether to calculate faithfulness metrics
        
        Returns:
            Dict with experiment results and metadata
        """
        
        # Set defaults
        if similarity_methods is None:
            similarity_methods = ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]
        
        if top_k_values is None:
            top_k_values = [5, 7, 10]
        
        if embedding_models is None:
            embedding_models = [None]  # Use default
        
        # Track experiment metadata
        experiment_start = datetime.now()
        total_runs = len(questions) * len(similarity_methods) * len(embedding_models) * len(top_k_values) * len(operations)
        
        results = []
        run_count = 0
        
        # Run experiments
        for question_idx, question in enumerate(questions):
            for embedding_model in embedding_models:
                for similarity_method in similarity_methods:
                    for top_k in top_k_values:
                        for operation in operations:
                            run_count += 1
                            
                            # Run single evaluation based on operation type
                            try:
                                result = self._run_single_operation(
                                    question=question,
                                    question_idx=question_idx,
                                    operation=operation,
                                    similarity_method=similarity_method,
                                    embedding_model=embedding_model,
                                    top_k=top_k,
                                    doc_name=doc_name,
                                    normalize_vectors=normalize_vectors,
                                    temperature=temperature,
                                    include_faithfulness=include_faithfulness
                                )
                                
                                result["run_number"] = run_count
                                result["total_runs"] = total_runs
                                results.append(result)
                                
                            except Exception as e:
                                # Log error but continue
                                results.append({
                                    "run_number": run_count,
                                    "total_runs": total_runs,
                                    "question_idx": question_idx,
                                    "question": question,
                                    "error": str(e),
                                    "status": "failed"
                                })
        
        experiment_end = datetime.now()
        duration = (experiment_end - experiment_start).total_seconds()
        
        # Collect actual embedding models used from results
        actual_models_used = set()
        for result in results:
            if result.get("status") == "success":
                model = result.get("configuration", {}).get("embedding_model")
                if model and model != "default":
                    actual_models_used.add(model)
        
        # Calculate summary statistics
        summary = self._calculate_summary_statistics(results)
        
        return {
            "experiment_metadata": {
                "start_time": experiment_start.isoformat(),
                "end_time": experiment_end.isoformat(),
                "duration_seconds": duration,
                "total_runs": total_runs,
                "successful_runs": sum(1 for r in results if r.get("status") != "failed"),
                "failed_runs": sum(1 for r in results if r.get("status") == "failed"),
                "questions_count": len(questions),
                "configurations": {
                    "similarity_methods": similarity_methods,
                    "embedding_models": list(actual_models_used) if actual_models_used else ["auto-detected"],
                    "top_k_values": top_k_values,
                    "operations": [op.get("type") for op in operations]
                }
            },
            "results": results,
            "summary": summary
        }
    
    def _run_single_operation(
        self,
        question: str,
        question_idx: int,
        operation: Dict[str, Any],
        similarity_method: str,
        embedding_model: Optional[str],
        top_k: int,
        doc_name: Optional[str],
        normalize_vectors: bool,
        temperature: Optional[float],
        include_faithfulness: bool
    ) -> Dict[str, Any]:
        """Run a single evaluation for any operation type (ask/compare/critique)."""
        
        from app.qa import answer_question
        from app.critique import run_critique
        from app.vector_store import get_document_embedding_model, get_documents_info
        
        start_time = time.time()
        operation_type = operation.get("type", "ask")
        
        # Detect actual embedding model if not specified
        actual_embedding_model = embedding_model
        if actual_embedding_model is None:
            if doc_name:
                # Get model from specific document
                actual_embedding_model = get_document_embedding_model(
                    doc_name, username=self.username, is_guest=self.is_guest
                )
            else:
                # Get most common model across all documents
                docs_info = get_documents_info(username=self.username, is_guest=self.is_guest)
                if docs_info:
                    from collections import Counter
                    models = [d.get('embedding_model') for d in docs_info if d.get('embedding_model')]
                    if models:
                        actual_embedding_model = Counter(models).most_common(1)[0][0]
        
        result = {
            "status": "success",
            "question_idx": question_idx,
            "question": question,
            "configuration": {
                "operation": operation_type,
                "similarity_method": similarity_method,
                "embedding_model": actual_embedding_model or "default",
                "top_k": top_k,
                "normalize_vectors": normalize_vectors,
                "temperature": temperature
            }
        }
        
        if operation_type == "ask":
            # Simple ask operation
            model = operation.get("model")
            answer, plain_chunks, sources = answer_question(
                question=question,
                k=top_k,
                doc_name=doc_name,
                model=model,
                similarity=similarity_method,
                normalize_vectors=normalize_vectors,
                embedding_model=embedding_model,
                temperature=temperature,
                username=self.username,
                is_guest=self.is_guest
            )
            
            latency = time.time() - start_time
            
            result["configuration"]["model"] = model or "default"
            result["answer"] = answer
            result["sources"] = sources
            result["metrics"] = self._calculate_metrics(answer, sources, question, include_faithfulness)
            result["metrics"]["latency_seconds"] = latency
            
        elif operation_type == "compare":
            # Compare operation
            models = operation.get("models", [])
            if len(models) < 2:
                models = [None, None]  # Default models
            
            # Run with first model
            answer1, _, sources1 = answer_question(
                question=question,
                k=top_k,
                doc_name=doc_name,
                model=models[0],
                similarity=similarity_method,
                normalize_vectors=normalize_vectors,
                embedding_model=embedding_model,
                temperature=temperature,
                username=self.username,
                is_guest=self.is_guest
            )
            
            # Run with second model
            answer2, _, sources2 = answer_question(
                question=question,
                k=top_k,
                doc_name=doc_name,
                model=models[1],
                similarity=similarity_method,
                normalize_vectors=normalize_vectors,
                embedding_model=embedding_model,
                temperature=temperature,
                username=self.username,
                is_guest=self.is_guest
            )
            
            latency = time.time() - start_time
            
            result["configuration"]["models"] = models
            result["answers"] = {
                "model_1": answer1,
                "model_2": answer2
            }
            result["sources"] = {
                "model_1": sources1,
                "model_2": sources2
            }
            
            # Calculate metrics for both models
            metrics1 = self._calculate_metrics(answer1, sources1, question, include_faithfulness)
            metrics1["latency_seconds"] = latency
            
            metrics2 = self._calculate_metrics(answer2, sources2, question, include_faithfulness)
            metrics2["latency_seconds"] = latency
            
            result["metrics"] = {
                "model_1": metrics1,
                "model_2": metrics2
            }
            
        elif operation_type == "critique":
            # Critique operation
            answer_model = operation.get("answer_model")
            critic_model = operation.get("critic_model")
            self_correct = operation.get("self_correct", True)
            max_rounds = 2 if self_correct else 1
            
            critique_result = run_critique(
                question=question,
                answer_model=answer_model,
                critic_model=critic_model,
                top_k=top_k,
                doc_name=doc_name,
                self_correct=self_correct,
                similarity=similarity_method,
                embedding_model=embedding_model,
                temperature=temperature,
                username=self.username,
                is_guest=self.is_guest
            )
            
            latency = time.time() - start_time
            
            result["configuration"]["answer_model"] = answer_model or "default"
            result["configuration"]["critic_model"] = critic_model or "default"
            result["configuration"]["self_correct"] = self_correct
            result["answer"] = critique_result.get("answer", "")
            result["critique_rounds"] = len(critique_result.get("rounds", []))
            result["sources"] = critique_result.get("sources", [])
            
            # Get metrics from final answer
            final_answer = critique_result.get("answer", "")
            sources = critique_result.get("sources", [])
            result["metrics"] = self._calculate_metrics(final_answer, sources, question, include_faithfulness)
            result["metrics"]["latency_seconds"] = latency
        
        return result
    
    def _calculate_metrics(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        question: str,
        include_faithfulness: bool
    ) -> Dict[str, Any]:
        """Calculate metrics for an answer."""
        metrics = {
            "answer_length": len(answer),
            "answer_word_count": len(answer.split()),
            "chunks_retrieved": len(sources)
        }
        
        if include_faithfulness:
            metrics["faithfulness"] = calculate_faithfulness_metrics(
                answer=answer,
                retrieved_chunks=sources,
                question=question
            )
        
        return metrics
    
    
    def _calculate_summary_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics across all runs."""
        
        successful_results = [r for r in results if r.get("status") == "success"]
        
        if not successful_results:
            return {"error": "No successful runs to summarize"}
        
        # Helper function to get metrics (handles both simple and compare operations)
        def get_metrics(result):
            metrics = result.get("metrics", {})
            # For compare operations, take model_1 metrics as representative
            if "model_1" in metrics:
                return metrics["model_1"]
            return metrics
        
        # Calculate averages
        all_metrics = [get_metrics(r) for r in successful_results]
        
        avg_latency = sum(m.get("latency_seconds", 0) for m in all_metrics) / len(all_metrics)
        avg_answer_length = sum(m.get("answer_length", 0) for m in all_metrics) / len(all_metrics)
        avg_chunks = sum(m.get("chunks_retrieved", 0) for m in all_metrics) / len(all_metrics)
        
        # Calculate faithfulness averages if available
        faithfulness_metrics = [m.get("faithfulness") for m in all_metrics if "faithfulness" in m]
        avg_faithfulness = None
        
        if faithfulness_metrics:
            avg_faithfulness = {
                "avg_hallucination_risk": sum(f["hallucination_risk"] for f in faithfulness_metrics) / len(faithfulness_metrics),
                "avg_evidence_coverage": sum(f["evidence_coverage"] for f in faithfulness_metrics) / len(faithfulness_metrics),
                "avg_citation_coverage": sum(f.get("citation_coverage", 0) for f in faithfulness_metrics) / len(faithfulness_metrics)
            }
        
        # Group by configuration
        by_similarity = {}
        by_top_k = {}
        
        for result in successful_results:
            sim = result["configuration"]["similarity_method"]
            if sim not in by_similarity:
                by_similarity[sim] = []
            by_similarity[sim].append(result)
            
            k = result["configuration"]["top_k"]
            if k not in by_top_k:
                by_top_k[k] = []
            by_top_k[k].append(result)
        
        # Calculate per-group averages
        similarity_stats = {}
        for sim, group_results in by_similarity.items():
            group_metrics = [get_metrics(r) for r in group_results]
            similarity_stats[sim] = {
                "count": len(group_results),
                "avg_latency": sum(m.get("latency_seconds", 0) for m in group_metrics) / len(group_metrics),
                "avg_answer_length": sum(m.get("answer_length", 0) for m in group_metrics) / len(group_metrics)
            }
        
        top_k_stats = {}
        for k, group_results in by_top_k.items():
            group_metrics = [get_metrics(r) for r in group_results]
            top_k_stats[k] = {
                "count": len(group_results),
                "avg_latency": sum(m.get("latency_seconds", 0) for m in group_metrics) / len(group_metrics),
                "avg_answer_length": sum(m.get("answer_length", 0) for m in group_metrics) / len(group_metrics)
            }
        
        return {
            "overall": {
                "avg_latency_seconds": round(avg_latency, 3),
                "avg_answer_length": round(avg_answer_length, 1),
                "avg_chunks_retrieved": round(avg_chunks, 1)
            },
            "faithfulness": avg_faithfulness,
            "by_similarity_method": similarity_stats,
            "by_top_k": top_k_stats
        }
    
    def export_to_csv(self, results_data: Dict[str, Any], output_path: str) -> str:
        """
        Export batch results to CSV format.
        
        Args:
            results_data: Results from run_batch_experiment
            output_path: Path to save CSV file
        
        Returns:
            Path to saved CSV file
        """
        
        results = results_data.get("results", [])
        
        if not results:
            raise ValueError("No results to export")
        
        # Define CSV columns
        fieldnames = [
            "run_number",
            "question_idx",
            "question",
            "operation",
            "similarity_method",
            "embedding_model",
            "top_k",
            "model",  # For ask, or model_1 for compare, or answer_model for critique
            "temperature",
            "answer_length",
            "answer_word_count",
            "chunks_retrieved",
            "latency_seconds",
            "hallucination_risk",
            "evidence_coverage",
            "answer"
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                if result.get("status") != "success":
                    continue
                
                config = result["configuration"]
                operation = config["operation"]
                
                # Get metrics (handle compare which has nested structure)
                metrics = result.get("metrics", {})
                if "model_1" in metrics:
                    # Compare operation - use model_1 metrics
                    metrics = metrics["model_1"]
                
                # Build base row
                row = {
                    "run_number": result.get("run_number"),
                    "question_idx": result.get("question_idx"),
                    "question": result.get("question"),
                    "operation": operation,
                    "similarity_method": config["similarity_method"],
                    "embedding_model": config["embedding_model"],
                    "top_k": config["top_k"],
                    "temperature": config.get("temperature"),
                    "answer_length": metrics.get("answer_length", 0),
                    "answer_word_count": metrics.get("answer_word_count", 0),
                    "chunks_retrieved": metrics.get("chunks_retrieved", 0),
                    "latency_seconds": metrics.get("latency_seconds", 0),
                    "answer": result.get("answer", "")
                }
                
                # Add model info based on operation type
                if operation == "ask":
                    row["model"] = config.get("model", "default")
                elif operation == "compare":
                    row["model"] = f"{config['models'][0]} vs {config['models'][1]}"
                elif operation == "critique":
                    row["model"] = f"{config.get('answer_model', 'default')} (critic: {config.get('critic_model', 'default')})"
                
                # Add faithfulness metrics if available
                if "faithfulness" in metrics:
                    row["hallucination_risk"] = metrics["faithfulness"].get("hallucination_risk")
                    row["evidence_coverage"] = metrics["faithfulness"].get("evidence_coverage")
                
                writer.writerow(row)
        
        return output_path
    
    def export_to_json(self, results_data: Dict[str, Any], output_path: str) -> str:
        """
        Export batch results to JSON format.
        
        Args:
            results_data: Results from run_batch_experiment
            output_path: Path to save JSON file
        
        Returns:
            Path to saved JSON file
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        return output_path


def create_sample_question_set(topic: str = "general", count: int = 20) -> List[str]:
    """
    Create a sample set of questions for batch evaluation.
    
    Args:
        topic: Topic category for questions
        count: Number of questions to generate
    
    Returns:
        List of questions
    """
    
    # This is a placeholder - in production, you'd want more sophisticated question generation
    sample_questions = [
        "What are the main findings discussed in the document?",
        "Can you summarize the key points?",
        "What methodology was used in this study?",
        "What are the limitations mentioned?",
        "What are the recommendations or conclusions?",
        "How does this relate to previous research?",
        "What are the implications of these findings?",
        "What future work is suggested?",
        "What data sources were used?",
        "What are the key definitions or concepts?",
        "What evidence supports the main argument?",
        "Are there any contradictions or disagreements?",
        "What quantitative results are reported?",
        "What qualitative insights are provided?",
        "What are the practical applications?",
        "What are the theoretical contributions?",
        "How was the data analyzed?",
        "What are the ethical considerations?",
        "What are the strengths of this approach?",
        "What areas need further investigation?"
    ]
    
    return sample_questions[:count]
