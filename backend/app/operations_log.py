import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


def get_user_operations_log_path(username: str, is_guest: bool = False) -> str:
    if is_guest:
        return f"data/guests/{username}/operations_log.jsonl"
    return f"data/users/{username}/operations_log.jsonl"


def log_ask_operation(
    question: str,
    answer: str,
    context: List[str],
    sources: List[Dict[str, Any]],
    top_k: int,
    doc_name: Optional[str],
    model: str,
    similarity: str,
    normalize_vectors: bool,
    embedding_model: Optional[str],
    temperature: Optional[float],
    username: Optional[str],
    is_guest: bool = False,
) -> None:
    try:
        log_entry = {
            "operation": "ask",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "parameters": {
                "question": question,
                "top_k": top_k,
                "doc_name": doc_name,
                "model": model,
                "similarity": similarity,
                "normalize_vectors": normalize_vectors,
                "embedding_model": embedding_model,
                "temperature": temperature,
            },
            "results": {
                "answer": answer,
                "context": context,
                "sources": sources,
                "answer_length": len(answer),
                "num_sources": len(sources),
            }
        }
        
        _write_log_entry(log_entry, username, is_guest)
    except Exception as e:
        print(f"Failed to log ask operation: {e}")


def log_compare_operation(
    question: str,
    model_left: str,
    model_right: str,
    answer_left: str,
    answer_right: str,
    context_left: List[str],
    context_right: List[str],
    sources_left: List[Dict[str, Any]],
    sources_right: List[Dict[str, Any]],
    top_k: int,
    doc_name: Optional[str],
    similarity: str,
    normalize_vectors: bool,
    embedding_model: Optional[str],
    temperature: Optional[float],
    username: Optional[str],
    is_guest: bool = False,
) -> None:
    try:
        log_entry = {
            "operation": "compare",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "parameters": {
                "question": question,
                "model_left": model_left,
                "model_right": model_right,
                "top_k": top_k,
                "doc_name": doc_name,
                "similarity": similarity,
                "normalize_vectors": normalize_vectors,
                "embedding_model": embedding_model,
                "temperature": temperature,
            },
            "results": {
                "left": {
                    "model": model_left,
                    "answer": answer_left,
                    "context": context_left,
                    "sources": sources_left,
                    "answer_length": len(answer_left),
                    "num_sources": len(sources_left),
                },
                "right": {
                    "model": model_right,
                    "answer": answer_right,
                    "context": context_right,
                    "sources": sources_right,
                    "answer_length": len(answer_right),
                    "num_sources": len(sources_right),
                }
            }
        }
        
        _write_log_entry(log_entry, username, is_guest)
    except Exception as e:
        print(f"Failed to log compare operation: {e}")


def log_advanced_analysis_operation(
    operation: str,
    parameters: Dict[str, Any],
    results: Dict[str, Any],
    username: Optional[str],
    is_guest: bool = False,
) -> None:
    try:
        log_entry = {
            "operation": f"advanced_{operation}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "parameters": parameters,
            "results": results,
        }
        
        _write_log_entry(log_entry, username, is_guest)
    except Exception as e:
        print(f"Failed to log advanced analysis operation: {e}")


def log_critique_operation(
    question: str,
    answer_model: str,
    critic_model: str,
    answer: str,
    context: List[str],
    sources: List[Dict[str, Any]],
    answer_critique_markdown: str,
    prompt_feedback_markdown: str,
    improved_prompt: str,
    prompt_issue_tags: List[str],
    scores: Optional[Dict[str, Any]],
    rounds: List[Dict[str, Any]],
    top_k: int,
    doc_name: Optional[str],
    self_correct: bool,
    similarity: str,
    normalize_vectors: bool,
    embedding_model: Optional[str],
    temperature: Optional[float],
    username: Optional[str],
    is_guest: bool = False,
) -> None:
    try:
        log_entry = {
            "operation": "critique",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "parameters": {
                "question": question,
                "answer_model": answer_model,
                "critic_model": critic_model,
                "top_k": top_k,
                "doc_name": doc_name,
                "self_correct": self_correct,
                "similarity": similarity,
                "normalize_vectors": normalize_vectors,
                "embedding_model": embedding_model,
                "temperature": temperature,
            },
            "results": {
                "answer": answer,
                "context": context,
                "sources": sources,
                "answer_critique_markdown": answer_critique_markdown,
                "prompt_feedback_markdown": prompt_feedback_markdown,
                "improved_prompt": improved_prompt,
                "prompt_issue_tags": prompt_issue_tags,
                "scores": scores,
                "rounds": rounds,
                "num_rounds": len(rounds),
            }
        }
        
        _write_log_entry(log_entry, username, is_guest)
    except Exception as e:
        print(f"Failed to log critique operation: {e}")


def _write_log_entry(
    log_entry: Dict[str, Any],
    username: Optional[str],
    is_guest: bool = False,
) -> None:
    if not username:
        return
    
    log_path = get_user_operations_log_path(username, is_guest)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def get_operations_log(
    username: str,
    is_guest: bool = False,
) -> List[Dict[str, Any]]:
    log_path = Path(get_user_operations_log_path(username, is_guest))
    
    if not log_path.exists():
        return []
    
    entries = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return entries


def reset_operations_log(
    username: str,
    is_guest: bool = False,
) -> None:
    log_path = Path(get_user_operations_log_path(username, is_guest))
    
    if log_path.exists():
        log_path.unlink()


def check_operations_log_exists(
    username: str,
    is_guest: bool = False,
) -> bool:
    log_path = Path(get_user_operations_log_path(username, is_guest))
    
    if not log_path.exists():
        return False
    
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        return True
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass
    
    return False
