import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import math

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass 
except ImportError:
    NLTK_AVAILABLE = False


def calculate_faithfulness_metrics(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    question: str = ""
) -> Dict[str, Any]:
    answer_sentences = _split_into_sentences(answer)
    chunk_texts = [c.get("text", "") for c in retrieved_chunks]

    model = None
    chunk_embeddings = None
    if SENTENCE_TRANSFORMER_AVAILABLE:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            chunk_embeddings = model.encode(chunk_texts, show_progress_bar=False)
        except Exception:
            model = None
            chunk_embeddings = None

    sentence_support = []
    total_supported = 0
    extracted_quotes = []
    
    for idx, sentence in enumerate(answer_sentences):
        support_info = _analyze_sentence_support(
            sentence, chunk_texts, retrieved_chunks, model, chunk_embeddings
        )
        sentence_support.append({
            "sentence": sentence,
            "sentence_id": idx,
            "supported": support_info["supported"],
            "supporting_chunks": support_info["supporting_chunks"],
            "confidence": support_info["confidence"],
            "confidence_lexical": support_info["confidence_lexical"],
            "confidence_semantic": support_info["confidence_semantic"],
            "quotes": support_info["quotes"]
        })
        
        if support_info["supported"]:
            total_supported += 1

        for quote in support_info["quotes"]:
            if quote not in extracted_quotes:
                extracted_quotes.append(quote)

    evidence_coverage = total_supported / len(answer_sentences) if answer_sentences else 0
    hallucination_risk = 1.0 - evidence_coverage
    citation_coverage = (total_supported / len(answer_sentences) * 100) if answer_sentences else 0
    extracted_quotes = extracted_quotes[:3]
    
    return {
        "sentence_support": sentence_support,
        "extracted_quotes": extracted_quotes,
        "hallucination_risk": round(hallucination_risk, 3),
        "evidence_coverage": round(evidence_coverage, 3),
        "citation_coverage": round(citation_coverage, 1),
        "total_sentences": len(answer_sentences),
        "supported_sentences": total_supported
    }


def _clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)

    text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'\1', text) 
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)    
    text = re.sub(r'\*([^*]+)\*', r'\1', text)    
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\|[-:\s]+\|', ' ', text)
    text = re.sub(r'^\|', '', text, flags=re.MULTILINE)
    text = re.sub(r'\|$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def _split_into_sentences(text: str) -> List[str]:
    cleaned_text = _clean_text(text)
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(cleaned_text)
            filtered = []
            for s in sentences:
                s = s.strip()
                if len(s) < 3:
                    continue
                if re.match(r'^\d+\.?$', s):
                    continue
                if re.match(r'^[^\w\s]+$', s):
                    continue
                filtered.append(s)
            
            return filtered
        except Exception:
            pass

    sentences = re.split(r'(?<!\d)(?<=[.!?])\s+', cleaned_text)

    filtered = []
    for s in sentences:
        s = s.strip()
        if not s or len(s) < 3:
            continue
        if re.match(r'^\d+\.?$', s):
            continue
        if re.match(r'^[^\w\s]+$', s):
            continue
        filtered.append(s)
    
    return filtered


def _analyze_sentence_support(
    sentence: str,
    chunk_texts: List[str],
    retrieved_chunks: List[Dict[str, Any]],
    model = None,
    chunk_embeddings = None
) -> Dict[str, Any]:
    sentence_lower = sentence.lower()
    sentence_tokens = set(_tokenize(sentence_lower))
    
    supporting_chunks = []
    quotes = []
    max_lexical_overlap = 0.0
    max_semantic_similarity = 0.0
    sentence_embedding = None
    if model is not None and chunk_embeddings is not None:
        try:
            sentence_embedding = model.encode(sentence, show_progress_bar=False)
        except Exception:
            sentence_embedding = None
    
    for idx, chunk_text in enumerate(chunk_texts):
        chunk_lower = chunk_text.lower()
        chunk_tokens = set(_tokenize(chunk_lower))

        if not sentence_tokens or not chunk_tokens:
            lexical_overlap = 0.0
        else:
            lexical_overlap = len(sentence_tokens & chunk_tokens) / len(sentence_tokens)
        semantic_similarity = 0.0
        if sentence_embedding is not None and chunk_embeddings is not None:
            try:
                semantic_similarity = float(cos_sim(sentence_embedding, chunk_embeddings[idx])[0][0])
                semantic_similarity = max(0.0, min(1.0, semantic_similarity))
            except Exception:
                semantic_similarity = 0.0
        potential_quotes = _extract_matching_phrases(sentence, chunk_text)

        if lexical_overlap > 0.3 or semantic_similarity > 0.5 or potential_quotes:
            supporting_chunks.append({
                "chunk_id": idx,
                "doc_name": retrieved_chunks[idx].get("doc_name", "Unknown"),
                "lexical_overlap": round(lexical_overlap, 3),
                "semantic_similarity": round(semantic_similarity, 3),
                "overlap": round(lexical_overlap, 3), 
                "rank": retrieved_chunks[idx].get("rank", idx + 1)
            })
            quotes.extend(potential_quotes)
            max_lexical_overlap = max(max_lexical_overlap, lexical_overlap)
            max_semantic_similarity = max(max_semantic_similarity, semantic_similarity)
    
    supported = max_lexical_overlap > 0.3 or max_semantic_similarity > 0.5 or len(quotes) > 0
    confidence = max(max_lexical_overlap, max_semantic_similarity)
    
    return {
        "supported": supported,
        "supporting_chunks": supporting_chunks,
        "confidence": round(confidence, 3),
        "confidence_lexical": round(max_lexical_overlap, 3),
        "confidence_semantic": round(max_semantic_similarity, 3),
        "quotes": quotes[:2]  
    }


def _tokenize(text: str) -> List[str]:
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


def _extract_matching_phrases(sentence: str, chunk: str, min_length: int = 5) -> List[str]:
    sentence_lower = sentence.lower()
    chunk_lower = chunk.lower()
    
    quotes = []
    words = sentence_lower.split()
    for n in range(min_length, min(len(words) + 1, 15)):  
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i+n])
            if phrase in chunk_lower and len(phrase.split()) >= min_length:
                if phrase not in [q.lower() for q in quotes]:
                    quotes.append(' '.join(sentence.split()[i:i+n]))
    
    return quotes


def calculate_retrieval_quality_metrics(
    retrieved_chunks: List[Dict[str, Any]],
    question: str = ""
) -> Dict[str, Any]:
    if not retrieved_chunks:
        return {
            "chunk_redundancy": 0.0,
            "diversity_score": 1.0,
            "document_coverage": 0,
            "unique_documents": [],
            "lexical_semantic_balance": 0.5,
            "avg_chunk_similarity": 0.0,
            "redundancy_details": []
        }
    
    chunk_texts = [c.get("text", "") for c in retrieved_chunks]
    pairwise_similarities = []
    redundancy_details = []
    
    for i in range(len(chunk_texts)):
        for j in range(i + 1, len(chunk_texts)):
            similarity = _calculate_text_similarity(chunk_texts[i], chunk_texts[j])
            pairwise_similarities.append(similarity)
            
            if similarity > 0.6: 
                redundancy_details.append({
                    "chunk_1": i,
                    "chunk_2": j,
                    "similarity": round(similarity, 3),
                    "doc_1": retrieved_chunks[i].get("doc_name", "Unknown"),
                    "doc_2": retrieved_chunks[j].get("doc_name", "Unknown")
                })
    
    avg_similarity = sum(pairwise_similarities) / len(pairwise_similarities) if pairwise_similarities else 0
    chunk_redundancy = avg_similarity
    diversity_score = 1.0 - chunk_redundancy
    unique_docs = list(set(c.get("doc_name", "Unknown") for c in retrieved_chunks))
    document_coverage = len(unique_docs)
    lexical_semantic_balance = _calculate_lexical_semantic_balance(retrieved_chunks)
    
    return {
        "chunk_redundancy": round(chunk_redundancy, 3),
        "diversity_score": round(diversity_score, 3),
        "document_coverage": document_coverage,
        "unique_documents": unique_docs,
        "lexical_semantic_balance": round(lexical_semantic_balance, 3),
        "avg_chunk_similarity": round(avg_similarity, 3),
        "redundancy_details": redundancy_details[:5] 
    }


def _calculate_text_similarity(text1: str, text2: str) -> float:
    tokens1 = set(_tokenize(text1.lower()))
    tokens2 = set(_tokenize(text2.lower()))
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0


def _calculate_semantic_similarity(text1: str, text2: str) -> float:
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        return 0.0
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([text1, text2])
        similarity = float(cos_sim(embeddings[0], embeddings[1])[0][0])
        return max(0.0, min(1.0, similarity))  
    except Exception:
        return 0.0


def _calculate_rouge_l(text1: str, text2: str) -> float:
    if not ROUGE_AVAILABLE:
        return 0.0
    
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(text1, text2)
        return scores['rougeL'].fmeasure
    except Exception:
        return 0.0


def _calculate_lexical_semantic_balance(retrieved_chunks: List[Dict[str, Any]]) -> float:
    if not retrieved_chunks or not retrieved_chunks[0].get("all_scores"):
        return 0.5  
    
    cosine_scores = []
    hybrid_scores = []
    
    for chunk in retrieved_chunks:
        all_scores = chunk.get("all_scores", {})
        if "cosine" in all_scores:
            cosine_scores.append(all_scores["cosine"])
        if "hybrid" in all_scores:
            hybrid_scores.append(all_scores["hybrid"])
    
    if not cosine_scores or not hybrid_scores:
        return 0.5
    avg_cosine = sum(cosine_scores) / len(cosine_scores)
    avg_hybrid = sum(hybrid_scores) / len(hybrid_scores)
    difference = avg_cosine - avg_hybrid
    balance = 0.5 + (difference / 2.0)
    return max(0.0, min(1.0, balance))


def calculate_counterfactual_metrics(
    original_answer: str,
    counterfactual_answer: str,
    original_chunks: List[Dict[str, Any]],
    counterfactual_chunks: List[Dict[str, Any]],
    counterfactual_type: str
) -> Dict[str, Any]:
    
    jaccard_similarity = _calculate_text_similarity(original_answer, counterfactual_answer)
    semantic_similarity = _calculate_semantic_similarity(original_answer, counterfactual_answer)
    rouge_l_score = _calculate_rouge_l(original_answer, counterfactual_answer)
    answer_similarity = semantic_similarity if semantic_similarity > 0 else jaccard_similarity
    original_texts = set(c.get("text", "") for c in original_chunks)
    counterfactual_texts = set(c.get("text", "") for c in counterfactual_chunks)
    chunk_overlap = len(original_texts & counterfactual_texts) / len(original_texts) if original_texts else 0
    retrieval_dependence = 1.0 - answer_similarity if chunk_overlap < 0.5 else answer_similarity
    
    return {
        "counterfactual_type": counterfactual_type,
        "answer_similarity": round(answer_similarity, 3),
        "answer_similarity_semantic": round(semantic_similarity, 3),
        "answer_similarity_rouge_l": round(rouge_l_score, 3),
        "answer_similarity_jaccard": round(jaccard_similarity, 3),
        "chunk_overlap": round(chunk_overlap, 3),
        "retrieval_dependence": round(retrieval_dependence, 3),
        "original_answer_length": len(original_answer),
        "counterfactual_answer_length": len(counterfactual_answer),
        "answer_collapsed": len(counterfactual_answer) < len(original_answer) * 0.5
    }
