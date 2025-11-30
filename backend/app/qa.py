from typing import List, Optional

from app.config import EmbeddingClient, LLMClient
from app.vector_store import similarity_search


def build_prompt(question: str, context_chunks: List[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context_text}

Question: {question}

- First, try to infer the best answer you can from the context, even if it is not stated in a single sentence.
- If you truly cannot infer an answer at all, then say you don't know.
- Context chunks may indicate their source as "[Source: DOC_NAME]". If different sources disagree, briefly point this out and explain.
- Be clear and concise.
"""


def answer_question(
    question: str,
    k: int = 7,
    doc_name: Optional[str] = None,
    model: Optional[str] = None,
):
    embed_client = EmbeddingClient()
    query_embedding = embed_client.embed_query(question)

    # Now returns full records with doc_name, text, score, etc.
    records = similarity_search(query_embedding, k=k, doc_name=doc_name)

    # Build two parallel things:
    # 1) context for the LLM (with [Source: ...] prefix)
    # 2) plain chunks + metadata for the frontend
    context_for_llm: List[str] = []
    sources: List[dict] = []

    for rec in records:
        text = rec.get("text") or ""
        if not text:
            continue

        doc = rec.get("doc_name") or "Unknown document"
        score = float(rec.get("score", 0.0))

        labeled = f"[Source: {doc}] {text}"
        context_for_llm.append(labeled)

        sources.append(
            {
                "doc_name": doc,
                "text": text,
                "score": score,
            }
        )

    llm = LLMClient()
    prompt = build_prompt(question, context_for_llm)
    answer = llm.complete(prompt, model=model)

    # Plain chunks for compatibility with existing UI / insights
    plain_chunks = [s["text"] for s in sources]

    return answer, plain_chunks, sources
