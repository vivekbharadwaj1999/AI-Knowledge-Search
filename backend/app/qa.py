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
- Be clear and concise.
"""


def answer_question(
    question: str,
    k: int = 7,
    doc_name: Optional[str] = None,
):
    embed_client = EmbeddingClient()
    query_embedding = embed_client.embed_query(question)
    top_chunks = similarity_search(query_embedding, k=k, doc_name=doc_name)

    llm = LLMClient()
    prompt = build_prompt(question, top_chunks)
    answer = llm.complete(prompt)

    return answer, top_chunks
