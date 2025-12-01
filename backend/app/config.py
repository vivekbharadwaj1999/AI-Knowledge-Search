import os
import hashlib
from typing import List, Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment or .env file")
    return Groq(api_key=api_key)


DEPRECATED_MODEL_FALLBACKS = {
    "llama-3.1-70b-versatile": "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768": "gemma2-27b-it",
}


class EmbeddingClient:
    """
    Simple local 'embedding' based on SHA-256.
    Not a real semantic embedding, but good enough
    to show the RAG pipeline structure without any API cost.
    """

    def __init__(self):
        pass

    def _fake_embed(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in h[:32]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._fake_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._fake_embed(text)


class LLMClient:

    def __init__(self):
        self.client = get_groq_client()

    def complete(self, prompt: str, model: Optional[str] = None) -> str:
        chosen_model = model or GROQ_MODEL

        if chosen_model in DEPRECATED_MODEL_FALLBACKS:
            print(f"[LLMClient] Model '{chosen_model}' deprecated, "
                  f"using '{DEPRECATED_MODEL_FALLBACKS[chosen_model]}' instead.")
            chosen_model = DEPRECATED_MODEL_FALLBACKS[chosen_model]

        resp = self.client.chat.completions.create(
            model=chosen_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
