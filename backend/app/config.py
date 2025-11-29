import os
import hashlib
from typing import List
from dotenv import load_dotenv

from groq import Groq

load_dotenv()

# Choose a Groq model, e.g. Llama 3.1 or Mixtral
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment or .env file")
    return Groq(api_key=api_key)


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
        # take first 32 bytes -> floats in [0,1]
        return [b / 255.0 for b in h[:32]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._fake_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._fake_embed(text)


class LLMClient:
    """
    Uses Groq chat.completions API to generate answers.
    """

    def __init__(self):
        self.client = get_groq_client()

    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
