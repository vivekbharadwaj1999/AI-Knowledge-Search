import os
from typing import Dict, Optional, List

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set. "
        "Create a .env file with GROQ_API_KEY=... or export it before starting the backend."
    )

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

GROQ_AVAILABLE_MODELS: Dict[str, str] = {
    "llama-3.1-8b-instant": "Llama 3.1 8B Instant — fast, lightweight",
    "llama-3.3-70b-versatile": "Llama 3.3 70B Versatile — high-quality general model",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout 17B 16E — efficient, balanced",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick 17B 128E — strong reasoning",
    "openai/gpt-oss-20b": "GPT OSS 20B — reliable all-round model",
    "openai/gpt-oss-120b": "GPT OSS 120B — high-capacity model",
    "meta-llama/llama-guard-4-12b": "Llama Guard 4 12B — safety model",
    "openai/gpt-oss-safeguard-20b": "GPT OSS Safeguard 20B — safety model",
    "moonshotai/kimi-k2-instruct-0905": "Kimi K2 Instruct 0905 — very large 256k context",
    "qwen/qwen3-32b": "Qwen3 32B — multilingual & strong general model",
}


class LLMClient:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.client = Groq(api_key=api_key or GROQ_API_KEY)

    def complete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        chosen_model = model or GROQ_MODEL
        if chosen_model not in GROQ_AVAILABLE_MODELS:
            chosen_model = GROQ_MODEL

        max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        temperature = (
            DEFAULT_TEMPERATURE if temperature is None else float(temperature)
        )

        resp = self.client.chat.completions.create(
            model=chosen_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return resp.choices[0].message.content


def get_model_label(model_id: str) -> str:
    return GROQ_AVAILABLE_MODELS.get(model_id, model_id)


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))


class EmbeddingClient:

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        return self.embed([text])[0]
