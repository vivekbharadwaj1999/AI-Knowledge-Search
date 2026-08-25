import os
from typing import Dict, Optional, List

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.models_catalog import (
    OPENROUTER_BASE_URL,
    get_catalog,
    get_model_label,
    is_allowed,
)

load_dotenv()

# Offline mode: returns canned completions so the whole app can be exercised
# without spending credits. Set USE_FAKE_LLM=true in .env.
USE_FAKE_LLM = os.getenv("USE_FAKE_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY and not USE_FAKE_LLM:
    raise RuntimeError(
        "OPENROUTER_API_KEY environment variable is not set. "
        "Create a .env file with OPENROUTER_API_KEY=... (get one at "
        "https://openrouter.ai/keys), or set USE_FAKE_LLM=true to run offline."
    )

# Used only for OpenAI-hosted *embedding* models; chat goes through OpenRouter.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# OpenRouter attributes traffic using these; they show up on your dashboard.
APP_TITLE = os.getenv("APP_TITLE", "AI Knowledge Search")
APP_PUBLIC_URL = os.getenv("APP_PUBLIC_URL", "http://localhost:5173")


class LLMError(RuntimeError):
    """An upstream model failure. The API layer maps this to a 502."""


class LLMClient:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self._fake = USE_FAKE_LLM
        if self._fake:
            self.client = None
        else:
            self.client = OpenAI(
                api_key=api_key or OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": APP_PUBLIC_URL,
                    "X-Title": APP_TITLE,
                },
            )

    def complete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        chosen_model = model or DEFAULT_MODEL
        max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        temperature = (
            DEFAULT_TEMPERATURE if temperature is None else float(temperature)
        )

        if self._fake:
            return (
                f"[USE_FAKE_LLM] Simulated answer from '{chosen_model}'. "
                f"Prompt was {len(prompt)} characters, temperature={temperature}, "
                f"max_tokens={max_tokens}."
            )

        # No silent fallback. The previous version quietly swapped an unknown
        # model for the default, which turned a stale model list into an
        # app-wide outage the moment that default was retired.
        if not is_allowed(chosen_model):
            raise LLMError(
                f"Model '{chosen_model}' is not available. It may have been retired by "
                f"its provider, or it exceeds this deployment's price limits. "
                f"See GET /llm-models for the current list."
            )

        try:
            resp = self.client.chat.completions.create(
                model=chosen_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise LLMError(f"Model '{chosen_model}' failed: {exc}") from exc

        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise LLMError(f"Model '{chosen_model}' returned no choices.")

        return choices[0].message.content or ""


def get_available_models(include_all: bool = False) -> List[Dict]:
    """Curated LLM list, fetched live from OpenRouter. See app/models_catalog.py."""
    return get_catalog(include_all=include_all)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

AVAILABLE_EMBEDDING_MODELS: Dict[str, Dict[str, any]] = {
    "all-MiniLM-L6-v2": {
        "label": "SBERT - all-MiniLM-L6-v2",
        "type": "local",
        "dimension": 384,
        "description": "Fast, lightweight model (local, free)"
    },
    "BAAI/bge-base-en-v1.5": {
        "label": "BGE - bge-base-en-v1.5",
        "type": "local",
        "dimension": 768,
        "description": "Strong general-purpose model (local, free)"
    },
    "intfloat/e5-base": {
        "label": "E5 - e5-base",
        "type": "local",
        "dimension": 768,
        "description": "Efficient embedding model (local, free)"
    },
    "intfloat/multilingual-e5-base": {
        "label": "E5 - multilingual-e5-base",
        "type": "local",
        "dimension": 768,
        "description": "Multilingual support (local, free)"
    },
    "hkunlp/instructor-large": {
        "label": "INSTRUCTOR - instructor-large",
        "type": "local",
        "dimension": 768,
        "description": "High quality, instruction-aware (local, free, heavier)"
    },
    "Alibaba-NLP/gte-large-en-v1.5": {
        "label": "GTE - Alibaba GTE-large (v1.5)",
        "type": "local",
        "dimension": 1024,
        "description": "State-of-the-art quality, matches OpenAI (local, free)"
    },
    "jinaai/jina-embeddings-v2-base-en": {
        "label": "Jina AI - v2-base-en",
        "type": "local",
        "dimension": 768,
        "description": "Optimized for long documents, 8K context (local, free)"
    },
    "text-embedding-3-small": {
        "label": "OpenAI - text-embedding-3-small",
        "type": "openai",
        "dimension": 1536,
        "description": "OpenAI's efficient model (API, paid)"
    },
    "text-embedding-3-large": {
        "label": "OpenAI - text-embedding-3-large",
        "type": "openai",
        "dimension": 3072,
        "description": "OpenAI's highest quality model (API, paid)"
    }
}

class EmbeddingClient:
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.model_type = AVAILABLE_EMBEDDING_MODELS.get(self.model_name, {}).get("type", "local")
        
        if self.model_type == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError(
                    f"OpenAI API key required for model '{self.model_name}'. "
                    "Set OPENAI_API_KEY in your .env file."
                )
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = None
        else:
            if "Alibaba-NLP" in self.model_name or "gte-large" in self.model_name:
                self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(self.model_name)
            self.openai_client = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if self.model_type == "openai":
            return self._embed_openai(texts)
        else:
            return self._embed_local(texts)
    
    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings.tolist()
    
    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        batch_size = 2048
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.openai_client.embeddings.create(
                input=batch,
                model=self.model_name
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        return self.embed([text])[0]

def get_embedding_dimension(model_name: str) -> int:
    model_info = AVAILABLE_EMBEDDING_MODELS.get(model_name, {})
    return model_info.get("dimension", 384)
