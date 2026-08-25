import os
from typing import Dict, Optional, List

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.models_catalog import (
    OPENROUTER_BASE_URL,
    get_catalog,
    get_embedding_catalog,
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

# Locally-run models. These are genuinely fixed: the weights are downloaded to
# disk by download_models.py, so the list is a property of this deployment
# rather than of any provider's catalog. Free, private, no network at query
# time -- but they hold the process's memory and slow startup.
LOCAL_EMBEDDING_MODELS: Dict[str, Dict[str, any]] = {
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
    }
}

# Dimensions learned from real responses. OpenRouter does not publish embedding
# dimensions in its catalog, so the first successful embed records the true
# vector length instead of guessing.
_learned_dimensions: Dict[str, int] = {}


def get_embedding_models(include_all: bool = False) -> Dict[str, Dict[str, any]]:
    """
    Local models plus the live OpenRouter embedding catalog, keyed by model ID.

    Local entries are always present. Remote entries are refreshed from the
    provider, so a retired embedding model stops being offered on its own.
    """
    merged: Dict[str, Dict[str, any]] = dict(LOCAL_EMBEDDING_MODELS)

    try:
        remote = get_embedding_catalog(
            include_all=include_all,
            # Don't offer to bill us for weights already sitting on disk.
            exclude_basenames=list(LOCAL_EMBEDDING_MODELS.keys()),
        )
    except Exception:
        # The local models are enough to keep the app usable.
        return merged

    for model in remote:
        price = model["prompt_price_per_m"]
        cost = "free" if model["is_free"] else f"${price:.3f}/M tokens"
        merged[model["id"]] = {
            "label": f"{model['vendor']} - {model['label']}",
            "type": "openrouter",
            "dimension": _learned_dimensions.get(model["id"], 0),
            "description": f"API via OpenRouter ({cost})",
        }

    return merged


def is_valid_embedding_model(model_name: str) -> bool:
    return model_name in get_embedding_models()


class EmbeddingClient:
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.model_type = get_embedding_models().get(self.model_name, {}).get("type", "local")

        if self.model_type == "openrouter":
            if not OPENROUTER_API_KEY:
                raise RuntimeError(
                    f"OPENROUTER_API_KEY required for embedding model '{self.model_name}'."
                )
            from openai import OpenAI
            self.openai_client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": APP_PUBLIC_URL,
                    "X-Title": APP_TITLE,
                },
            )
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
        
        if self.model_type == "openrouter":
            return self._embed_api(texts)
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
    
    def _embed_api(self, texts: List[str]) -> List[List[float]]:
        """OpenAI-compatible embeddings call, routed through OpenRouter."""
        batch_size = 2048
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model=self.model_name,
                )
            except Exception as exc:
                raise LLMError(
                    f"Embedding model '{self.model_name}' failed: {exc}"
                ) from exc
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        # Record the real vector length; the catalog does not publish it.
        if all_embeddings:
            _learned_dimensions[self.model_name] = len(all_embeddings[0])

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        return self.embed([text])[0]

def get_embedding_dimension(model_name: str) -> int:
    if model_name in _learned_dimensions:
        return _learned_dimensions[model_name]
    info = get_embedding_models().get(model_name, {})
    return info.get("dimension") or 0
