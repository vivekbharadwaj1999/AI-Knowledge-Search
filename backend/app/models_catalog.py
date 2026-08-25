"""
Live LLM catalog backed by OpenRouter.

This module replaces the old hardcoded model dict. Instead of
a hand-maintained list of model IDs (which silently rots every time a provider
retires a model), the catalog is fetched from OpenRouter's public ``/models``
endpoint and filtered down to an affordable, capable subset.

Curation happens server-side so the browser never receives the full ~300-model
catalog. Every rule below is overridable from ``.env`` -- the point is that the
app stores *selection rules*, never model names.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# --- Curation rules -------------------------------------------------------
# Price ceilings are the budget guard: OpenRouter carries models at $15+ per
# million tokens, and this is a public demo. Anything above these limits is
# never offered and never accepted.
MAX_PROMPT_PRICE_PER_M = float(os.getenv("MODEL_MAX_PROMPT_PRICE", "1.00"))
MAX_COMPLETION_PRICE_PER_M = float(os.getenv("MODEL_MAX_COMPLETION_PRICE", "3.00"))
MIN_CONTEXT_LENGTH = int(os.getenv("MODEL_MIN_CONTEXT", "32000"))
MAX_MODELS_PER_VENDOR = int(os.getenv("MODEL_MAX_PER_VENDOR", "3"))
CATALOG_TTL_SECONDS = int(os.getenv("MODEL_CATALOG_TTL", "3600"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("MODEL_CATALOG_TIMEOUT", "15"))

# Which labs to offer. Vendors are pinned here rather than models, on purpose:
# a company name is stable in a way a model ID is not. "meta-llama" will still
# be meta-llama next year; llama-3.1-8b-instant lasted nine months.
#
# OpenRouter carries ~48 vendors under any sane price ceiling, most of them
# niche roleplay or coding-assistant shops. Without this the dropdown runs past
# 100 entries. Set MODEL_VENDORS="" to allow every vendor, bounded by
# MODEL_MAX_VENDORS.
DEFAULT_VENDORS = [
    "openai",
    "anthropic",
    "google",
    "meta",
    "meta-llama",
    "mistralai",
    "deepseek",
    "qwen",
    "moonshotai",
    "x-ai",
    "microsoft",
    "amazon",
    "cohere",
    "nvidia",
]
_vendors_raw = os.getenv("MODEL_VENDORS")
VENDOR_ALLOWLIST = (
    [v.strip().lower() for v in _vendors_raw.split(",") if v.strip()]
    if _vendors_raw is not None
    else list(DEFAULT_VENDORS)
)

# Backstop used only when VENDOR_ALLOWLIST is empty: keep the vendors
# contributing the most eligible models rather than all 48 of them.
MAX_VENDORS = int(os.getenv("MODEL_MAX_VENDORS", "12"))

# Embedding models are priced per input token only and are far cheaper than
# chat models, so they get their own (much lower) ceiling.
EMBEDDING_MAX_PRICE_PER_M = float(os.getenv("EMBEDDING_MAX_PRICE", "0.20"))

# Embedding curation. The goal is a short list of models that differ from each
# other, rather than five near-identical variants from one vendor.
#   - models duplicating a locally-hosted one are dropped (paying per token for
#     weights already on disk is pointless)
#   - vendors are then taken round-robin, so the cap trims redundancy inside a
#     vendor before it trims variety across vendors
EMBEDDING_MAX_MODELS = int(os.getenv("EMBEDDING_MAX_MODELS", "14"))
EMBEDDING_MAX_PER_VENDOR = int(os.getenv("EMBEDDING_MAX_PER_VENDOR", "2"))

# Substrings that disqualify a model ID. Defaults cover two classes of model
# that pass every numeric filter but break a synchronous chat app:
#   :batch  -- asynchronous batch endpoints; they never return a completion
#              inline, so selecting one would hang the request
#   safety/guard/moderation -- classifiers that emit verdicts, not answers
MODEL_EXCLUDE_PATTERNS = [
    p.strip().lower()
    for p in os.getenv(
        "MODEL_EXCLUDE_PATTERNS",
        ":batch,-safety,-guard,guardrail,moderation",
    ).split(",")
    if p.strip()
]

# Models pinned here always survive curation, even if they breach the rules.
# Comma-separated list of OpenRouter model IDs.
PINNED_MODELS = [m.strip() for m in os.getenv("MODEL_ALLOWLIST", "").split(",") if m.strip()]

# Cosmetic only -- an unknown vendor key is title-cased rather than dropped, so
# a new provider appearing in the catalog still renders sensibly.
VENDOR_LABELS: Dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "meta": "Meta",
    "meta-llama": "Meta",
    "mistralai": "Mistral",
    "moonshotai": "Moonshot (Kimi)",
    "deepseek": "DeepSeek",
    "qwen": "Qwen (Alibaba)",
    "x-ai": "xAI",
    "microsoft": "Microsoft",
    "nvidia": "NVIDIA",
    "amazon": "Amazon",
    "cohere": "Cohere",
    "nousresearch": "Nous Research",
    "perplexity": "Perplexity",
    "liquid": "Liquid AI",
    "ai21": "AI21 Labs",
    "inflection": "Inflection",
    # Embedding-model publishers
    "baai": "BAAI",
    "voyageai": "Voyage AI",
    "intfloat": "intfloat (E5)",
    "thenlper": "thenlper (GTE)",
    "sentence-transformers": "Sentence Transformers",
    "jinaai": "Jina AI",
    "hkunlp": "INSTRUCTOR",
    "alibaba-nlp": "Alibaba (GTE)",
}


class CatalogError(RuntimeError):
    """Raised when the model catalog cannot be loaded and no cached copy exists."""


_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {"fetched_at": 0.0, "all": [], "curated": []}


def _price_per_million(raw: Optional[str]) -> Optional[float]:
    """OpenRouter quotes prices per token as strings; convert to $ per 1M tokens."""
    if raw is None:
        return None
    try:
        return float(raw) * 1_000_000
    except (TypeError, ValueError):
        return None


def _fetch_raw(params: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Fetch a catalog listing. This endpoint is public and needs no API key."""
    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/models"
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = client.get(url, params=params or {})
        response.raise_for_status()
        payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        raise CatalogError("Unexpected response shape from OpenRouter /models")
    return data


def _fetch_embeddings_raw() -> List[Dict[str, Any]]:
    """
    Embedding models come from a separate listing.

    The default /models response covers text/image/audio generation only --
    it carries no embedding entries at all, and nothing in an entry's
    architecture block marks one. The query parameter is the only reliable
    way to identify them, so we let the endpoint do the classifying rather
    than inspecting fields ourselves.
    """
    return _fetch_raw({"output_modalities": "embeddings"})


def _normalize(entry: Dict[str, Any], *, is_embedding: bool = False) -> Optional[Dict[str, Any]]:
    """Reduce an OpenRouter catalog entry to the fields the app cares about."""
    model_id = entry.get("id")
    if not model_id:
        return None

    pricing = entry.get("pricing") or {}
    prompt_price = _price_per_million(pricing.get("prompt"))
    if prompt_price is None:
        return None
    # Embedding models bill on input only and may omit a completion price
    # entirely; treating that as free keeps them in the catalog instead of
    # silently dropping every one of them.
    completion_price = _price_per_million(pricing.get("completion")) or 0.0

    architecture = entry.get("architecture") or {}
    input_modalities = architecture.get("input_modalities") or ["text"]
    output_modalities = architecture.get("output_modalities") or ["text"]

    vendor_key = model_id.split("/")[0] if "/" in model_id else "other"
    # OpenRouter prefixes variant namespaces with "~" (e.g. "~google").
    # Fold those into the parent vendor instead of inventing a new one.
    vendor_key = vendor_key.lstrip("~").lower()

    return {
        "id": model_id,
        "label": entry.get("name") or model_id,
        "vendor": VENDOR_LABELS.get(vendor_key, vendor_key.replace("-", " ").title()),
        "vendor_key": vendor_key,
        "context_length": int(entry.get("context_length") or 0),
        "prompt_price_per_m": round(prompt_price, 4),
        "completion_price_per_m": round(completion_price, 4),
        "is_free": prompt_price == 0 and completion_price == 0,
        "input_modalities": input_modalities,
        "output_modalities": output_modalities,
        # Set from which listing the entry came, not from its own metadata.
        "is_embedding": is_embedding,
        "description": (entry.get("description") or "").strip()[:300],
    }


def _is_eligible(model: Dict[str, Any]) -> bool:
    """Capability floor and price ceiling. Pinned models bypass this."""
    if model.get("is_embedding"):
        return False
    model_id = model["id"].lower()
    if any(pattern in model_id for pattern in MODEL_EXCLUDE_PATTERNS):
        return False
    if "text" not in model["input_modalities"]:
        return False
    # Text out and nothing else. A model advertising ["text", "image"] is an
    # image generator that happens to also emit text -- not what this app wants.
    if [m for m in model["output_modalities"] if m != "text"]:
        return False
    if "text" not in model["output_modalities"]:
        return False
    if model["context_length"] < MIN_CONTEXT_LENGTH:
        return False
    if model["prompt_price_per_m"] > MAX_PROMPT_PRICE_PER_M:
        return False
    if model["completion_price_per_m"] > MAX_COMPLETION_PRICE_PER_M:
        return False
    return True


def _pick_for_vendor(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Given one vendor's eligible models sorted most-expensive-first, keep the
    strongest few plus the cheapest. Price is a rough capability proxy, and
    since everything here is already under the ceiling, the expensive end is
    still cheap. Always keeping the cheapest guarantees a free or near-free
    option per vendor stays reachable.
    """
    if len(models) <= MAX_MODELS_PER_VENDOR:
        return models

    picked = models[: max(MAX_MODELS_PER_VENDOR - 1, 1)]
    cheapest = models[-1]
    if cheapest["id"] not in {m["id"] for m in picked}:
        picked = picked + [cheapest]
    return picked


def _curate(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn the full catalog into a short, vendor-diverse dropdown list."""
    eligible = [m for m in models if _is_eligible(m)]

    if VENDOR_ALLOWLIST:
        eligible = [m for m in eligible if m["vendor_key"] in VENDOR_ALLOWLIST]

    # Group by display name rather than vendor key: OpenRouter splits some
    # companies across several namespaces ("meta" and "meta-llama"), and keying
    # on the raw prefix would hand each one its own quota.
    by_vendor: Dict[str, List[Dict[str, Any]]] = {}
    for model in sorted(
        eligible,
        key=lambda m: (-m["prompt_price_per_m"], -m["context_length"]),
    ):
        by_vendor.setdefault(model["vendor"], []).append(model)

    # With no allowlist, keep only the best-represented vendors so an open
    # configuration still yields a usable dropdown.
    vendors = list(by_vendor)
    if not VENDOR_ALLOWLIST and len(vendors) > MAX_VENDORS:
        vendors.sort(key=lambda v: (-len(by_vendor[v]), v.lower()))
        vendors = vendors[:MAX_VENDORS]

    curated: List[Dict[str, Any]] = []
    for vendor in vendors:
        curated.extend(_pick_for_vendor(by_vendor[vendor]))

    # Pinned models always make the cut, rules or not.
    seen = {m["id"] for m in curated}
    for model in models:
        if model["id"] in PINNED_MODELS and model["id"] not in seen:
            curated.append(model)
            seen.add(model["id"])

    curated.sort(key=lambda m: (m["vendor"].lower(), m["prompt_price_per_m"]))
    return curated


def _is_embedding(model: Dict[str, Any]) -> bool:
    return bool(model.get("is_embedding"))


def _is_embedding_eligible(model: Dict[str, Any]) -> bool:
    if not _is_embedding(model):
        return False
    model_id = model["id"].lower()
    if any(pattern in model_id for pattern in MODEL_EXCLUDE_PATTERNS):
        return False
    if model["prompt_price_per_m"] > EMBEDDING_MAX_PRICE_PER_M:
        return False
    return True


def _basename(model_id: str) -> str:
    """Last path segment, lowercased — 'sentence-transformers/all-MiniLM-L6-v2' -> 'all-minilm-l6-v2'."""
    return model_id.split("/")[-1].split(":")[0].strip().lower()


def get_embedding_catalog(
    *,
    include_all: bool = False,
    exclude_basenames: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Embedding models from the live catalog, curated for variety.

    ``exclude_basenames`` drops models already available locally — the caller
    supplies them, since this module must not import config.
    """
    refresh()
    with _cache_lock:
        models = list(_cache["all"])

    embeddings = [m for m in models if _is_embedding(m)]

    if include_all:
        embeddings.sort(key=lambda m: (m["prompt_price_per_m"], m["vendor"].lower()))
        return embeddings

    excluded = {_basename(b) for b in (exclude_basenames or [])}
    eligible = [
        m
        for m in embeddings
        if _is_embedding_eligible(m) and _basename(m["id"]) not in excluded
    ]

    # Strongest first within each vendor; price is the available proxy.
    by_vendor: Dict[str, List[Dict[str, Any]]] = {}
    for model in sorted(eligible, key=lambda m: (-m["prompt_price_per_m"], m["id"])):
        by_vendor.setdefault(model["vendor"], []).append(model)

    # Round-robin: every vendor contributes one before any contributes two.
    vendors = sorted(by_vendor, key=lambda v: (-len(by_vendor[v]), v.lower()))
    picked: List[Dict[str, Any]] = []
    for rank in range(EMBEDDING_MAX_PER_VENDOR):
        for vendor in vendors:
            if len(picked) >= EMBEDDING_MAX_MODELS:
                break
            if rank < len(by_vendor[vendor]):
                picked.append(by_vendor[vendor][rank])
        if len(picked) >= EMBEDDING_MAX_MODELS:
            break

    # Pinned models always survive, whatever the caps did.
    seen = {m["id"] for m in picked}
    for model in embeddings:
        if model["id"] in PINNED_MODELS and model["id"] not in seen:
            picked.append(model)
            seen.add(model["id"])

    picked.sort(key=lambda m: (m["prompt_price_per_m"], m["vendor"].lower()))
    return picked


def refresh(force: bool = False) -> None:
    """Refresh the cached catalog if it is missing or stale."""
    with _cache_lock:
        age = time.time() - _cache["fetched_at"]
        if not force and _cache["all"] and age <= CATALOG_TTL_SECONDS:
            return

        try:
            raw = _fetch_raw()
            raw_embeddings = _fetch_embeddings_raw()
        except Exception as exc:
            # A stale catalog beats a dead app: only fail if we have nothing.
            if _cache["all"]:
                return
            raise CatalogError(
                f"Could not load the model catalog from OpenRouter: {exc}"
            ) from exc

        normalized = [n for n in (_normalize(e) for e in raw) if n]
        normalized += [
            n
            for n in (_normalize(e, is_embedding=True) for e in raw_embeddings)
            if n
        ]
        if not normalized:
            if _cache["all"]:
                return
            raise CatalogError("OpenRouter returned an empty model catalog")

        _cache["all"] = normalized
        _cache["curated"] = _curate(normalized)
        _cache["fetched_at"] = time.time()


def get_catalog(*, include_all: bool = False, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Return the curated model list, or the entire catalog when ``include_all``.

    ``include_all`` is an inspection escape hatch (``GET /llm-models?all=true``);
    it is not what the dropdown renders.
    """
    refresh(force=force_refresh)
    with _cache_lock:
        return list(_cache["all"] if include_all else _cache["curated"])


def find_model(model_id: str) -> Optional[Dict[str, Any]]:
    for model in get_catalog(include_all=True):
        if model["id"] == model_id:
            return model
    return None


def is_allowed(model_id: str) -> bool:
    """
    A model is usable if it exists and either passes the rules or is pinned.

    Deliberately broader than the curated dropdown: anything genuinely
    affordable can be driven through the API for experimentation, while models
    above the price ceiling are refused outright.
    """
    if model_id in PINNED_MODELS:
        return True
    model = find_model(model_id)
    return bool(model and _is_eligible(model))


def get_model_label(model_id: str) -> str:
    model = find_model(model_id)
    return model["label"] if model else model_id


def describe_rules() -> Dict[str, Any]:
    """Surface the active curation rules so the UI can explain the filtering."""
    return {
        "max_prompt_price_per_m": MAX_PROMPT_PRICE_PER_M,
        "max_completion_price_per_m": MAX_COMPLETION_PRICE_PER_M,
        "min_context_length": MIN_CONTEXT_LENGTH,
        "max_models_per_vendor": MAX_MODELS_PER_VENDOR,
        "vendors": VENDOR_ALLOWLIST or f"any (top {MAX_VENDORS})",
        "pinned": PINNED_MODELS,
        "excluded_patterns": MODEL_EXCLUDE_PATTERNS,
        "max_embedding_price_per_m": EMBEDDING_MAX_PRICE_PER_M,
    }
