import os
from typing import Dict, Optional
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set. "
        "Set it before starting the backend."
    )

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

GROQ_AVAILABLE_MODELS: Dict[str, str] = {
    "llama-3.1-8b-instant": "Llama 3.1 8B Instant (128k, fast & cheap)",
    "llama-3.3-70b-versatile": "Llama 3.3 70B Versatile (128k, powerful)",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout 17B 16E (128k)",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick 17B 128E (128k)",
    "openai/gpt-oss-20b": "GPT OSS 20B (128k)",
    "openai/gpt-oss-120b": "GPT OSS 120B (128k)",
    "meta-llama/llama-guard-4-12b": "Llama Guard 4 12B (Safety)",
    "openai/gpt-oss-safeguard-20b": "GPT OSS Safeguard 20B (Safety)",
    "moonshotai/kimi-k2-instruct-0905": "Kimi K2 Instruct 0905 (256k, expensive)",
    "qwen/qwen3-32b": "Qwen3 32B (131k)",
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
