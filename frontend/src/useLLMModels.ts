import { useEffect, useState } from "react";
import { fetchLLMModels, type LLMModel } from "./api";

type LLMModelsState = {
  models: LLMModel[];
  defaultModel: string;
  loading: boolean;
  error: string | null;
};

/**
 * Loads the LLM catalog from the backend once per mount.
 *
 * The list is served live from OpenRouter and curated server-side, so model
 * names never appear in this codebase. When a provider retires a model it
 * simply stops showing up in the dropdowns instead of breaking every request.
 */
export function useLLMModels(): LLMModelsState {
  const [models, setModels] = useState<LLMModel[]>([]);
  const [defaultModel, setDefaultModel] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const result = await fetchLLMModels();
        if (cancelled) return;
        setModels(result.models);
        setDefaultModel(result.default);
        setError(null);
      } catch (err: any) {
        if (cancelled) return;
        setError(
          err?.response?.data?.detail ??
            err?.message ??
            "Could not load the model list."
        );
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  return { models, defaultModel, loading, error };
}
