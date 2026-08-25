import type { LLMModel } from "../api";

const BASE_CLASS =
  "w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-100";

type ModelSelectProps = {
  models: LLMModel[];
  value: string;
  onChange: (id: string) => void;
  loading?: boolean;
  error?: string | null;
  className?: string;
  disabled?: boolean;
};

function priceTag(model: LLMModel): string {
  if (model.is_free) return "free";
  return `$${model.prompt_price_per_m.toFixed(2)}/M`;
}

/**
 * One dropdown for every model picker in the app.
 *
 * Options are grouped by vendor so a list spanning Meta, OpenAI, Moonshot,
 * DeepSeek, Qwen and friends stays scannable, and each option carries its live
 * input price so the cost of an experiment is visible before running it.
 */
export default function ModelSelect({
  models,
  value,
  onChange,
  loading = false,
  error = null,
  className,
  disabled = false,
}: ModelSelectProps) {
  const selectClass = className ?? BASE_CLASS;

  if (loading) {
    return (
      <select className={selectClass} disabled>
        <option>Loading models…</option>
      </select>
    );
  }

  if (error || models.length === 0) {
    return (
      <select className={selectClass} disabled title={error ?? undefined}>
        <option>{error ? "Model list unavailable" : "No models available"}</option>
      </select>
    );
  }

  const byVendor = new Map<string, LLMModel[]>();
  for (const model of models) {
    const bucket = byVendor.get(model.vendor);
    if (bucket) bucket.push(model);
    else byVendor.set(model.vendor, [model]);
  }

  return (
    <select
      className={selectClass}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
    >
      {[...byVendor.entries()].map(([vendor, vendorModels]) => (
        <optgroup key={vendor} label={vendor}>
          {vendorModels.map((model) => (
            <option key={model.id} value={model.id} title={model.description}>
              {model.label} ({priceTag(model)})
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}
