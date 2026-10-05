# Credit-Risk_ai_by_surojit

An upload-driven credit and business data analysis application. The Streamlit app profiles the active CSV, Excel, or text-based PDF upload, calculates available financial measures, surfaces data-derived patterns and warnings, and exports reports from that analysis. Without an upload, the application asks the user to provide data rather than showing demo results.

## Financial statement uploads

The Streamlit application accepts CSV, XLSX, XLS, and text-based PDF statements. Upload processing runs locally: it reads workbook sheets and PDF tables/text, detects common financial fields, parses Indian-number formats and units, calculates available measures, and reports observed data coverage. Scanned-image PDFs are not OCR processed.

Dataset-specific supervised models are evaluated only when an uploaded dataset contains an evaluable target; these are hold-out metrics, not row-level predictions or calibrated probabilities of default. Where no target is available, the app reports transparent rule-based indices, statistical patterns, and anomaly signals. Rule-based indices are heuristic points, not probabilities of default or lending decisions. PDF and Excel reports are generated from the active upload's analysis.

## Free LLM copilot (optional)

The AI Copilot and Dataset Copilot answer from the active upload using a transparent rule-based engine by default; no network call is made. To generate answers with a free language model instead, configure any OpenAI-compatible chat endpoint through Streamlit secrets (`.streamlit/secrets.toml`) or environment variables:

```toml
LLM_API_BASE = "http://localhost:11434/v1"   # local Ollama server: free, no key, data stays on the machine
LLM_MODEL = "llama3.2"
```

Free cloud options include Groq's free tier (`LLM_API_BASE = "https://api.groq.com/openai/v1"` plus `LLM_API_KEY`) and OpenRouter's `:free` models (`https://openrouter.ai/api/v1`). Only the derived credit context for the selected record, or compact dataset metadata, is sent to the endpoint — never the uploaded file. If the endpoint is unreachable, times out, or returns an empty answer, the copilot automatically falls back to the rule-based engine, and every answer is labeled with its source.

## Synthetic development data

`python -m msme_ews.demo data/demo_msme.csv --companies 240 --periods 3` (with `src` on `PYTHONPATH`) writes a deterministic synthetic company-period CSV for development and tests. The application never loads it; uploads are the only data source for analysis, and models trained on this generator are flagged and refused by the prediction API.
