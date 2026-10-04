# Credit-Risk_ai_by_surojit

An upload-driven credit and business data analysis application. The Streamlit app profiles the active CSV, Excel, or text-based PDF upload, calculates available financial measures, surfaces data-derived patterns and warnings, and exports reports from that analysis. Without an upload, the application asks the user to provide data rather than showing demo results.

## Financial statement uploads

The Streamlit application accepts CSV, XLSX, XLS, and text-based PDF statements. Upload processing runs locally: it reads workbook sheets and PDF tables/text, detects common financial fields, parses Indian-number formats and units, calculates available measures, and reports observed data coverage. Scanned-image PDFs are not OCR processed.

Dataset-specific supervised models are evaluated only when an uploaded dataset contains an evaluable target; these are hold-out metrics, not row-level predictions or calibrated probabilities of default. Where no target is available, the app reports transparent rule-based indices, statistical patterns, and anomaly signals. Rule-based indices are heuristic points, not probabilities of default or lending decisions. PDF and Excel reports are generated from the active upload's analysis.
