# Credit-Risk_ai_by_surojit

Early Warning System for MSME Loan Distress. It includes default-probability prediction, SHAP explainability, fairness analysis, model monitoring, and local financial-statement analysis.

## Financial statement uploads

The Streamlit application accepts CSV, XLSX, XLS, and text-based PDF statements. Upload processing runs locally: it reads workbook sheets and PDF tables/text, normalizes common financial labels and Indian-number units, calculates available ratios, and reports the observed financial-field coverage. Scanned-image PDFs are not OCR processed.

When a record contains enough observed inputs, the existing ML model and SHAP explanation are used. Sparse records use explicitly labeled rule-based indicators; these are heuristic risk indices, not calibrated probabilities of default. PDF assessment reports and Excel analysis workbooks are generated offline.
