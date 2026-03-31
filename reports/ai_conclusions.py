# reports/ai_conclusions.py
# ─────────────────────────────────────────────────────────────
# Calls the Google Gemini API to generate Conclusions, Research Gaps,
# and Future Work sections based on actual training results.
#
# Model  : gemini-2.0-flash  (free tier — 1,500 req/day, 1M tokens/min)
# SDK    : google-genai  (pip install google-genai)
# Key    : set GEMINI_API_KEY in your .env or environment
#
# Returns a dict with three string keys:
#   conclusions, research_gaps, future_work
#
# Falls back to rule-based text if the API call fails,
# so the report always has content.
# ─────────────────────────────────────────────────────────────

import json
import os
import re


# ── Prompt builder ─────────────────────────────────────────

def _build_prompt(runs: list[dict], profile: dict, meta: dict) -> str:
    """
    Build the user prompt sent to Gemini from training run data.
    Keeps it concise and factual so the model stays grounded.
    """
    target_col   = runs[0].get("target_col",   "target")  if runs else "target"
    dataset_name = runs[0].get("dataset_name", "unknown") if runs else "unknown"

    # Summarise each run
    # FIX: guard against missing/non-numeric metric values before using :.4f
    run_summaries = []
    for r in runs:
        m    = r.get("metrics", {})
        bp   = r.get("best_params")
        auto = f" (auto-tuned: {bp})" if bp else " (default params)"

        def _fmt(key: str) -> str:
            v = m.get(key)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"

        run_summaries.append(
            f"- {r.get('model_label', r.get('model_key', '?'))}{auto}: "
            f"R2={_fmt('R2')}, RMSE={_fmt('RMSE')}, MAE={_fmt('MAE')}"
        )

    # Dataset profile summary
    profile_lines = []
    if profile:
        profile_lines = [
            f"  Rows: {profile.get('n_rows', '?')}",
            f"  Numeric features: {profile.get('n_features', '?')}",
            f"  Missing values: {profile.get('missing_pct', 0):.1f}%",
            f"  Time-series: {'Yes' if profile.get('is_time_series') else 'No'}",
            f"  Warnings: {'; '.join(profile.get('warnings', [])) or 'None'}",
        ]

    prompt = f"""You are assisting in writing a professional ML research report.
Below is a summary of supervised regression training experiments.
Write three concise, factual sections based ONLY on the data provided.

DATASET
  Name: {dataset_name}
  Target variable: {target_col}
{chr(10).join(profile_lines)}

MODEL RESULTS
{chr(10).join(run_summaries)}

ADDITIONAL NOTES FROM USER
{meta.get('notes', 'None provided')}

Write the following three sections. Each must be 3-5 sentences.
Be specific - reference actual model names, R2 values, and dataset characteristics.
Do NOT speculate beyond what the data shows. Do NOT add headings.

Return a valid JSON object with exactly these three keys:
{{
  "conclusions": "...",
  "research_gaps": "...",
  "future_work": "..."
}}

conclusions: Summarise what was found - which model performed best, how good the fit was,
             what the metrics indicate about predictive quality for this target variable.

research_gaps: Identify limitations visible in this experiment - dataset size, missing data,
               feature quality, model family coverage, or validation gaps.

future_work: Suggest concrete next steps - additional models to try, data improvements,
             deployment considerations, or cross-validation strategies.
"""
    return prompt


# ── Fallback text (rule-based) ─────────────────────────────

def _fallback(runs: list[dict], profile: dict) -> dict:
    """
    Generate rule-based conclusions if the API call fails.
    Ensures the report always has content in these sections.
    """
    best   = max(runs, key=lambda r: r.get("metrics", {}).get("R2", -999), default=None)
    target = runs[0].get("target_col", "the target variable") if runs else "the target variable"

    if best:
        m     = best.get("metrics", {})
        label = best.get("model_label", "the best model")
        r2    = m.get("R2",   0)
        rmse  = m.get("RMSE", 0)
        quality = (
            "excellent" if r2 >= 0.95 else
            "strong"    if r2 >= 0.85 else
            "moderate"  if r2 >= 0.70 else "weak"
        )
        conclusions = (
            f"Across {len(runs)} training run(s), {label} achieved the best performance "
            f"with R2 = {r2:.4f} and RMSE = {rmse:.4f} on the test set, indicating "
            f"{quality} predictive performance for {target}. "
            f"The model generalised well to the held-out test partition using the "
            f"configured train/test split strategy."
        )
    else:
        conclusions = "No runs were available to summarise."

    missing_pct = profile.get("missing_pct", 0) if profile else 0
    n_rows      = profile.get("n_rows",      0) if profile else 0

    gaps = (
        f"The dataset contains {n_rows} samples"
        + (
            f" with {missing_pct:.1f}% missing values that were imputed with column means"
            if missing_pct > 0 else ""
        )
        + ". A larger, cleaner dataset would improve model reliability. "
        "Cross-validation on an independent external dataset has not been performed, "
        "which limits confidence in generalisation. "
        "Deep learning architectures (CNN, LSTM) and physics-informed models "
        "were not explored in this experiment."
    )

    future = (
        "Future work should include testing on additional datasets to validate "
        "cross-domain generalisability. Expanding the feature set with domain-specific "
        "engineered variables may improve predictive accuracy. "
        "Deploying the best model in a real-time monitoring pipeline and comparing "
        "against mechanistic or first-principles models would provide stronger validation."
    )

    return {
        "conclusions":   conclusions,
        "research_gaps": gaps,
        "future_work":   future,
    }


# ── Main API call ──────────────────────────────────────────

def generate_ai_sections(
    runs:    list[dict],
    profile: dict,
    meta:    dict,
) -> dict:
    """
    Call Google Gemini API to generate Conclusions, Research Gaps, Future Work.

    Parameters
    ----------
    runs    : list of saved run dicts (with metrics, model_label, etc.)
    profile : dataset_profile dict from dataset_profiler
    meta    : report_cfg dict (title, author, notes, etc.)

    Returns
    -------
    dict with keys: conclusions, research_gaps, future_work
    Each value is a string paragraph ready for insertion into the report.
    Always returns non-empty content — falls back to rule-based text on any failure.
    """
    # FIX: always attempt fallback first so we have guaranteed content,
    # then try to upgrade with Gemini if the key is available.
    fallback_result = _fallback(runs, profile)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return fallback_result

    try:
        from google import genai
        from google.genai import types as genai_types

        prompt = _build_prompt(runs, profile, meta)

        client   = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model    = "gemini-2.0-flash",
            contents = prompt,
            config   = genai_types.GenerateContentConfig(
                system_instruction = (
                    "You are an expert machine learning researcher writing sections of "
                    "a technical academic report. Be concise, precise, and formal. "
                    "Respond ONLY with a valid JSON object - no markdown fences, no preamble."
                ),
                max_output_tokens = 1000,
                temperature       = 0.4,
            ),
        )

        raw = (response.text or "").strip()

        # Strip markdown fences if Gemini wraps the JSON anyway
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        parsed = json.loads(raw)

        # FIX: validate each key has actual content; fall back per-key if empty
        result = {}
        for key in ("conclusions", "research_gaps", "future_work"):
            val = str(parsed.get(key, "")).strip()
            result[key] = val if val else fallback_result[key]

        return result

    except Exception:
        return fallback_result
