# pages/6_Report_Generator.py
# ─────────────────────────────────────────────────────────────
# v4 — Smart per-run artifact selection.
# Step 2 shows one expander per selected run. Each expander
# contains checkboxes limited to artifacts actually saved for
# that run. Only selected items go into the report.
# ─────────────────────────────────────────────────────────────

from config.sidebar_config import apply_page_config, render_sidebar
apply_page_config("Report Generator — MLForge")

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from backend.session_store import (
    get_runs, runs_to_dataframe, get_best_run,
    available_artifacts, artifact_label, ARTIFACT_LABELS,
)
from config.page_header import render_header

css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

render_sidebar()
render_header()
runs = get_runs()

st.title("Report Generator")
st.caption("Configure, select content per run, write sections, then export.")

if not runs:
    st.info("No runs saved yet. Train models on Pages 1–4 and save runs first.", icon="💾")
    st.stop()

# ══════════════════════════════════════════════════════════
# STEP 1 — Report settings
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Step 1 — Report settings</div>',
            unsafe_allow_html=True)

rc1, rc2 = st.columns(2)
with rc1:
    report_title  = st.text_input("Report title", value="MLForge Training Report")
    author_name   = st.text_input("Author name",  placeholder="Your name")
    institution   = st.text_input(
        "Institution / Course",
        placeholder="e.g. Jadavpur University · ML Minor Project"
    )

with rc2:
    export_format    = st.selectbox("Export format", ["Word (.docx)", "PDF", "LaTeX (.tex)"])
    include_plots    = st.toggle("Include plots", value=True)
    include_all_runs = st.toggle(
        "Include all saved runs", value=True,
        help="Off → only the best run by R² is pre-selected."
    )
    extra_notes = st.text_area(
        "Additional context (optional)",
        placeholder="Dataset background, study objectives, domain specifics…",
        height=90,
    )

# ══════════════════════════════════════════════════════════
# STEP 2 — Select runs + per-run content
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Step 2 — Select runs & content</div>',
            unsafe_allow_html=True)

comp_df      = runs_to_dataframe(runs)
best_run_obj = get_best_run()
default_ids  = (
    [r["run_id"] for r in runs]
    if include_all_runs
    else ([best_run_obj["run_id"]] if best_run_obj else [])
)

# Compact overview table
st.dataframe(
    comp_df[["#", "time", "model", "R²", "RMSE", "artifacts", "notes"]],
    width='stretch',
    hide_index=True,
)

selected_ids = st.multiselect(
    "Run IDs to include",
    options=[r["run_id"] for r in runs],
    default=default_ids,
    format_func=lambda rid: (
        f"#{rid} — "
        + next((r["model_label"] for r in runs if r["run_id"] == rid), "?")
    ),
)
selected_runs = [r for r in runs if r["run_id"] in selected_ids]

if not selected_runs:
    st.warning("Select at least one run to continue.")
    st.stop()

st.markdown(
    "Expand each run below to choose which saved artifacts to include in the report. "
    "Only artifacts that were saved on Page 1 are shown."
)

# per_run_include: {run_id: [artifact_key, ...]}
per_run_include: dict[int, list[str]] = {}

for run in selected_runs:
    rid           = run["run_id"]
    model_lbl     = run["model_label"]
    m             = run["metrics"]
    avail_keys    = available_artifacts(run)

    with st.expander(
        f"Run #{rid} — {model_lbl}  "
        f"(R²={m.get('R2',0):.4f}, RMSE={m.get('RMSE',0):.4f})  "
        f"· {len(avail_keys)} artifact(s) saved",
        expanded=True,
    ):
        if not avail_keys:
            st.caption(
                "No artifacts were saved for this run. "
                "Only metrics and AI interpretation will appear in the report."
            )
            per_run_include[rid] = []
            continue

        # Show metrics always (not an artifact, always included)
        st.caption("✓ Metrics table — always included")

        # Checkboxes for each saved artifact
        run_selected_keys = []
        cols = st.columns(2)
        for i, key in enumerate(avail_keys):
            with cols[i % 2]:
                checked = st.checkbox(
                    artifact_label(key),
                    value=True,
                    key=f"art_{rid}_{key}",
                )
                if checked:
                    run_selected_keys.append(key)

        per_run_include[rid] = run_selected_keys

        # Summary of what will appear
        if run_selected_keys:
            st.caption(
                f"Will include: {', '.join(artifact_label(k) for k in run_selected_keys)}"
            )

# ══════════════════════════════════════════════════════════
# STEP 3 — Written sections (AI / Manual per section)
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Step 3 — Written sections</div>',
            unsafe_allow_html=True)
st.markdown(
    "Each section can be **AI-generated** (Claude analyses your results) "
    "or **written manually**. Toggle independently per section."
)

for key in ["sec_conclusions", "sec_gaps", "sec_future"]:
    if key not in st.session_state:
        st.session_state[key] = ""

report_cfg_for_ai = {
    "title":       report_title,
    "author":      author_name,
    "institution": institution,
    "notes":       extra_notes,
}
dataset_profile = st.session_state.get("dataset_profile", {})


def _section_widget(
    label:       str,
    state_key:   str,
    ai_key:      str,
    placeholder: str,
    caption:     str,
) -> str:
    st.markdown(f"#### {label}")
    st.caption(caption)

    toggle_key = f"toggle_{state_key}"
    if toggle_key not in st.session_state:
        st.session_state[toggle_key] = "AI Generated"

    mode = st.radio(
        f"Mode for {label}",
        ["AI Generated", "Write my own"],
        horizontal=True,
        key=toggle_key,
        label_visibility="collapsed",
    )

    if mode == "AI Generated":
        gen_btn = st.button(f"Generate {label}", key=f"gen_{state_key}")
        if gen_btn:
            with st.spinner(f"Claude is writing {label.lower()}…"):
                from reports.ai_conclusions import generate_ai_sections
                result = generate_ai_sections(
                    selected_runs, dataset_profile, report_cfg_for_ai
                )
                st.session_state[state_key] = result.get(ai_key, "")

        current = st.session_state[state_key]
        if current:
            st.markdown(
                f"<div style='background:rgba(79,134,198,0.07); "
                f"border-left:3px solid #4f86c6; padding:0.8rem 1rem; "
                f"border-radius:0 6px 6px 0; font-size:0.92rem;'>"
                f"{current}</div>",
                unsafe_allow_html=True,
            )
            st.caption("✓ AI-generated — will appear verbatim in the report.")
        else:
            st.info("Click the button above to generate this section.", icon="🤖")
        return current

    else:
        text = st.text_area(
            label,
            value=st.session_state[state_key],
            height=130,
            label_visibility="collapsed",
            key=f"manual_{state_key}",
            placeholder=placeholder,
        )
        st.session_state[state_key] = text
        return text


conclusions_text = _section_widget(
    label       = "Conclusions",
    state_key   = "sec_conclusions",
    ai_key      = "conclusions",
    placeholder = "Summarise which model performed best and what the metrics indicate…",
    caption     = "Which model performed best, what the metrics indicate about fit quality.",
)
gaps_text = _section_widget(
    label       = "Research Gaps",
    state_key   = "sec_gaps",
    ai_key      = "research_gaps",
    placeholder = "Describe limitations — dataset size, missing data, model coverage…",
    caption     = "Limitations visible in this experiment.",
)
future_text = _section_widget(
    label       = "Future Work",
    state_key   = "sec_future",
    ai_key      = "future_work",
    placeholder = "Describe next steps — more models, better data, deployment…",
    caption     = "Concrete next steps.",
)

# ══════════════════════════════════════════════════════════
# STEP 4 — Preview
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Step 4 — Preview</div>',
            unsafe_allow_html=True)

with st.expander("Preview report structure", expanded=False):
    pc1, pc2 = st.columns([1, 1])
    with pc1:
        st.markdown(f"**{report_title or 'MLForge Training Report'}**")
        if author_name:   st.markdown(f"Author: {author_name}")
        if institution:   st.markdown(f"Institution: {institution}")
        st.markdown(f"Date: {datetime.now().strftime('%d %B %Y')}")
    with pc2:
        profile = st.session_state.get("dataset_profile")
        if profile:
            mc1, mc2 = st.columns(2)
            mc1.metric("Rows",    f"{profile['n_rows']:,}")
            mc2.metric("Features", profile["n_features"])
            mc3, mc4 = st.columns(2)
            mc3.metric("Missing %",  f"{profile['missing_pct']:.1f}%")
            mc4.metric("Time-series","Yes" if profile["is_time_series"] else "No")

    st.divider()
    st.markdown("**Content per run:**")
    for run in selected_runs:
        rid       = run["run_id"]
        selected  = per_run_include.get(rid, [])
        m         = run["metrics"]
        items     = ["Metrics"] + [artifact_label(k) for k in selected]
        st.markdown(
            f"- **Run #{rid} {run['model_label']}** "
            f"(R²={m.get('R2',0):.4f}) — "
            + ", ".join(items)
        )

    sections = ["1. Introduction", "2. Dataset Summary", "3. Model Results"]
    offset   = 3
    if len(selected_runs) > 1:
        offset += 1
        sections.append("4. Runs Comparison")
    sections += [
        f"{offset+1}. Conclusions",
        f"{offset+2}. Research Gaps",
        f"{offset+3}. Future Work",
        f"{offset+4}. References (dynamic)",
    ]
    st.divider()
    st.markdown("**Sections:**")
    for s in sections:
        st.markdown(f"- {s}")

# ══════════════════════════════════════════════════════════
# STEP 5 — Export
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Step 5 — Export</div>',
            unsafe_allow_html=True)
st.caption(
    "References are generated dynamically — only papers relevant to the models "
    "and tools used in your runs will appear."
)

if st.button("Generate report", type="primary"):
    ai_sections = {
        "conclusions":   conclusions_text.strip(),
        "research_gaps": gaps_text.strip(),
        "future_work":   future_text.strip(),
    }

    report_cfg = {
        "title":         report_title or "MLForge Training Report",
        "author":        author_name,
        "institution":   institution,
        "date":          datetime.now().strftime("%d %B %Y"),
        "notes":         extra_notes,
        "include_plots": include_plots,
        "format":        export_format,
    }

    try:
        with st.spinner(f"Building {export_format} report…"):

            kwargs = dict(
                runs            = selected_runs,
                report_cfg      = report_cfg,
                dataset_profile = dataset_profile,
                ai_sections     = ai_sections,
                per_run_include = per_run_include,
            )

            if export_format == "Word (.docx)":
                from reports.docx_exporter import build_docx
                buf, filename = build_docx(**kwargs)
                mime = (
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                )

            elif export_format == "PDF":
                from reports.pdf_exporter import build_pdf
                buf, filename = build_pdf(**kwargs)
                mime = "application/pdf"

            else:
                from reports.latex_exporter import build_latex
                buf, filename = build_latex(**kwargs)
                mime = "application/zip"

        st.success(f"Report ready — **{filename}**")
        st.download_button(
            label=f"⬇  Download {export_format}",
            data=buf,
            file_name=filename,
            mime=mime,
            type="primary",
        )

    except Exception as e:
        st.error(f"Report generation failed: {e}")
        st.exception(e)
