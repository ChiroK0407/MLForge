# reports/docx_exporter.py
# ─────────────────────────────────────────────────────────────
# Generates a styled Word (.docx) report using python-docx.
# v2 changes:
#   - Professional cover with accent band + large metadata
#   - Dynamic references (only what was used)
#   - AI-generated Conclusions / Research Gaps / Future Work
#   - Tighter, more intentional typography throughout
# ─────────────────────────────────────────────────────────────

import io
from docx                  import Document
from docx.shared           import Inches, Pt, RGBColor, Cm, Emu
from docx.enum.text        import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table       import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns          import qn
from docx.oxml             import OxmlElement

from reports.report_header_helper import add_report_header_to_doc
from reports.report_builder import build_report_payload


# ── Palette ────────────────────────────────────────────────
NAVY      = RGBColor(0x1a, 0x1a, 0x2e)   # dark navy — headings
ACCENT    = RGBColor(0x4f, 0x86, 0xc6)   # steel blue — accents
ACCENT2   = RGBColor(0x2e, 0x5f, 0x9e)   # deeper blue — cover band
MUTED     = RGBColor(0x6b, 0x7b, 0x8d)   # grey — captions
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
WARN      = RGBColor(0xb0, 0x6b, 0x00)   # amber — warnings
SUCCESS   = RGBColor(0x1a, 0x9e, 0x5c)   # green — good metrics

# ── Hex strings for XML shading ────────────────────────────
HEX_ACCENT  = "4F86C6"
HEX_ACCENT2 = "2E5F9E"
HEX_LIGHT   = "EBF2FA"
HEX_ROW_A   = "F4F8FD"
HEX_ROW_B   = "FFFFFF"
HEX_HEADER  = "E2EDF8"


# ══════════════════════════════════════════════════════════
# XML helpers
# ══════════════════════════════════════════════════════════

def _set_cell_bg(cell, hex_col: str) -> None:
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_col)
    tcPr.append(shd)


def _cell_borders(cell, colour: str = "D0DFF0") -> None:
    tc    = cell._tc
    tcPr  = tc.get_or_add_tcPr()
    tcBdr = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"),   "single")
        b.set(qn("w:sz"),    "4")
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), colour)
        tcBdr.append(b)
    tcPr.append(tcBdr)


def _para_bottom_border(p, colour: str = HEX_ACCENT, size: str = "8") -> None:
    """Add a coloured bottom rule under a paragraph."""
    pPr  = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bot  = OxmlElement("w:bottom")
    bot.set(qn("w:val"),   "single")
    bot.set(qn("w:sz"),    size)
    bot.set(qn("w:space"), "2")
    bot.set(qn("w:color"), colour)
    pBdr.append(bot)
    pPr.append(pBdr)


def _set_para_shade(p, hex_col: str) -> None:
    """Shade the full paragraph background (used for cover band effect)."""
    pPr  = p._p.get_or_add_pPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_col)
    pPr.append(shd)


# ══════════════════════════════════════════════════════════
# Typography helpers
# ══════════════════════════════════════════════════════════

def _h1(doc: Document, text: str) -> None:
    """Section heading — large, dark, with blue bottom rule."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(22)
    p.paragraph_format.space_after  = Pt(8)
    run = p.add_run(text)
    run.bold           = True
    run.font.size      = Pt(14)
    run.font.color.rgb = NAVY
    run.font.name      = "Calibri"
    _para_bottom_border(p, HEX_ACCENT, "10")


def _h2(doc: Document, text: str) -> None:
    """Subsection heading — accent blue."""
    p   = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after  = Pt(5)
    run = p.add_run(text)
    run.bold           = True
    run.font.size      = Pt(11.5)
    run.font.color.rgb = ACCENT
    run.font.name      = "Calibri"


def _body(doc: Document, text: str, size: float = 10.5) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after      = Pt(6)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p.paragraph_format.line_spacing      = Pt(13)
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.name = "Calibri"


def _caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.space_before = Pt(4)
    run = p.add_run(text)
    run.font.size      = Pt(9)
    run.font.italic    = True
    run.font.color.rgb = MUTED
    run.font.name      = "Calibri"


def _warning(doc: Document, text: str) -> None:
    p   = doc.add_paragraph()
    p.paragraph_format.space_after  = Pt(5)
    p.paragraph_format.left_indent  = Cm(0.5)
    run = p.add_run(f"⚠  {text.lstrip('⚠️ℹ️ ').strip()}")
    run.font.size      = Pt(9.5)
    run.font.color.rgb = WARN
    run.font.name      = "Calibri"


def _bullet(doc: Document, text: str, size: float = 10) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.name = "Calibri"


def _numbered(doc: Document, text: str, size: float = 10) -> None:
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.name = "Calibri"


def _spacer(doc: Document, pts: float = 6) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(pts)


def _insert_image(doc: Document, buf: io.BytesIO, width_in: float = 5.8) -> None:
    if buf is None:
        return
    buf.seek(0)
    try:
        doc.add_picture(buf, width=Inches(width_in))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    except Exception:
        pass


# ══════════════════════════════════════════════════════════
# Table helpers
# ══════════════════════════════════════════════════════════

def _simple_table(
    doc: Document,
    headers: list[str],
    rows:    list[list[str]],
    header_bg: str = HEX_ACCENT2,
) -> None:
    """
    Generic styled table with coloured header row and alternating body rows.
    """
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style     = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT

    # Header row
    hcells = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hcells[i].text = h
        _set_cell_bg(hcells[i], header_bg)
        _cell_borders(hcells[i], "3C6FA8")
        hcells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        para = hcells[i].paragraphs[0]
        para.paragraph_format.space_before = Pt(3)
        para.paragraph_format.space_after  = Pt(3)
        if para.runs:
            r = para.runs[0]
        else:
            r = para.add_run(h)
            hcells[i].text = ""
            para.add_run(h)
            r = para.runs[0]
        r.bold            = True
        r.font.color.rgb  = WHITE
        r.font.size       = Pt(9)
        r.font.name       = "Calibri"

    # Data rows
    for j, row_data in enumerate(rows):
        cells = tbl.add_row().cells
        bg    = HEX_ROW_A if j % 2 == 0 else HEX_ROW_B
        for i, val in enumerate(row_data):
            cells[i].text = str(val)
            _set_cell_bg(cells[i], bg)
            _cell_borders(cells[i], "C5D8EE")
            cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            para = cells[i].paragraphs[0]
            para.paragraph_format.space_before = Pt(2)
            para.paragraph_format.space_after  = Pt(2)
            if para.runs:
                para.runs[0].font.size = Pt(9)
                para.runs[0].font.name = "Calibri"

    _spacer(doc, 10)


def _metrics_table_single(doc: Document, m: dict) -> None:
    """Compact 2-column metrics table for a single run."""
    rows = [
        ["R²",   f"{m.get('R2',  0):.4f}"],
        ["RMSE", f"{m.get('RMSE',0):.4f}"],
        ["MAE",  f"{m.get('MAE', 0):.4f}"],
        ["MSE",  f"{m.get('MSE', 0):.6f}"],
    ]
    _simple_table(doc, ["Metric", "Value"], rows, header_bg=HEX_ACCENT)


def _comparison_table(doc: Document, comparison_rows: list[dict]) -> None:
    if not comparison_rows:
        return
    headers = list(comparison_rows[0].keys())
    rows    = [[str(v) for v in r.values()] for r in comparison_rows]
    _simple_table(doc, headers, rows, header_bg=HEX_ACCENT2)


# ══════════════════════════════════════════════════════════
# Cover page
# ══════════════════════════════════════════════════════════

def _build_cover(doc, meta: dict) -> None:
    """
    Standalone cover page — no running header on this page
    (section.different_first_page_header_footer = True handles that).

    Layout:
      ┌──────────────────────────────────────┐
      │                                      │
      │        [logo, 1.5 inch, centred]     │
      │                                      │
      │    MLForge   (38pt, #af0606, bold)   │
      │  Training Report  (20pt, navy, ital) │
      │                                      │
      │  ════════════════════════════════    │  ← thick red rule
      │                                      │
      │  Author:        Chirodeep Karmakar   │
      │  Institution:   …                    │
      │  Date:          01 April 2026        │
      │  Runs included: 3                    │
      │                                      │
      │  ────────────────────────────────    │  ← thin muted footer rule
      │  Generated by MLForge  (8pt, muted)  │
      └──────────────────────────────────────┘
    """
    from docx.shared    import Inches, Pt, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns   import qn
    from docx.oxml      import OxmlElement
    from reports.report_header_helper import LOGO_PATH

    RED   = RGBColor(0xAF, 0x06, 0x06)
    NAVY  = RGBColor(0x1A, 0x1A, 0x2E)
    MUTED = RGBColor(0x9A, 0x9A, 0x9A)
    RED_FADED = RGBColor(0xD4, 0x60, 0x60)

    def _rule(para, colour_hex: str, size: str, space_after: float = 20):
        para.paragraph_format.space_before = Pt(0)
        para.paragraph_format.space_after  = Pt(space_after)
        pPr  = para._p.get_or_add_pPr()
        pBdr = OxmlElement("w:pBdr")
        bot  = OxmlElement("w:bottom")
        bot.set(qn("w:val"),   "single")
        bot.set(qn("w:sz"),    size)
        bot.set(qn("w:space"), "4")
        bot.set(qn("w:color"), colour_hex)
        pBdr.append(bot)
        pPr.append(pBdr)

    def _spacer(pts: float):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after  = Pt(pts)

    def _centred(text: str, size: float, colour: RGBColor,
                 bold: bool = False, italic: bool = False,
                 space_after: float = 6) -> None:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after  = Pt(space_after)
        r = p.add_run(text)
        r.bold = bold; r.font.italic = italic
        r.font.size = Pt(size)
        r.font.color.rgb = colour
        r.font.name = "Calibri"
        return p

    # ── Vertical breathing room at top ────────────────────
    _spacer(36)

    # ── Logo (large, centred) ─────────────────────────────
    if LOGO_PATH.exists():
        logo_p = doc.add_paragraph()
        logo_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        logo_p.paragraph_format.space_before = Pt(0)
        logo_p.paragraph_format.space_after  = Pt(18)
        logo_p.add_run().add_picture(str(LOGO_PATH), width=Inches(1.5))
    else:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(18)
        r = p.add_run("ML")
        r.bold = True; r.font.size = Pt(56)
        r.font.color.rgb = RED; r.font.name = "Calibri"

    # ── MLForge wordmark ──────────────────────────────────
    wm_p = doc.add_paragraph()
    wm_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    wm_p.paragraph_format.space_before = Pt(0)
    wm_p.paragraph_format.space_after  = Pt(6)

    r_ml = wm_p.add_run("ML")
    r_ml.bold = True; r_ml.font.size = Pt(40)
    r_ml.font.color.rgb = RED; r_ml.font.name = "Calibri"

    r_forge = wm_p.add_run("Forge")
    r_forge.bold = True; r_forge.font.size = Pt(40)
    r_forge.font.color.rgb = RED_FADED; r_forge.font.name = "Calibri"

    # ── Report title (user's title string) ────────────────
    _centred(
        meta.get("title", "Training Report"),
        size=20, colour=NAVY, italic=True, space_after=28,
    )

    # ── Thick red rule ────────────────────────────────────
    _rule(doc.add_paragraph(), "AF0606", size="18", space_after=22)

    # ── Metadata rows ─────────────────────────────────────
    def _meta_row(label: str, value: str):
        if not value:
            return
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after  = Pt(8)
        p.paragraph_format.left_indent  = Cm(2.0)

        r_lbl = p.add_run(f"{label}:   ")
        r_lbl.bold = True; r_lbl.font.size = Pt(12)
        r_lbl.font.color.rgb = RED; r_lbl.font.name = "Calibri"

        r_val = p.add_run(value)
        r_val.font.size = Pt(12)
        r_val.font.color.rgb = NAVY; r_val.font.name = "Calibri"

    _meta_row("Author",       meta.get("author", ""))
    _meta_row("Institution",  meta.get("institution", ""))
    _meta_row("Date",         meta.get("date", ""))
    n_runs = meta.get("n_runs")
    if n_runs:
        _meta_row("Runs included", str(n_runs))

    # ── Bottom spacer + thin rule + generated-by note ─────
    _spacer(24)
    _rule(doc.add_paragraph(), "CCCCCC", size="4", space_after=6)

    footer_p = doc.add_paragraph()
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_p.paragraph_format.space_before = Pt(0)
    footer_p.paragraph_format.space_after  = Pt(0)
    r_ft = footer_p.add_run("Generated by MLForge  ·  Universal Supervised Regression Training Platform")
    r_ft.font.size = Pt(8); r_ft.font.color.rgb = MUTED
    r_ft.font.name = "Calibri"; r_ft.font.italic = True

    doc.add_page_break()


# ══════════════════════════════════════════════════════════
# Main builder
# ══════════════════════════════════════════════════════════

def build_docx(
    runs:            list[dict],
    report_cfg:      dict,
    dataset_profile: dict,
    ai_sections:     dict | None = None,
) -> tuple[bytes, str]:
    """
    Build a styled Word report and return (bytes, filename).

    Parameters
    ----------
    runs            : saved run dicts from session_store
    report_cfg      : {title, author, institution, date, notes, include_plots}
    dataset_profile : output of dataset_profiler.profile_dataset()
    ai_sections     : {conclusions, research_gaps, future_work} — pre-edited
                      strings from Page 6. If None, fallback text is used.
    """
    payload = build_report_payload(runs, report_cfg, dataset_profile, ai_sections)
    meta    = {
        **payload["meta"],
        "n_runs": payload["n_runs"],   # passes run count to cover page
    }
    profile  = payload["profile"]
    doc = Document()
    
    # ── Page setup ────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Cm(2.2)
        section.bottom_margin = Cm(2.2)
        section.left_margin   = Cm(2.8)
        section.right_margin  = Cm(2.4)
    
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    
    # ── Per-page header (logo + wordmark + red rule + page number) ──
    add_report_header_to_doc(doc)
    
    # ── Cover ─────────────────────────────────────────
    _build_cover(doc, meta)

    # ── Section counter ───────────────────────────────────
    sec = [1]   # mutable int so nested helpers can increment

    def next_sec(title: str) -> None:
        _h1(doc, f"{sec[0]}. {title}")
        sec[0] += 1

    # ══════════════════════════════════════════════════════
    # 1. INTRODUCTION
    # ══════════════════════════════════════════════════════
    next_sec("Introduction")

    used_models = ", ".join(
        {r.get("model_label", r.get("model_key", "")) for r in runs}
    )
    _body(doc,
        "This report was generated by MLForge — a supervised regression training "
        "platform covering SVR (linear, polynomial, RBF kernels), ensemble methods, "
        "boosting models, linear regressors, neural networks, and probabilistic models. "
        "It is designed to replicate and compare ML algorithms reported in the "
        "hydrogen production and biofuel ML literature."
    )
    _body(doc, f"Models used in this report: {used_models}.")

    if meta.get("notes"):
        _spacer(doc, 4)
        _body(doc, meta["notes"])

    # ══════════════════════════════════════════════════════
    # 2. DATASET SUMMARY
    # ══════════════════════════════════════════════════════
    next_sec("Dataset Summary")

    if profile:
        ds_name = runs[0].get("dataset_name", "Unknown") if runs else "Unknown"
        target  = runs[0].get("target_col",   "—")       if runs else "—"

        _simple_table(doc,
            ["Property", "Value"],
            [
                ["Dataset",              ds_name],
                ["Rows",                 f"{profile.get('n_rows', '—'):,}"],
                ["Columns",              str(profile.get('n_cols', '—'))],
                ["Numeric features",     str(profile.get('n_features', '—'))],
                ["Missing values",       f"{profile.get('missing_pct', 0):.1f}%"],
                ["Time-series dataset",  "Yes" if profile.get("is_time_series") else "No"],
                ["Target column",        target],
            ],
            header_bg=HEX_ACCENT2,
        )

        for w in profile.get("warnings", []):
            _warning(doc, w)

    else:
        _body(doc, "No dataset profile available.")

    # ══════════════════════════════════════════════════════
    # 3. MODEL RESULTS
    # ══════════════════════════════════════════════════════
    next_sec("Model Results")

    for run in payload["runs"]:
        _h2(doc, f"Run #{run['run_id']} — {run['model_label']}")

        m  = run.get("metrics", {})
        an = run.get("analysis", {})

        _metrics_table_single(doc, m)

        # Interpretation
        if an.get("summary_sentence"):
            _body(doc, an["summary_sentence"])
        if an.get("mae_context"):
            _caption(doc, an["mae_context"])
        if an.get("rmse_context"):
            _caption(doc, an["rmse_context"])

        # Recommendations
        recs = an.get("recommendations", [])
        if recs:
            _spacer(doc, 4)
            p = doc.add_paragraph()
            r = p.add_run("Recommendations:")
            r.bold      = True
            r.font.size = Pt(10)
            r.font.name = "Calibri"
            for rec in recs:
                _bullet(doc, rec)

        # Auto-tuned params
        bp = run.get("best_params")
        if bp:
            _spacer(doc, 4)
            _caption(doc, f"Auto-tuned hyperparameters: {bp}")

        # Plots
        plots = run.get("plot_bufs", {})
        if plots.get("actual_vs_predicted"):
            _spacer(doc, 6)
            _caption(doc, "Figure: Actual vs Predicted values on test set")
            _insert_image(doc, plots["actual_vs_predicted"])

        if plots.get("residuals"):
            _spacer(doc, 6)
            _caption(doc, "Figure: Residual distribution")
            _insert_image(doc, plots["residuals"])

        _spacer(doc, 8)

    # ══════════════════════════════════════════════════════
    # 4. RUNS COMPARISON  (only if > 1 run)
    # ══════════════════════════════════════════════════════
    if len(payload["runs"]) > 1:
        next_sec("Runs Comparison")
        _comparison_table(doc, payload["comparison_rows"])

        if payload.get("comp_plot_buf"):
            _caption(doc, "Figure: R² comparison across all runs")
            _insert_image(doc, payload["comp_plot_buf"])

    # ══════════════════════════════════════════════════════
    # AI SECTIONS  (Conclusions / Research Gaps / Future Work)
    # ══════════════════════════════════════════════════════
    ai = payload.get("ai_sections", {})

    next_sec("Conclusions")
    _body(doc, ai.get("conclusions", ""))

    next_sec("Research Gaps")
    _body(doc, ai.get("research_gaps", ""))

    next_sec("Future Work")
    _body(doc, ai.get("future_work", ""))

    # ══════════════════════════════════════════════════════
    # REFERENCES  (dynamic — only what was used)
    # ══════════════════════════════════════════════════════
    next_sec("References")

    for i, (authors, text) in enumerate(payload["references"], 1):
        p   = doc.add_paragraph(style="List Number")
        p.paragraph_format.space_after = Pt(5)
        r1  = p.add_run(f"{authors} — ")
        r1.bold      = True
        r1.font.size = Pt(10)
        r1.font.name = "Calibri"
        r2  = p.add_run(text)
        r2.font.size = Pt(10)
        r2.font.name = "Calibri"

    # ── Serialise ──────────────────────────────────────────
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    safe  = meta.get("title", "MLForge_Report").replace(" ", "_")
    fname = f"{safe}_{meta.get('date','report').replace(' ','_')}.docx"

    return buf.getvalue(), fname
