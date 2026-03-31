# config/page_header.py
# ─────────────────────────────────────────────────────────────
# Renders the consistent MLForge header used on every page.
# Uses a single st.markdown() flexbox block — bypasses Streamlit
# column width constraints entirely so sizing is pure CSS.
#
# Usage (top of any page, after apply_page_config + CSS load):
#
#   from config.page_header import render_header
#   render_header()
# ─────────────────────────────────────────────────────────────

import base64
import streamlit as st
from pathlib import Path


def _logo_img_tag(width: int = 96) -> str:
    """
    Return an <img> tag with the logo base64-embedded, or an empty string
    if assets/logo.png doesn't exist (fallback badge is used instead).
    Embedding as base64 means the image is not subject to Streamlit's
    column-width capping.
    """
    logo_path = Path("assets/logo.png")
    if not logo_path.exists():
        return ""
    data = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    return (
        f'<img src="data:image/png;base64,{data}" '
        f'width="{width}" height="{width}" '
        f'style="border-radius:14px; object-fit:contain; display:block;" />'
    )


def render_header(logo_size: int = 96) -> None:
    """
    Render the MLForge logo + wordmark header as a single flexbox HTML block.
    No st.columns() — sizing is governed entirely by CSS, not Streamlit layout.

    Parameters
    ----------
    logo_size : pixel width/height of the logo image (default 96).
                Increase to 128 for even larger display.
    """
    img_tag = _logo_img_tag(logo_size)

    # Fallback badge when no logo.png is present
    if img_tag:
        logo_html = img_tag
    else:
        logo_html = f"""
        <div style="
            width:{logo_size}px; height:{logo_size}px;
            background: linear-gradient(135deg, #4f86c6, #2e5f9e);
            border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-size: {logo_size * 0.32:.0f}px; font-weight: 800;
            color: white; letter-spacing: -0.04em;
            flex-shrink: 0;
        ">ML</div>"""

    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 0.5rem 0 0.25rem;
        ">
            {logo_html}
            <div>
                <div style="
                    font-size: 2.6rem;
                    font-weight: 700;
                    letter-spacing: -0.04em;
                    line-height: 1.1;
                    color: #1a1a2e;
                ">ML<span style="opacity:0.35;">Forge</span></div>
                <div style="
                    font-size: 0.95rem;
                    opacity: 0.5;
                    margin-top: 3px;
                    font-weight: 300;
                    letter-spacing: 0.02em;
                ">Universal supervised regression training platform</div>
            </div>
        </div>
        <hr style="margin-top:1rem; margin-bottom:0; border:none;
                   border-top:1px solid rgba(0,0,0,0.1);" />
        """,
        unsafe_allow_html=True,
    )

    # Small spacer after the HR so content doesn't crowd the divider
    st.markdown("<div style='margin-bottom:0.75rem;'></div>",
                unsafe_allow_html=True)
