"""Shared UI helpers for the pyTEM Streamlit app.

Edit the constants below in one place; they appear in the footer of every page.
"""

import datetime

import streamlit as st

# --- Edit these once; they propagate to every page footer --------------------
APP_AUTHORS = "the PyTEM authors"
APP_LICENSE = "see repository LICENSE"   # TODO: set once a license is chosen
APP_REPO_URL = "https://github.com/TODO/pyTEM"


def render_footer():
    """Render a consistent license / author footer at the bottom of a page."""
    st.divider()
    st.caption(
        f"**PyTEM**   ·  "
        f"© {datetime.date.today().year} {APP_AUTHORS}  ·  "
        f"License: {APP_LICENSE}  ·  "
        f"[Source]({APP_REPO_URL})  \n"
        "For teaching and research purposes. Synthetic examples are illustrative, not field surveys."
    )
