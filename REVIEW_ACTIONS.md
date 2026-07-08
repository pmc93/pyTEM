# Review Response Tracker: Streamlit App

Source: `2026 07 01 Review_Geophysics.docx` (Thomas Reimann, 2026-07-01)

This document lists each recommended change from the review and tracks what has been implemented in the Streamlit app. Update the **Status** and **What was done** columns as work progresses.

Status legend: `TODO` (not started), `WIP` (in progress), `DONE` (implemented), `WONTFIX` (decided against, with reason).

---

## Section: TEM vs. VES

| # | Recommendation | Status | What was done |
|---|----------------|--------|---------------|
| 1 | Add a sketch or conceptual figure near the beginning to introduce the methods visually. | WIP | Labelled figure placeholder added near the top of the landing page (`[FIGURE PLACEHOLDER · Review #1]`) describing the intended TEM+VES overview sketch; artwork still to be drawn. |
| 2 | TEM "Data" section is hard to follow; explain the meaning of `dB/dt` with a short explanation or illustration. | WIP | Placeholder added under the TEM card (`Review #2`) describing a current-step / dB/dt decay illustration; artwork still to be drawn. |
| 3 | Add a simple conceptual sketch of the measurement principle for both TEM and VES. | WIP | Placeholders added under both method cards (`Review #3 (TEM)` and `Review #3 (VES)`) describing the measurement-principle sketches; artwork still to be drawn. |
| 4 | VES "Data" section is hard to interpret; add explanations or a simple figure. | WIP | Placeholder added under the VES card (`Review #4`) describing an annotated apparent-resistivity curve; artwork still to be drawn. |
| 5 | Reconsider the order/presentation of the first comparison table and the accompanying bullet points. | TODO | |

## Section: User Interface (TEM vs. VES)

| # | Recommendation | Status | What was done |
|---|----------------|--------|---------------|
| 6 | In the "Check your intuition..." section, add a small sketch illustrating the situation. | WIP | Placeholder added above the quiz (`Review #6`) describing a layered-earth cartoon with a buried conductor over/under a resistive basement; artwork still to be drawn. |
| 7 | (Positive) "I like the balloons :)" — keep the celebratory balloons. | DONE | Existing behavior retained. |

## Section: Forward Modeling

| # | Recommendation | Status | What was done |
|---|----------------|--------|---------------|
| 8 | Clarify terminology: "layered resistivity model" vs. "forward modeling". Make clear the sliders change the model, not the measurements. | DONE | Rewrote the Forward Modeling intro to state the sliders change the *earth model* (right) while the left panel is the *simulated measurement*, not something edited directly. |
| 9 | Explain the system parameters; use Streamlit `info`/help tooltips on the widgets for short explanations. | DONE | Added `help=` tooltips to all TEM system parameters (Tx loop side, time gates, early/late time) and VES survey parameters (AB/2 min/max, points, number of layers). |
| 10 | Clarify that the third layer is the half-space (only two sets of parameters are editable). Note whether the half-space vertical extent is fixed or adjustable. | DONE | Half-space row now labelled "Half-space (infinite depth)" with a caption noting it has no thickness and extends downward forever; "Number of layers" help repeats this. |
| 11 | (Positive) "Try it yourself" section is excellent. | DONE | Existing section retained. |

## Section: User Interface (Forward Modeling)

| # | Recommendation | Status | What was done |
|---|----------------|--------|---------------|
| 12 | "Compute forward model" button may be unnecessary; consider auto-updating plots. | DONE | Removed both compute buttons (plots already recompute on every slider change) and added a caption "The plot updates automatically when you move a slider." |
| 13 | Streamlit tabs introduce high computational load. Replace with the lighter workaround used in the gw-inux Theis Derivatives module. Reviewer offered to help. Reference: https://github.com/gw-inux/Jupyter-Notebooks/blob/main/90_Streamlit_apps/GWP_Pumping_Test_Derivatives/content/01_Theis_Deriv_Ini.py | TODO | |
| 14 | When switching between TEM and VES, keep the layered structure unchanged to ease comparison. | TODO | |
| 15 | Organize thickness and resistivity sliders side by side using columns to save space and ease comparison. | DONE | `_model_ui` now renders thickness and resistivity in two `st.columns` per layer. |

## Section: Jacobian Sensitivity

| # | Recommendation | Status | What was done |
|---|----------------|--------|---------------|
| 16 | (Positive) Section looks very clean. | DONE | No change needed. |
| 17 | Add a "Try it yourself" section similar to the one in Forward Modeling. | TODO | |

## Section: Inversion

| # | Recommendation | Status | What was done |
|---|----------------|--------|---------------|
| 18 | Confirm whether the TEM/VES plots in Section 2 ("Corrupt...") and Section 3 ("Run the...") show the same data. If same, use identical legend names and axis ranges (VES y-axis currently differs). If different, distinguish with different colors/line styles. | TODO | |

## Section: Field Example

| # | Recommendation | Status | What was done |
|---|----------------|--------|---------------|
| 19 | Display the thicknesses of the geological units from the example, or clarify whether this is intentionally hidden/unavailable. | TODO | |
| 20 | In the VES plot, "Schlumberger" appears for the first time. Add a brief note clarifying it is the VES configuration introduced earlier. | TODO | |
| 21 | Section 2 ("Invert...") has a max model depth slider. Consider offering the same option in earlier sections. | TODO | |
| 22 | Add an optional toggle to display a simplified interpretation plot (showing the interpreted layers) alongside the explanation. | TODO | |

## Minor Suggestions: Script / UI Architecture

| # | Recommendation | Status | What was done |
|---|----------------|--------|---------------|
| 23 | Consider a flexible multipage app structure (reviewer can provide a template). Avoids emojis in file names, which can cause GitHub issues. | TODO | |
| 24 | Separate the markdown text from the code and include it via placeholders. Eases independent editing of educational content and future translations. Reviewer offered to help. | TODO | |

---

## Follow-up offered by reviewer

- Reviewer offered help implementing the tabs workaround (#13), the multipage structure (#23), and the markdown/code separation (#24).
- Reviewer offered a short Zoom meeting to discuss any suggestions.
