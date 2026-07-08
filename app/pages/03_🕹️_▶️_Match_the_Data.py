import os
import sys

import numpy as np
import matplotlib.pyplot as plt
# -- Matplotlib font sizes (mobile-friendly) --------------------------
plt.rcParams.update({
    "font.size":       16,
    "axes.labelsize":  18,
    "axes.titlesize":  18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from pytem import fwd_circle_central
from ves import forward as ves_forward
from _shared import render_footer

# ── Shared utilities ──────────────────────────────────────────────────────────
_RHO = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
_VES_FILTER = "gs11"


def _stair(thick, rho, bottom=None, extra=None):
    depths = [0.0] + list(np.cumsum(thick))
    if bottom is None:
        bottom = depths[-1] + (extra if extra is not None else max(depths[-1] * 0.3, 20.0))
    r_s, d_s = [], []
    for i, r in enumerate(rho):
        d_top = depths[i]
        d_bot = depths[i + 1] if i < len(thick) else bottom
        r_s += [r, r]
        d_s += [d_top, d_bot]
    return r_s, d_s


def _nearest_rho(value):
    return min(_RHO, key=lambda x: abs(x - value))


# ── Cached forward calls ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _tem_fwd(h_t, rho_t, r, times_t):
    return -fwd_circle_central(list(h_t), list(rho_t),
                               tx_radius=r, times=np.array(times_t))


@st.cache_data(show_spinner=False)
def _ves_fwd(ab2_t, rho_t, h_t, filt):
    return ves_forward(np.array(ab2_t), np.array(rho_t), np.array(h_t), filt)


# ── Target-data generators (hidden truth, no added noise) ─────────────────
@st.cache_data(show_spinner=False)
def _tem_target(thick_t, rho_t, tx_r, times_t, b_coeff):
    times = np.array(times_t)
    clean = _tem_fwd(tuple(thick_t), tuple(rho_t), tx_r, times_t)
    sigma = b_coeff * times ** (-0.5)
    obs = clean
    return obs, sigma


@st.cache_data(show_spinner=False)
def _ves_target(thick_t, rho_t, ab2_t, frac):
    clean = _ves_fwd(ab2_t, tuple(rho_t), tuple(thick_t), _VES_FILTER)
    obs = clean
    sigma = frac * obs
    return obs, sigma


def _rms(pred, obs, sigma):
    """Noise-normalised RMS misfit (≈1 means the model fits to the noise level)."""
    return float(np.sqrt(np.mean(((pred - obs) / sigma) ** 2)))


def _fit_feedback(rms):
    if rms < 1.5:
        st.success(f"Excellent fit. Your curve threads the data at the noise level "
                   f"(normalised RMS = {rms:.2f}).")
    elif rms < 3.0:
        st.info(f"Good fit, but not quite at the noise level yet "
                f"(normalised RMS = {rms:.2f}). Fine-tune the sliders.")
    elif rms < 8.0:
        st.warning(f"Getting closer (normalised RMS = {rms:.2f}). Which part of the "
                   f"curve still misses the data?")
    else:
        st.error(f"Still far off (normalised RMS = {rms:.2f}). Try changing one "
                 f"parameter at a time and watch which way the curve moves.")


# ── Page header ───────────────────────────────────────────────────────────────
st.header(":green[Match the data by hand]")
st.markdown(
    "On the *Forward Modeling* page you built a layered earth and computed the data it "
    "would produce. In the field we have the opposite problem: we measure the data "
    "first, and we want to recover the earth model that created it. That reverse step "
    "is called **inversion**."
)
st.markdown(
    "The best way to understand inversion is to do it yourself. Here **you** are the "
    "inversion. You are handed field-like data (a set of measured points) from a "
    "**hidden two-layer earth** (a layer over a half-space). Adjust the three "
    "model parameters, the two layer resistivities and the layer thickness, until "
    "your predicted curve passes through the measurements. The **misfit** below "
    "scores how close you are."
)
st.markdown(
    "As you work, notice *how* you decide which way to turn each slider: you change a "
    "parameter, see which way the curve moves, and step towards a better fit. A "
    "computer needs that same information in a precise form so it can update the model "
    "on its own. The **next page** measures exactly this, how sensitive the data are to "
    "each parameter, which is what guides an automated **inversion**. This exercise "
    "also reveals why several different models can fit the same data (**non-uniqueness**)."
)

with st.expander("What is the misfit?", expanded=False):
    st.markdown(
        r"""
        Each data point has an error bar $\sigma_i$. The **normalised RMS misfit**
        measures, on average, how many error bars your model misses each point by:

        $$
        \mathrm{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}
        \left(\frac{d_i^{\text{pred}} - d_i^{\text{obs}}}{\sigma_i}\right)^2 }
        $$

        - **RMS ≈ 1** means your curve fits the data about as well as the noise
          allows. You cannot meaningfully do better; chasing the wiggles below this
          is just fitting noise.
        - **RMS ≫ 1** means the model is missing real structure in the data.
        """
    )

tab_tem, tab_ves = st.tabs(["🧲 TEM", "⚡️ VES"])

# ═══════════════════════════════════════════════════════════════════════════════
# TEM TAB  (hidden truth: conductive basement, which TEM sees well)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tem:
    st.subheader(":blue-background[TEM - match the decay curve]", divider="blue")

    tx_side = 40.0
    tx_r = float(np.sqrt(tx_side ** 2 / np.pi))
    times = np.logspace(-5.0, -2.0, 25)
    _b = 3e-11  # TEM noise-floor coefficient

    # Hidden true model (a layer over a half-space)
    _TEM_TRUE = dict(h1=30.0, rho1=10.0, rho2=100.0)
    obs_t, sig_t = _tem_target((_TEM_TRUE["h1"],),
                               (_TEM_TRUE["rho1"], _TEM_TRUE["rho2"]),
                               tx_r, tuple(times.tolist()), _b)

    st.info(
        "**Hint:** work from shallow to deep. The **early-time** signal is set by the "
        "top layer; the **late-time tail** by the half-space. Fit the top layer "
        "first, then adjust the half-space. Watch for the conductive layer, which "
        "holds the TEM curve up for longer."
    )

    st.markdown("**Your model** (a layer over a half-space)")
    col_r, col_m = st.columns(2)
    with col_r:
        rho1 = float(st.select_slider("Resistivity of layer 1 [Ohm.m]", _RHO, value=50,
                                      key="match_tem_r1",
                                      help="Resistivity of the top layer."))
        rho2 = float(st.select_slider("Resistivity of half-space [Ohm.m]", _RHO, value=200,
                                      key="match_tem_r2",
                                      help="Resistivity of the basement, which extends "
                                           "to infinite depth."))
    with col_m:
        h1 = float(st.slider("Thickness of layer 1 [m]", 1, 100, 20,
                             key="match_tem_h1",
                             help="How thick the top layer is."))

    st.caption("The plot and misfit update automatically as you move a slider.")

    try:
        pred_t = _tem_fwd((h1,), (rho1, rho2), tx_r, tuple(times.tolist()))
        rms_t = _rms(pred_t, obs_t, sig_t)

        st.metric("Normalised RMS misfit", f"{rms_t:.2f}", help="Aim for about 1.0")
        _fit_feedback(rms_t)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
        ax1.loglog(times, obs_t, "o", color="black", ms=5,
                   label="Data", zorder=3)
        ax1.loglog(times, pred_t, "-", color="steelblue", lw=2,
                   label="Your model", zorder=4)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel(r"|dB/dt| [V/m$^2$]")
        ax1.grid(True, which="both", ls="--", alpha=0.6)
        ax1.legend()

        rs, ds = _stair([h1], [rho1, rho2])
        ax2.semilogx(rs, ds, color="steelblue", lw=2, label="Your model")
        ax2.invert_yaxis()
        ax2.set_xlabel(r"Resistivity [Ohm.m]")
        ax2.set_ylabel("Depth [m]")
        ax2.grid(True, which="both", ls="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Reveal the hidden true model"):
            st.markdown(
                f"""
                The data were generated from:

                - **Layer 1:** rho = {_TEM_TRUE['rho1']:.0f} Ohm.m, thickness = {_TEM_TRUE['h1']:.0f} m
                - **Half-space:** rho = {_TEM_TRUE['rho2']:.0f} Ohm.m

                Notice how strongly the curve responds to the **conductive top layer**:
                TEM is very sensitive to conductors, so that layer is well constrained.
                The resistive half-space is harder to pin down on its own.
                """
            )
    except Exception as _e:
        st.warning(f"⚠️ Could not compute: {_e}. Try adjusting the sliders.")

# ═══════════════════════════════════════════════════════════════════════════════
# VES TAB  (hidden truth: resistive basement, which VES sees well)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ves:
    st.subheader(":orange-background[VES - match the apparent-resistivity curve]",
                 divider="orange")

    ab2 = np.logspace(0, np.log10(300), 25)
    _frac = 0.05  # 5% relative error

    _VES_TRUE = dict(h1=20.0, rho1=300.0, rho2=50.0)
    obs_v, sig_v = _ves_target((_VES_TRUE["h1"],),
                               (_VES_TRUE["rho1"], _VES_TRUE["rho2"]),
                               tuple(ab2.tolist()), _frac)

    st.info(
        "**Hint:** work from shallow to deep. **Small AB/2** (left) is set by the top "
        "layer; **large AB/2** (right) by the half-space. Fit the left of the curve "
        "first, then adjust the half-space. Watch for the resistive layer, which lifts "
        "the apparent-resistivity curve."
    )

    st.markdown("**Your model** (a layer over a half-space)")
    col_r, col_m = st.columns(2)
    with col_r:
        rho1 = float(st.select_slider("Resistivity of layer 1 [Ohm.m]", _RHO, value=100,
                                      key="match_ves_r1",
                                      help="Resistivity of the top layer."))
        rho2 = float(st.select_slider("Resistivity of half-space [Ohm.m]", _RHO, value=100,
                                      key="match_ves_r2",
                                      help="Resistivity of the basement, which extends "
                                           "to infinite depth."))
    with col_m:
        h1 = float(st.slider("Thickness of layer 1 [m]", 1, 100, 30,
                             key="match_ves_h1",
                             help="How thick the top layer is."))

    st.caption("The plot and misfit update automatically as you move a slider.")

    try:
        pred_v = _ves_fwd(tuple(ab2.tolist()), (rho1, rho2), (h1,), _VES_FILTER)
        rms_v = _rms(pred_v, obs_v, sig_v)

        st.metric("Normalised RMS misfit", f"{rms_v:.2f}", help="Aim for about 1.0")
        _fit_feedback(rms_v)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
        ax1.loglog(ab2, obs_v, "o", color="black", ms=5,
                   label="Data", zorder=3)
        ax1.loglog(ab2, pred_v, "-", color="darkorange", lw=2,
                   label="Your model", zorder=4)
        ax1.set_xlabel("AB/2 [m]")
        ax1.set_ylabel("Apparent resistivity [Ohm.m]")
        ax1.grid(True, which="both", ls="--", alpha=0.6)
        ax1.legend()

        rs, ds = _stair([h1], [rho1, rho2])
        ax2.semilogx(rs, ds, color="darkorange", lw=2, label="Your model")
        ax2.invert_yaxis()
        ax2.set_xlabel(r"Resistivity [Ohm.m]")
        ax2.set_ylabel("Depth [m]")
        ax2.grid(True, which="both", ls="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Reveal the hidden true model"):
            st.markdown(
                f"""
                The data were generated from:

                - **Layer 1:** rho = {_VES_TRUE['rho1']:.0f} Ohm.m, thickness = {_VES_TRUE['h1']:.0f} m
                - **Half-space:** rho = {_VES_TRUE['rho2']:.0f} Ohm.m

                VES forces current through the **resistive top layer**, so the rising
                part of the curve constrains it well. The complementary strength to TEM.
                """
            )
    except Exception as _e:
        st.warning(f"⚠️ Could not compute: {_e}. Try adjusting the sliders.")

# ═══════════════════════════════════════════════════════════════════════════════# The real earth is more complex
# ═════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader(":violet[The real earth is more complicated]", divider="violet")
st.markdown(
    """
    Matching even a **two-layer** model by hand is already fiddly, and you probably
    noticed that several settings fit almost equally well. Now consider the catch: the
    real subsurface is **not** a tidy stack of a few uniform layers.

    A handful of sliders cannot represent that, and turning every knob by hand quickly
    becomes impossible once there are many layers. That is why real interpretation uses
    an **automated inversion** (the next page): the computer represents the ground with
    *many thin layers* and adjusts them all at once to fit the data, while a
    **regularisation** (smoothing) term keeps the result geologically reasonable.

    But automation does not remove the deeper problem. Because the data constrain only
    certain **combinations** of parameters, many different models still fit the same
    measurements, this is **non-uniqueness**, discussed next. The goal is therefore not
    a single "true" model, but the range of models the data allow, ideally narrowed by
    combining complementary methods such as TEM and VES.
    """
)

# ═════════════════════════════════════════════════════════════════════════# Non-uniqueness discussion
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader(":violet[Why more than one model fits: non-uniqueness]", divider="violet")
st.markdown(
    r"""
    You may have found **several different models** that give an almost equally low
    misfit. That is not a failure of effort: it is a fundamental property of the
    physics. The data simply do not contain enough information to separate every
    parameter, so a whole family of models fits within the error bars. Geophysicists
    call this **non-uniqueness** (or *equivalence*).

    Two classic equivalence rules for a layer over a half-space:

    - **Conductive layer (TEM-like):** only the layer **conductance**
      $S = h / \rho$ is well resolved. Halve the thickness *and* halve the
      resistivity and the curve barely moves, because $S$ is unchanged.
    - **Resistive layer (VES-like):** only the **transverse resistance**
      $T = h \cdot \rho$ is well resolved. Double the thickness and halve the
      resistivity and the curve is almost identical.

    **Try it on the TEM tab:** the conductive top layer has `rho1 = 10`, `h1 = 30`
    ($S = h/\rho = 3$). Now set `rho1 = 5`, `h1 = 15` (same $S$). The misfit stays low
    even though the model looks different: the thickness and resistivity of a conductor
    trade off, and only their ratio (the conductance) is well constrained.

    **The practical lesson:** a single sounding constrains certain *combinations* of
    parameters far better than the parameters themselves. This is exactly why we
    (1) quote uncertainties, not just a best model, (2) add prior information or
    regularisation, and (3) combine complementary methods, since TEM pins down the
    conductor and VES pins down the resistor.
    """
)

# ═══════════════════════════════════════════════════════════════════════════════
# Questions
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader(":violet[Check your understanding]", divider="violet")

_MATCH_QUIZ = [
    {
        "q": "You reach a normalised RMS misfit of about 1.0 and keep tweaking the "
             "sliders to push it lower. What are you most likely doing?",
        "options": [
            "Recovering the true earth ever more precisely",
            "Fitting the measurement noise rather than real structure",
            "Reducing the error bars on the data",
        ],
        "answer": "Fitting the measurement noise rather than real structure",
        "why": "RMS ≈ 1 already means the model matches the data to the noise level. "
               "Going below that fits the random scatter, not the ground.",
    },
    {
        "q": "On the TEM tab you find that halving both rho1 and the top-layer "
             "thickness barely changes the misfit. This is because TEM mainly "
             "resolves:",
        "options": [
            "The layer conductance S = h / rho, not h and rho separately",
            "The transverse resistance T = h × rho",
            "The absolute value of rho1 on its own",
        ],
        "answer": "The layer conductance S = h / rho, not h and rho separately",
        "why": "For a conductive layer the induced currents depend on the conductance "
               "h/rho, so thickness and resistivity trade off and only their ratio is "
               "well constrained.",
    },
    {
        "q": "Two quite different models fit the same sounding within its error bars. "
             "The best way to describe the result of the survey is:",
        "options": [
            "A single exact model",
            "A family of models consistent with the data, i.e. with uncertainty",
            "That the data are wrong",
        ],
        "answer": "A family of models consistent with the data, i.e. with uncertainty",
        "why": "Non-uniqueness means the data support a range of models. Honest "
               "interpretation reports that range rather than one exact answer.",
    },
    {
        "q": "Which single change would most reduce the ambiguity between the top "
             "layer and the half-space?",
        "options": [
            "Collecting the same sounding again with the same method",
            "Adding a complementary measurement (e.g. combine TEM with VES)",
            "Using finer slider steps",
        ],
        "answer": "Adding a complementary measurement (e.g. combine TEM with VES)",
        "why": "TEM and VES are sensitive to different things (conductors vs. "
               "resistors), so jointly they constrain parameters that neither pins "
               "down alone. Repeating an identical survey adds little new information.",
    },
]

_match_user = [
    st.radio(_item["q"], _item["options"], index=None, key=f"match_quiz_{_i}")
    for _i, _item in enumerate(_MATCH_QUIZ)
]

if st.button("Check my answers", key="match_quiz_check"):
    _score = 0
    for _i, _item in enumerate(_MATCH_QUIZ):
        if _match_user[_i] == _item["answer"]:
            _score += 1
            st.success(f"Q{_i + 1}: Correct. {_item['why']}")
        elif _match_user[_i] is None:
            st.warning(
                f"Q{_i + 1}: Not answered. Correct answer: "
                f"**{_item['answer']}**. {_item['why']}"
            )
        else:
            st.error(
                f"Q{_i + 1}: Not quite. Correct answer: "
                f"**{_item['answer']}**. {_item['why']}"
            )
    st.markdown(f"**Score: {_score} / {len(_MATCH_QUIZ)}**")

render_footer()
