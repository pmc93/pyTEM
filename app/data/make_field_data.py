"""
Generate a realistic 'field' TEM sounding for the West-Africa regolith example.

The sounding mimics a central-loop TEM measurement over a hard-rock (crystalline
basement) weathering profile typical of West Africa:

    Layer 1  - lateritic / ferricrete cap      resistive   (~800 Ohm.m,  6 m)
    Layer 2  - clay-rich saprolite regolith    conductive  (~40 Ohm.m,  30 m)  <-- aquifer
    Layer 3  - fresh crystalline basement      very resistive (~3000 Ohm.m)

Realistic measurement noise is added (relative error + a late-time noise floor),
so the resulting CSV behaves like genuine field data when loaded and inverted.

Run once with:
    <python> make_field_data.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pytem import fwd_circle_central
from ves import forward as ves_forward

# ── True (undisclosed) earth model ──────────────────────────────────────────
true_thick = [6.0, 30.0]            # m
true_rho = [800.0, 40.0, 3000.0]    # Ohm.m

# ── Acquisition geometry ────────────────────────────────────────────────────
tx_side = 40.0                       # 40 x 40 m transmitter loop
tx_radius = float(np.sqrt(tx_side ** 2 / np.pi))   # equal-area circle
times = np.logspace(np.log10(2e-5), np.log10(8e-3), 26)   # 26 gate centres [s]

# ── Forward response (positive dB/dt convention used by the app) ────────────
dbdt_clean = -fwd_circle_central(true_thick, true_rho,
                                 tx_radius=tx_radius, times=times)

# ── Realistic noise: 4 % relative + power-law late-time floor ───────────────
rng = np.random.default_rng(20260526)   # fixed seed -> reproducible "field" data
rel_err = 0.04
noise_floor = 1e-11 * (times / 1e-3) ** (-0.5)      # V/m^2
sigma = np.sqrt((rel_err * np.abs(dbdt_clean)) ** 2 + noise_floor ** 2)
dbdt_obs = dbdt_clean + rng.normal(size=dbdt_clean.shape) * sigma

# ── Write CSV ───────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "west_africa_regolith.csv")
header = (
    "# Synthetic 'field' TEM sounding - West Africa hard-rock regolith example\n"
    "# Central-loop TEM, 40 x 40 m transmitter loop\n"
    "# Columns: time_s, dbdt_Vm2 (positive), uncertainty_Vm2\n"
)
with open(out, "w", encoding="utf-8") as f:
    f.write(header)
    f.write("time_s,dbdt_Vm2,uncertainty_Vm2\n")
    for t, d, s in zip(times, dbdt_obs, sigma):
        f.write(f"{t:.6e},{d:.6e},{s:.6e}\n")

print(f"Wrote {len(times)} gates to {out}")
print(f"tx_radius = {tx_radius:.3f} m  (from {tx_side} m square loop)")

# ── Matching VES (Schlumberger) sounding over the SAME true model ───────────
_VES_FILTER = "gs11"
ab2 = np.logspace(np.log10(1.5), np.log10(400.0), 24)      # AB/2 spacings [m]
rhoa_clean = ves_forward(ab2, true_rho, true_thick, _VES_FILTER)

ves_rel_err = 0.05                                          # 5 % log-normal noise
rhoa_sigma = ves_rel_err * rhoa_clean
rhoa_obs = rhoa_clean * np.exp(rng.normal(0.0, ves_rel_err, size=ab2.shape))

out_ves = os.path.join(os.path.dirname(__file__), "west_africa_regolith_ves.csv")
header_ves = (
    "# Synthetic 'field' VES sounding - West Africa hard-rock regolith example\n"
    "# Schlumberger array, same site/true model as the TEM sounding\n"
    "# Columns: ab2_m, rhoa_ohmm, uncertainty_ohmm\n"
)
with open(out_ves, "w", encoding="utf-8") as f:
    f.write(header_ves)
    f.write("ab2_m,rhoa_ohmm,uncertainty_ohmm\n")
    for a, r, s in zip(ab2, rhoa_obs, rhoa_sigma):
        f.write(f"{a:.6e},{r:.6e},{s:.6e}\n")

print(f"Wrote {len(ab2)} spacings to {out_ves}")
