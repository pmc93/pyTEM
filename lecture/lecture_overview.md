# Near-Surface Geophysics for Hydrogeology, Around the World

**Guest lecture, S&I DSC seminar series, DTU**
**Speaker:** Paul McLachlan (Assistant Professor, DTU)
**Date:** 22 June
**Format:** 60-minute talk. The ResIPy hands-on is a **separate session** (see end of this document).

> The talk has one job: tell the story of how a near-surface geophysics career actually unfolds, from a field problem, to a method, to a global application. Electrical methods and ResIPy get a short teaser here; the real hands-on happens later.
>
> Total budget for this slot: **60 minutes** (aim to land the content in ~50 and leave ~10 for questions).

---

## 0. Framing (3-5 min)

- Who the audience is: GFR and S&I DSC people, mixed backgrounds, some new to geophysics.
- The through-line of the talk: **water is the problem, geophysics is the lens.** Every site, every continent, the question is some version of "where is the water, how much, how does it move, and is it changing?"
- One sentence on why electrical methods: they are sensitive to the things hydrogeologists care about (porosity, saturation, salinity, clay/lithology, and via IP, surface chemistry and texture).

---

## Part 1 - The career journey (20-25 min)

A suggested narrative arc. Fill the brackets with your own dates, places, and stories. The idea is to make each stage about *a problem that pulled you to a method*, not a CV list.

### 1.1 Origins: how I got into near-surface geophysics
- Background and first degree: `[where, in what]`.
- The moment geophysics clicked: `[the project / field campaign / person]`.
- Early skills: fieldwork, data that did not behave, learning that the subsurface is never homogeneous.

### 1.2 PhD: a focused hydrogeophysical question
- The core question: `[e.g. groundwater-surface water exchange, the critical zone, contaminant transport]`.
- Methods you built expertise in: electrical resistivity tomography (ERT), induced polarization (IP), self-potential, `[others]`.
- Supervisor / group and what they taught you: `[Lancaster / Binley group, Aarhus HGG, or as appropriate]`.
- One result you are still proud of, and one thing that failed and taught you more.

### 1.3 Postdoc / research positions: going broader
- New environments and scales: `[lab, plot, catchment, region]`.
- New methods added to the toolbox: time-lapse monitoring, petrophysical joint interpretation, `[EM / TEM, GPR, ...]`.
- The shift from "make a nice image" to "answer a quantitative hydrological question".

### 1.4 Working around the world
- A map slide: pin every country/site you have worked in. This is the memorable visual.
- For 3-4 contrasting sites, one slide each with the same template:
  - **Place & setting** (climate, geology, the water problem)
  - **Why geophysics** (what couldn't be measured any other way)
  - **Method used** (ERT / IP / EM / TEM ...)
  - **What we learned** (the hydrogeological payoff)
  - Suggested contrasts: a humid temperate catchment, an arid/semi-arid aquifer, a coastal saltwater-intrusion site, an agricultural/managed-aquifer site.

### 1.5 Now at DTU
- Your current role and research direction: `[group, themes]`.
- How this connects to the audience: collaborations, data, students, the DSC.
- Where the field is going: time-lapse and autonomous monitoring, petrophysical inversion, machine learning on geophysical data, open-source tools (ResIPy, and tools like the pyTEM work in this group).

> Transition line into the hands-on: "Most of what you just saw rests on one workflow - turning resistance measurements into a subsurface model. Let's actually do that now."

---

## Part 2 - ResIPy teaser (within the talk, ~5 min)

Keep this short in the 60-minute slot. The goal is only to show *what the tool does*, so people are motivated to come to the hands-on session. The full walkthrough below lives in the separate ResIPy session.

- One slide: a raw pseudosection in, a clean resistivity (and phase) section out.
- One line: "this is open-source, scriptable, and we will all run it together in the hands-on session."
- Show the five-step mental model (below) as a single slide, then move on.

---

# Separate session - ResIPy hands-on (60-90 min)

ResIPy is the open-source Python interface (GUI + API) to the R* family of inversion codes (R2 / cR2 for 2D resistivity & IP, R3t / cR3t for 3D and time-lapse). The aim is for everyone to run a full processing-to-inversion pipeline once, then know where the knobs are.

### 2.1 What ResIPy is and why it exists (5 min)
- Lowers the barrier to high-quality ERT/IP inversion: no hand-editing of mesh and protocol files.
- Two ways to drive it:
  - **GUI** for teaching, QC, and quick jobs.
  - **Python API** for scripting, batch/time-lapse, and reproducibility.
- The engine underneath: Occam-style regularised inversion (the same regularised Gauss-Newton idea used across our group's codes).

### 2.2 The mental model (5 min)
The workflow is always the same five steps:

1. **Import** data (survey file: electrode positions + quadripole measurements).
2. **Filter / QC**: reciprocal error, stacking error, contact resistance, obvious outliers.
3. **Mesh**: triangular (2D) or tetrahedral (3D); finer near electrodes.
4. **Invert**: choose resistivity-only or complex (IP), set regularisation and error model.
5. **Interpret**: resistivity (and phase/chargeability) sections, fit quality, sensitivity/coverage.

### 2.3 Live demo - GUI (10-15 min)
- Load an example survey (ResIPy ships with built-in datasets).
- Show reciprocal error filtering and why an **error model** matters for honest inversion.
- Build a mesh, run a 2D inversion, read the resistivity section.
- Toggle to an IP dataset and show the phase image alongside resistivity.
- Point out the fit: RMS / chi-squared near 1 = fitting to noise, not over- or under-fitting.

### 2.4 Live demo - Python API (10 min)
Minimal script to show reproducibility (audience can copy this):

```python
from resipy import Project

k = Project(typ='R2')             # 2D resistivity (use 'cR2' for IP)
k.createSurvey('data/survey.csv', ftype='Syscal')
k.filterRecip(percent=5)          # drop quadripoles with >5% reciprocal error
k.fitErrorModel()                 # data-driven error weighting
k.createMesh(typ='trian')         # triangular mesh, refined at electrodes
k.invert()                        # regularised Gauss-Newton inversion
k.showResults(attr='Resistivity(log10)')
```

- Swap `typ='cR2'` to invert IP and plot `attr='Phase(mrad)'`.
- Mention time-lapse: a list of surveys + `k.invert(reg_mode=...)` for difference inversion.

### 2.5 Tie-back to hydrogeology (3-5 min)
- Resistivity -> porosity / saturation / salinity via Archie or a site petrophysical relation.
- IP/phase -> surface area, clay content, lithology, and biogeochemical change.
- Time-lapse -> moving water: infiltration, recharge, intrusion, remediation.
- Caveat slide: resolution falls off with depth, sensitivity is non-uniform, and inversion is non-unique. Always show coverage/sensitivity, never just the pretty section.

---

## Talk wrap-up & discussion (~10 min, end of the 60-min slot)

- One-slide summary: **field problem -> method -> inversion -> hydrogeological answer.**
- How people here can collaborate: shared data, student projects, joint TEM + ERT/IP interpretation (links naturally to the group's pyTEM work).
- Resources to share:
  - ResIPy docs & GitHub (gitlab.com / the official repo) and the example datasets.
  - Key reading: Binley & Slater, *Resistivity and Induced Polarization* (2020); Binley et al. (2015) hydrogeophysics review.
  - Your own papers as worked examples: `[add 2-3 references]`.
- Open Q&A.

---

## Practical checklist

**For the talk:**
- [ ] World map slide with site pins (the centerpiece visual).
- [ ] 2-3 of your own field photos per site - these carry the "down-to-earth" tone people expect.
- [ ] One clean ResIPy before/after slide for the teaser.
- [ ] Fill the `[...]` brackets in the career section.

**For the separate ResIPy session:**
- [ ] Install ResIPy beforehand and confirm it launches (`pip install resipy`, or the bundled executable).
- [ ] Pre-load the example dataset(s) so the demo never waits on a download.
- [ ] Have one resistivity and one IP dataset ready.
- [ ] Backup: pre-computed inversion results in case live inversion is slow on the room's hardware.

## Suggested timing - the 60-minute talk

| Block | Minutes |
|-------|--------:|
| Framing | 3 |
| Career journey (origins -> PhD -> postdoc) | 18 |
| Working around the world (the map + site slides) | 18 |
| Now at DTU + where the field is going | 6 |
| ResIPy teaser | 5 |
| Discussion / Q&A | 10 |
| **Total** | **60** |

## Suggested timing - the separate ResIPy hands-on (60-90 min)

| Block | Minutes |
|-------|--------:|
| What ResIPy is + mental model | 10 |
| GUI demo (import -> filter -> mesh -> invert) | 20 |
| Python API demo (reproducible script) | 15 |
| IP and time-lapse | 10 |
| Hydrogeology tie-back + their own data | 10-30 |
