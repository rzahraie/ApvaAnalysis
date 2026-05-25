# APVA Jupyter Script Workflow

Going forward, generated Jupyter/Python research scripts will be written into this directory instead of being pasted into the chat.

Suggested workflow:

1. Pull latest repo changes.
2. Open the desired script from `scripts/`.
3. Paste into Jupyter or execute directly.
4. Commit generated artifacts (`tables/`, `figures/`, exported CSVs).
5. Push updates.
6. Ask ChatGPT to analyze the new outputs.

Recommended naming convention:

- `apva_<phase>_<version>.py`
- Example:
  - `apva_walk_forward_day_split_v1.py`
  - `apva_regime_pressure_v3.py`
  - `apva_transition_markov_validation_v1.py`

This preserves:
- reproducibility,
- auditability,
- deterministic research history,
- and prompt continuity.
