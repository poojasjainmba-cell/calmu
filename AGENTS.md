# AGENTS.md

This is a Python Streamlit app.

- Run dashboard with: `streamlit run dashboard.py`
- Run first sync with: `python sync.py`
- Run field profiler with: `python field_profiler.py`
- Run tests with: `pytest`
- Never hardcode HubSpot tokens.
- Use Contact Owner first.
- Use Deal Owner only as fallback.
- Do not guess HubSpot fields.
- Hide charts when fields are missing.
- Do not double-count revenue.
