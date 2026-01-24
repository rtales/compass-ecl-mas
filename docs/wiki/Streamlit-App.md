# Streamlit App

Run:
```bash
streamlit run app/streamlit_app.py
```

## What the app is for
The Streamlit UI is meant for:
- interactive constraint calibration (what happens if we tighten ΔEOpp?)
- visual exploration of F1 × ΔEOpp × Coverage trade-offs
- packaging Overleaf-ready artifacts
- reviewing results from a CLI run by uploading `bundle.zip`

## Where outputs go
When you run from the UI, the app will:
- write outputs to a new `outputs/<run_id>/` folder (same format as the CLI)
- offer download buttons for `bundle.zip` and Overleaf drop-ins

## Review mode
If you already ran the CLI:
1) open the app
2) choose “Review”
3) upload `outputs/<run_id>/bundle.zip`

The app will load:
- metrics tables
- Pareto plots
- MAS selection summary

