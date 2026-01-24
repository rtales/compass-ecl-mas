# Decision Brief PDF (Decision-makers)

The pipeline can optionally generate a **Decision Brief** PDF:
- designed for non-technical stakeholders (policy, school leaders, auditors)
- summarizes the *performance vs. equity vs. coverage* trade-offs
- highlights the MAS-selected candidate and why it was selected
- includes “what to do next” recommendations (governance steps)

## Enable it
Install the PDF dependency:
```bash
python -m pip install reportlab
```

## Output location
If enabled, you will find:
- `outputs/<run_id>/decision_brief.pdf`

## Recommended usage
Use the Decision Brief as:
- an executive summary in governance meetings
- documentation for audit trails
- a companion to the paper-grade plots

It is *not* a substitute for the full technical artifacts (CSV/tables/figures); it is a communication layer.

