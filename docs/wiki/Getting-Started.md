# Getting Started

## 1) Requirements
- Python **3.10+** (tested on macOS and Linux; Python 3.13 works but pay attention to SciPy pins when using Fairlearn)
- A working C/C++ toolchain is *not* required (wheels are used for SciPy on mainstream platforms)

## 2) Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[ui,data]"
```

Optional extras:
```bash
# Decision Brief PDF
python -m pip install reportlab

# Strong constrained-optimization baseline
python -m pip install -e ".[fairlearn]"
```

### macOS + Python 3.13 note (Fairlearn)
`fairlearn==0.13.0` requires `scipy<1.16`. If pip upgrades SciPy, re-pin:
```bash
python -m pip install "scipy<1.16.0" --force-reinstall
```

## 3) Quick run
```bash
python -m compass_ecl_mas.cli.run_all --config configs/education_quick.yaml
```

You should see:
- an `[OK] <run_id>` line
- `outputs/<run_id>/overleaf_dropin/` created

## 4) Run the Streamlit app (optional)
```bash
streamlit run app/streamlit_app.py
```

