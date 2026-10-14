# Heart Disease Prediction — Educational ML Demo

A Streamlit demonstration of Logistic Regression and Decision Tree classification on the included heart disease dataset.

**Not a medical device.** This demo is not clinically validated, cannot diagnose heart disease or detect a heart attack, and must not be used for medical decisions. Use only synthetic example inputs.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy

Push `app.py`, `heart.csv`, and `requirements.txt` to GitHub. At https://share.streamlit.io/ create a new app, select your repository and `main` branch, and set entrypoint `app.py`.

## Notes

The app trains its models on startup using a stratified train/test split. Test metrics are illustrative, not clinical validation. Dataset provenance, label semantics, feature encodings, licensing, leakage, fairness, and external validation should be reviewed before broader use.
