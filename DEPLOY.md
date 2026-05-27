# Deploy this dashboard

## Streamlit Community Cloud (free, ~3 min)

1. [streamlit.io/cloud](https://streamlit.io/cloud) → **Sign in with GitHub** (use `nanettetada`).
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `nanettetada/review-sentiment-engine`
   - **Branch:** `main`
   - **Main file path:** `dashboard.py`
4. **Deploy**.

## Notes

- Auto-rebuilds on every push to `main`.
- Free tier sleeps after ~7 days idle.
- Alternative: Hugging Face Spaces with SDK = Streamlit.
