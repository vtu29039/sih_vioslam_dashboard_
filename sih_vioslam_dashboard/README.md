---
title: SIH VIO-SLAM Dashboard
emoji: 🚁
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.41.1
python_version: "3.11"
app_file: app.py
pinned: false
---

# Zero-GNSS VIO-SLAM Navigation & Risk Dashboard

Live demo dashboard powered by **DynaFusion-VIO**.

## Run locally
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Hugging Face Spaces
1. Create an account at huggingface.co, then New Space -> SDK: Streamlit -> Public.
2. Upload every file in this folder (keep the `src/` and `config/` folders), or `git push` to the Space repo.
3. Wait for the build; open the app at `https://<user>-<space>.hf.space` (direct URL, not the embedded page).
4. Choose "Browser camera", click START and allow camera access.

Optional (if the video will not connect on some networks): add Space secrets
`TURN_URL`, `TURN_USERNAME`, `TURN_CREDENTIAL` from a free TURN provider.
