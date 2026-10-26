# Sus.exe — Narrative Radar

Hackathon prototype for PS-03: Social Media Coordinated Disinformation Detection System.

## Features
- TF-IDF cosine similarity between posts
- Narrative clusters based on similar-post relationships
- Cross-account links when similar posts occur within a selected time window
- Review queue, CSV export, and interactive network visualization
- CSV upload for a custom dataset
- Explicit caveats to avoid treating similarity as proof of disinformation

## Run locally
Use Python 3.10+.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Windows Command Prompt:
```bat
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## CSV format
Required columns: `post_id, account, timestamp, text`
Optional: `platform`

Example:
```csv
post_id,account,timestamp,text,platform
P1,@sample1,2026-10-26 09:00,"Example post text",SampleNet
P2,@sample2,2026-10-26 09:04,"Similar example post",SampleNet
```

## Important limitations
This is a short hackathon prototype, not a validated disinformation classifier. It detects similarity and timing patterns that may warrant human review. It does not determine whether a claim is true, whether an account is a bot, or whether users coordinated intentionally. Demo records are synthetic.
