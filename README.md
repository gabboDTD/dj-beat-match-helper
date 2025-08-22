# 🎚️ DJ Beat-Match Helper

A tiny Streamlit app that:
1. Loads two local MP3s
2. Estimates their BPM
3. Shows a turntable-style pitch fader for Track 1 (reference)
4. Tells you exactly where to set Track 2’s pitch to beat-match Track 1

## ✨ Features
- BPM detection via `librosa`
- Suggested pitch in %
- Visual pitch fader (Plotly)
- Configurable pitch range (±%)

## 📦 Requirements
- Python 3.10+
- [Poetry](https://python-poetry.org/)
- `ffmpeg` installed for MP3 decoding

## 🚀 Quickstart

```bash
# clone
git clone https://github.com/<your-username>/dj-beat-match-helper.git
cd dj-beat-match-helper

# setup (first time)
poetry install

# run
poetry run streamlit run app.py
