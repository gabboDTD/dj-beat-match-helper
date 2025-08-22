import os
from pathlib import Path
import numpy as np
import streamlit as st
import librosa
import plotly.graph_objects as go

# ---------- Core DSP ----------

def load_audio(path, sr=22050, mono=True):
    """Load audio with librosa, returning (y, sr)."""
    # y, sr = librosa.load(path, sr=sr, mono=mono)
    # Load with a fixed offset and duration to avoid long intros/outros
    y, sr = librosa.load(path, sr=22050, mono=True, offset=120.0, duration=60.0)
    # Basic normalization to avoid level issues
    if y.size > 0 and np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y, sr

def estimate_bpm(y, sr):
    """
    Estimate BPM using librosa's beat tracker.
    Returns a float BPM (rounded to 2 decimals).
    """
    if y is None or len(y) == 0:
        return None
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if np.isscalar(tempo):
        return float(np.round(tempo, 2))
    # In rare cases tempo can be an array; take mean
    return float(np.round(np.mean(tempo), 2))

def required_pitch_percent(bpm_target, bpm_source):
    """
    Pitch percentage needed to change 'bpm_source' to 'bpm_target'.
    Positive => speed up source; Negative => slow down.
    """
    if bpm_source is None or bpm_target is None or bpm_source == 0:
        return None
    return (bpm_target / bpm_source - 1.0) * 100.0

# ---------- UI Helpers ----------

def render_pitch_fader(current_pct=0.0, target_pct=None, pitch_range=16.0, title="Pitch Fader"):
    """
    Draw a vertical 'turntable pitch' fader with current and (optional) target markers.
    """
    low = -pitch_range
    high = pitch_range

    # Clamp values to the visible range for rendering
    def clamp(x):
        if x is None:
            return None
        return max(low, min(high, x))

    current_pct = clamp(current_pct)
    target_pct = clamp(target_pct)

    fig = go.Figure()

    # Background: the fader slot
    fig.add_shape(
        type="rect",
        x0=0.4, x1=0.6,
        y0=low, y1=high,
        line=dict(width=2),
        fillcolor="rgba(200,200,200,0.2)"
    )

    # Center "0%" line
    fig.add_shape(
        type="line",
        x0=0.3, x1=0.7,
        y0=0, y1=0,
        line=dict(width=2)
    )

    # Current position marker
    fig.add_trace(go.Scatter(
        x=[0.5],
        y=[current_pct],
        mode="markers+text",
        marker=dict(size=16, symbol="square"),
        text=[f"Current: {current_pct:.2f}%"],
        textposition="middle right",
        hovertemplate="Current pitch: %{y:.2f}%<extra></extra>",
        showlegend=False,
    ))

    # Target position marker (optional)
    if target_pct is not None:
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[target_pct],
            mode="markers+text",
            marker=dict(size=16, symbol="diamond"),
            text=[f"Target:  {target_pct:.2f}%"],
            textposition="middle left",
            hovertemplate="Target pitch: %{y:.2f}%<extra></extra>",
            showlegend=False,
        ))

        # Line between current & target for clarity
        fig.add_shape(
            type="line",
            x0=0.5, x1=0.5,
            y0=current_pct, y1=target_pct,
            line=dict(width=2, dash="dot")
        )

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=True, title="Pitch %", range=[low, high]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

def pick_file(label, folder_path):
    """
    Offer a selectbox of .mp3 files from a local folder.
    """
    folder = Path(folder_path).expanduser()
    mp3s = sorted([p for p in folder.glob("*.mp3") if p.is_file()])
    if not mp3s:
        st.warning(f"No MP3 files found in: {folder}")
        return None
    names = [p.name for p in mp3s]
    choice = st.selectbox(label, names)
    return folder / choice

# ---------- App ----------

st.set_page_config(page_title="DJ Beat-Match Helper", page_icon="🎚️", layout="centered")

st.title("🎚️ DJ Beat-Match Helper")
st.write(
    "Load two local MP3 tracks. I’ll estimate their BPM and show a turntable-style pitch fader.\n"
    "For Track 2, you’ll see exactly where to set the pitch to beat-match Track 1."
)

with st.sidebar:
    st.header("Settings")
    default_folder = st.text_input(
        "Local folder with MP3 files",
        value=str(Path.cwd()),
        help="Browse to a folder that contains your MP3s."
    )
    pitch_range = st.slider("Pitch fader range (±%)", min_value=4, max_value=50, value=16, step=1)
    sr = st.select_slider("Analysis sample rate", options=[11025, 16000, 22050, 32000, 44100], value=22050)
    st.caption("Tip: If detection seems off, try a different sample rate.")

st.subheader("1) Choose your tracks")
col1, col2 = st.columns(2)
with col1:
    path_1 = pick_file("Track 1 (reference)", default_folder)
with col2:
    path_2 = pick_file("Track 2 (to match Track 1)", default_folder)

run = st.button("Analyze")

if run:
    if path_1 is None or path_2 is None:
        st.error("Please select both tracks.")
        st.stop()

    with st.spinner("Loading and analyzing audio…"):
        try:
            y1, sr1 = load_audio(str(path_1), sr=sr)
            y2, sr2 = load_audio(str(path_2), sr=sr)
        except Exception as e:
            st.error(f"Error reading files: {e}")
            st.stop()

        bpm1 = estimate_bpm(y1, sr1)
        bpm2 = estimate_bpm(y2, sr2)

    st.subheader("2) Results")

    # Track 1 card
    with st.container(border=True):
        st.markdown(f"**Track 1 (reference):** `{path_1.name}`")
        if bpm1 is None:
            st.error("Could not detect BPM for Track 1.")
        else:
            st.write(f"**Estimated BPM:** {bpm1:.2f}")
            st.write("**Pitch position (deck):** 0.00% (reference)")
            render_pitch_fader(current_pct=0.0, target_pct=None, pitch_range=float(pitch_range), title="Track 1 Pitch")

    # Track 2 card
    with st.container(border=True):
        st.markdown(f"**Track 2 (to match Track 1):** `{path_2.name}`")
        if bpm2 is None:
            st.error("Could not detect BPM for Track 2.")
        else:
            st.write(f"**Estimated BPM:** {bpm2:.2f}")
            delta_pct = required_pitch_percent(bpm_target=bpm1, bpm_source=bpm2) if bpm1 else None

            if delta_pct is None:
                st.error("Could not compute the required pitch adjustment.")
            else:
                st.write("**Current pitch (deck):** 0.00%")
                sign = "speed up (+)" if delta_pct >= 0 else "slow down (−)"
                st.write(f"**Set pitch to:** `{delta_pct:+.2f}%` → {sign} to match Track 1 at {bpm1:.2f} BPM")

                # Show what the resulting BPM would be after applying the suggested pitch
                matched_bpm = bpm2 * (1.0 + delta_pct / 100.0)
                st.caption(f"After adjustment, Track 2 BPM ≈ {matched_bpm:.2f}")

                # Visual fader
                render_pitch_fader(
                    current_pct=0.0,
                    target_pct=float(delta_pct),
                    pitch_range=float(pitch_range),
                    title="Track 2 Pitch"
                )

    # Quality notes
    with st.expander("Why might BPM be slightly off?"):
        st.markdown(
            "- Very sparse or very dense percussion can fool automatic beat trackers.\n"
            "- Double/half-tempo ambiguity (e.g., 85 vs 170 BPM) can occur.\n"
            "- If needed, try a different analysis sample rate or prep files with clear intros."
        )

else:
    st.info("Pick two MP3s from your folder and click **Analyze**.")

