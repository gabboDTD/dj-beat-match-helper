import os
from pathlib import Path
import re
import numpy as np
import streamlit as st
import librosa
import plotly.graph_objects as go

# ---------- Constants ----------
HOP_LENGTH = 512        # keep consistent across onset/beat computations
FRAME_LENGTH = 2048

# ---------- Utils ----------

def parse_time_to_seconds(s: str) -> float:
    """
    Parse 'mm:ss' or 'm:ss.s' or plain seconds ('75' or '75.5') to float seconds.
    """
    s = (s or "").strip()
    if not s:
        return 0.0
    if ":" in s:
        # allow mm:ss(.ms)
        m, sec = s.split(":", 1)
        return float(m) * 60.0 + float(sec)
    return float(s)

def load_audio(path, sr=22050, mono=True, start_sec=0.0, duration=None):
    """
    Load a slice of audio (from start_sec, for 'duration' seconds) and normalize to [-1, 1].
    """
    y, sr = librosa.load(path, sr=sr, mono=mono, offset=float(start_sec), duration=duration)
    if y.size > 0:
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
    return y, sr

def estimate_bpm_and_beats(y, sr, hop_length=HOP_LENGTH):
    """
    Return (bpm, beat_times_sec) using librosa's beat tracker.
    beat_times are relative to the loaded slice (start=0 of the slice).
    """
    if y is None or len(y) == 0:
        return None, np.array([])
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    bpm = float(np.round(tempo, 2)) if np.isscalar(tempo) else float(np.round(np.mean(tempo), 2))
    return bpm, beat_times

def required_pitch_percent(bpm_target, bpm_source):
    """
    Pitch percentage needed to change 'bpm_source' to 'bpm_target'.
    Positive => speed up source; Negative => slow down.
    """
    if not bpm_source or not bpm_target:
        return None
    return (bpm_target / bpm_source - 1.0) * 100.0

def waveform_envelope(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """
    Compute a lightweight envelope (RMS) for plotting instead of millions of samples.
    Returns (times_sec, rms)
    """
    if y is None or len(y) == 0:
        return np.array([]), np.array([])
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True).flatten()
    times = librosa.frames_to_time(np.arange(len(rms)), sr=22050, hop_length=hop_length)
    # normalize RMS to ~[-1,1] visual scale (positive only; we'll reflect for pretty fill if desired)
    if np.max(rms) > 0:
        rms = rms / np.max(rms)
    return times, rms

def render_pitch_fader(current_pct=0.0, target_pct=None, pitch_range=16.0, title="Pitch Fader"):
    low = -pitch_range
    high = pitch_range

    def clamp(x):
        if x is None:
            return None
        return max(low, min(high, x))

    current_pct = clamp(current_pct)
    target_pct = clamp(target_pct)

    fig = go.Figure()

    # fader slot
    fig.add_shape(type="rect", x0=0.4, x1=0.6, y0=low, y1=high, line=dict(width=2), fillcolor="rgba(200,200,200,0.2)")
    # zero line
    fig.add_shape(type="line", x0=0.3, x1=0.7, y0=0, y1=0, line=dict(width=2))

    fig.add_trace(go.Scatter(
        x=[0.5], y=[current_pct],
        mode="markers+text",
        marker=dict(size=16, symbol="square"),
        text=[f"Current: {current_pct:.2f}%"],
        textposition="middle right",
        hovertemplate="Current pitch: %{y:.2f}%<extra></extra>",
        name="Current"
    ))

    if target_pct is not None:
        fig.add_trace(go.Scatter(
            x=[0.5], y=[target_pct],
            mode="markers+text",
            marker=dict(size=16, symbol="diamond"),
            text=[f"Target:  {target_pct:.2f}%"],
            textposition="middle left",
            hovertemplate="Target pitch: %{y:.2f}%<extra></extra>",
            name="Target"
        ))
        fig.add_shape(type="line", x0=0.5, x1=0.5, y0=current_pct, y1=target_pct, line=dict(width=2, dash="dot"))

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=True, title="Pitch %", range=[low, high]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_waveform_with_beats(y, sr, beat_times, title, window_start, window_len,
                             matched_beat_times=None):
    """
    Plot a normalized envelope as waveform + vertical lines for beats (and optional matched beats).
    Times on x-axis are relative to the chosen window (0..window_len).
    """
    # envelope in slice time (starts at ~0)
    times_env, env = waveform_envelope(y)

    # Create figure
    fig = go.Figure()

    # Waveform (envelope)
    fig.add_trace(go.Scatter(
        x=times_env, y=env,
        mode="lines",
        name="Waveform (RMS)"
    ))
    # optional mirror for visual symmetry
    fig.add_trace(go.Scatter(
        x=times_env, y=-env,
        mode="lines",
        name="Waveform (mirror)",
        showlegend=False,
        hoverinfo="skip"
    ))

    # Beat grid (native)
    for bt in beat_times:
        fig.add_shape(type="line", x0=bt, x1=bt, y0=-1.1, y1=1.1, line=dict(width=1))
    # Add a small invisible trace just for legend label
    if len(beat_times) > 0:
        fig.add_trace(go.Scatter(
            x=[beat_times[0]], y=[1.05],
            mode="markers",
            name="Beat grid",
            marker=dict(size=6),
            hoverinfo="skip",
            showlegend=True
        ))

    # Matched beat grid (Track 2 adjusted to Track 1)
    if matched_beat_times is not None and len(matched_beat_times) > 0:
        for bt in matched_beat_times:
            fig.add_shape(type="line", x0=bt, x1=bt, y0=-1.1, y1=1.1, line=dict(width=1, dash="dot"))
        fig.add_trace(go.Scatter(
            x=[matched_beat_times[0]], y=[1.05],
            mode="markers",
            name="Matched grid",
            marker=dict(size=6),
            hoverinfo="skip",
            showlegend=True
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title=f"Time in window (s)  —  window: {window_start:.2f}s → {(window_start+window_len):.2f}s",
                   range=[0, max(window_len, times_env[-1] if times_env.size else window_len)]),
        yaxis=dict(title="Norm. amplitude", range=[-1.15, 1.15]),
        margin=dict(l=40, r=20, t=60, b=40),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- App ----------

st.set_page_config(page_title="DJ Beat-Match Helper", page_icon="🎚️", layout="centered")
st.title("🎚️ DJ Beat-Match Helper")
st.write(
    "Load two local MP3 tracks. I’ll estimate their BPM, show turntable-style pitch, "
    "and draw waveforms with beat grids. Track 2 also shows a *matched* grid "
    "for alignment with Track 1."
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

    st.markdown("**Analysis window**")
    start_str = st.text_input("Start (mm:ss or seconds)", value="1:00")
    window_len = st.number_input("Window length (seconds)", min_value=5.0, max_value=180.0, value=30.0, step=5.0)

def pick_file(label, folder_path):
    folder = Path(folder_path).expanduser()
    mp3s = sorted([p for p in folder.glob("*.mp3") if p.is_file()])
    if not mp3s:
        st.warning(f"No MP3 files found in: {folder}")
        return None
    names = [p.name for p in mp3s]
    choice = st.selectbox(label, names)
    return folder / choice

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

    start_sec = parse_time_to_seconds(start_str)
    if start_sec < 0:
        start_sec = 0

    with st.spinner("Loading and analyzing audio…"):
        try:
            # Load selected window only
            y1, sr1 = load_audio(str(path_1), sr=sr, start_sec=start_sec, duration=window_len)
            y2, sr2 = load_audio(str(path_2), sr=sr, start_sec=start_sec, duration=window_len)
        except Exception as e:
            st.error(f"Error reading files: {e}")
            st.stop()

        bpm1, beats1 = estimate_bpm_and_beats(y1, sr1, hop_length=HOP_LENGTH)
        bpm2, beats2 = estimate_bpm_and_beats(y2, sr2, hop_length=HOP_LENGTH)
        delta_pct = required_pitch_percent(bpm_target=bpm1, bpm_source=bpm2) if (bpm1 and bpm2) else None
        matched_bpm = bpm2 * (1.0 + (delta_pct or 0.0) / 100.0) if bpm2 else None

    st.subheader("2) Results")

    # Track 1 card
    with st.container(border=True):
        st.markdown(f"**Track 1 (reference):** `{path_1.name}`")
        if bpm1 is None:
            st.error("Could not detect BPM for Track 1.")
        else:
            st.write(f"**Estimated BPM (in window):** {bpm1:.2f}")
            st.write("**Pitch position (deck):** 0.00% (reference)")
            render_pitch_fader(current_pct=0.0, target_pct=None, pitch_range=float(pitch_range), title="Track 1 Pitch")
            # Waveform + beat grid
            plot_waveform_with_beats(
                y=y1, sr=sr1, beat_times=beats1,
                title="Track 1 — Waveform & Beat Grid",
                window_start=start_sec, window_len=window_len
            )
            st.caption(f"Beats detected in window: {len(beats1)}")

    # Track 2 card
    with st.container(border=True):
        st.markdown(f"**Track 2 (to match Track 1):** `{path_2.name}`")
        if bpm2 is None:
            st.error("Could not detect BPM for Track 2.")
        else:
            st.write(f"**Estimated BPM (in window):** {bpm2:.2f}")
            if delta_pct is None:
                st.error("Could not compute the required pitch adjustment.")
                matched_beats_for_plot = None
            else:
                sign = "speed up (+)" if delta_pct >= 0 else "slow down (−)"
                st.write("**Current pitch (deck):** 0.00%")
                st.write(f"**Set pitch to:** `{delta_pct:+.2f}%` → {sign} to match Track 1 at {bpm1:.2f} BPM")
                if matched_bpm:
                    st.caption(f"After adjustment, Track 2 BPM ≈ {matched_bpm:.2f}")

                # Time-warp beat positions to show where they'd land after pitch change.
                # Speeding up (positive %) compresses time; slowing down expands time.
                time_scale = 1.0 + (delta_pct / 100.0)
                matched_beats_for_plot = beats2 / time_scale if time_scale and time_scale > 0 else None

            render_pitch_fader(
                current_pct=0.0,
                target_pct=float(delta_pct) if delta_pct is not None else None,
                pitch_range=float(pitch_range),
                title="Track 2 Pitch"
            )

            # Waveform + beat grid (native) + matched grid (dotted)
            plot_waveform_with_beats(
                y=y2, sr=sr2, beat_times=beats2,
                matched_beat_times=matched_beats_for_plot if matched_beats_for_plot is not None else None,
                title="Track 2 — Waveform, Beat Grid & Matched Grid",
                window_start=start_sec, window_len=window_len
            )
            st.caption(f"Beats detected in window: {len(beats2)}")

    with st.expander("Notes & tips"):
        st.markdown(
            "- BPM and beats are estimated **only on the selected window**.\n"
            "- Double/half tempo ambiguity (e.g., 85 vs 170 BPM) can happen—use the same window for both tracks and, if needed, try a different window.\n"
            "- The **matched grid** for Track 2 shows where beats would fall **after** applying the suggested pitch, drawn over the *current* waveform time axis."
        )

else:
    st.info("Pick two MP3s, choose the analysis window (e.g., start 1:00, length 30s), then click **Analyze**.")

