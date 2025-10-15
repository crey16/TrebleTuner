#!/usr/bin/env python3
# notes_by_pitch_changes.py  — prints: time(s)   hz   note
import argparse, pathlib, sys, math
import numpy as np
import soundfile as sf
import aubio

def hz_to_midi(hz):
    return 69 + 12*math.log2(hz/440.0) if hz > 0 else None
def midi_to_hz(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))
def midi_to_note(m):
    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    n = int(round(m))
    return f"{names[n%12]}{n//12-1}"

def median_midi_in(times, midi_seq, t0, t1):
    i0 = np.searchsorted(times, t0, side="left")
    i1 = np.searchsorted(times, t1, side="left")
    window = [m for m in midi_seq[i0:i1] if m is not None]
    return (float(np.median(window)) if window else None)

def main():
    # --- Configuration: Set your audio file and parameters here ---
    audio_file_path = "audios/audio7.wav" # <-- CHANGE THIS to your audio file
    
    # Pitch and Onset parameters (previously command-line arguments)
    algo = "yin"
    onset = "hfc"
    win = 2048
    hop = 256
    conf = 0.55      # Pitch confidence gate
    min_hz = 40.0
    max_hz = 2000.0
    smooth_ms = 150.0 # Pitch smoothing
    minioi_ms = 80.0  # Minimum onset spacing
    tail_ms = 40.0    # Skip this much after onset before measuring
    # --- End Configuration ---

    # --- File Handling ---
    path = pathlib.Path(audio_file_path).expanduser()
    if not path.is_file():
        sys.exit(f"Error: Audio file not found at '{path}'")
    
    print(f"Processing audio file: {path.name}\n")
    # --- End File Handling ---

    # load mono
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim == 2: x = x.mean(axis=1).astype(np.float32)

    # aubio pitch
    pit = aubio.pitch(algo, win, hop, sr); pit.set_unit("Hz")
    pit.set_silence(-40); pit.set_tolerance(0.8)

    # aubio onset
    ons = aubio.onset(onset, win, hop, sr)
    ons.set_silence(-40)
    ons.set_minioi_s(max(0.01, minioi_ms/1000.0))

    # pass 1: gather per-frame times + pitched frames and onsets
    times, hz_list, confs, onsets = [], [], [], []
    for i in range(0, len(x)-hop, hop):
        frame = x[i:i+hop]
        t = i / sr
        if ons(frame).item() > 0:
            onsets.append(t)
        hz = float(pit(frame).item()); conf_val = float(pit.get_confidence())
        if hz > 0 and min_hz <= hz <= max_hz and conf_val >= conf:
            times.append(t); hz_list.append(hz); confs.append(conf_val)

    # ensure we start and end with boundaries
    if not onsets or onsets[0] > 0.05: onsets = [0.0] + onsets
    onsets.append((len(x)-1)/sr)

    print("time\thz\tnote")
    if not times:
        return

    # smooth pitch in MIDI + light octave correction
    midi_raw = [hz_to_midi(hz) for hz in hz_list]
    W = max(1, int(round((smooth_ms/1000.0) * sr / hop)))
    midi_s = []
    for k in range(len(midi_raw)):
        lo, hi = max(0, k-W//2), min(len(midi_raw), k+W//2+1)
        vals = [m for m in midi_raw[lo:hi] if m is not None]
        midi_s.append(float(np.median(vals)) if vals else None)

    corrected = []
    for m in midi_s:
        if not corrected or m is None or corrected[-1] is None:
            corrected.append(m); continue
        prev, cand = corrected[-1], m
        if abs(cand - prev) > 6:
            while cand - prev > 6: cand -= 12
            while prev - cand > 6: cand += 12
        corrected.append(cand)

    # one note per onset-to-onset window
    tail = tail_ms/1000.0
    for k in range(len(onsets)-1):
        t0, t1 = onsets[k], onsets[k+1]
        rep = median_midi_in(times, corrected, t0 + tail, t1)
        if rep is None:  # fallback: right at onset
            rep = median_midi_in(times, corrected, t0, min(t1, t0 + 0.25))
        if rep is None:
            continue
        print(f"{t0:.3f}\t{midi_to_hz(rep):.1f}\t{midi_to_note(rep)}")

if __name__ == "__main__":
    main()