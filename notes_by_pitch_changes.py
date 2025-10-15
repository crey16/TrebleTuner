#!/usr/bin/env python3
# notes_by_pitch_changes.py  — prints: time(s)    hz    note
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
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(pathlib.Path.home()/ "Downloads" / "audio4.wav"))
    ap.add_argument("--algo", default="yin", choices=["yin","yinfft","yinfast","mcomb","schmitt","fcomb","default"])
    ap.add_argument("--onset", default="hfc", choices=["default","energy","hfc","complex","phase","specdiff","kl","mkl"])
    ap.add_argument("--win", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--conf", type=float, default=0.55)      # pitch confidence gate
    ap.add_argument("--min-hz", type=float, default=40.0)
    ap.add_argument("--max-hz", type=float, default=2000.0)
    ap.add_argument("--smooth-ms", type=float, default=150.0) # pitch smoothing
    ap.add_argument("--minioi-ms", type=float, default=80.0)  # minimum onset spacing
    ap.add_argument("--tail-ms", type=float, default=40.0)    # skip this much after onset before measuring
    args = ap.parse_args()

    path = pathlib.Path(args.path).expanduser()
    if not path.exists(): sys.exit(f"File not found: {path}")

    # load mono
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim == 2: x = x.mean(axis=1).astype(np.float32)

    hop, win = args.hop, args.win

    # aubio pitch
    pit = aubio.pitch(args.algo, win, hop, sr); pit.set_unit("Hz")
    pit.set_silence(-40); pit.set_tolerance(0.8)

    # aubio onset
    ons = aubio.onset(args.onset, win, hop, sr)
    ons.set_silence(-40)
    ons.set_minioi_s(max(0.01, args.minioi_ms/1000.0))

    # pass 1: gather per-frame times + pitched frames and onsets
    times, hz_list, confs, onsets = [], [], [], []
    for i in range(0, len(x)-hop, hop):
        frame = x[i:i+hop]
        t = i / sr
        if ons(frame).item() > 0:
            onsets.append(t)
        hz = float(pit(frame).item()); conf = float(pit.get_confidence())
        if hz > 0 and args.min_hz <= hz <= args.max_hz and conf >= args.conf:
            times.append(t); hz_list.append(hz); confs.append(conf)

    # ensure we start and end with boundaries
    if not onsets or onsets[0] > 0.05: onsets = [0.0] + onsets
    onsets.append((len(x)-1)/sr)

    print("time\thz\tnote")
    if not times:
        return

    # smooth pitch in MIDI + light octave correction
    midi_raw = [hz_to_midi(hz) for hz in hz_list]
    W = max(1, int(round((args.smooth_ms/1000.0) * sr / hop)))
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
    tail = args.tail_ms/1000.0
    for k in range(len(onsets)-1):
        t0, t1 = onsets[k], onsets[k+1]
        rep = median_midi_in(times, corrected, t0 + tail, t1)
        if rep is None:  # fallback: right at onset
            rep = median_midi_in(times, corrected, t0, min(t1, t0 + 0.25))
        if rep is None:
            continue
        print(f"{t0:.3f}\t{midi_to_hz(rep):.1f}\t{midi_to_note(rep)}")

# --- put this right above: if __name__ == "__main__": main() ---
from pathlib import Path
import sys

def _defaults_when_no_args():
    if len(sys.argv) == 1:  # no CLI args → use your presets
        sys.argv += [
            str(Path.home() / "Downloads" / "audio4.wav"),
            "--onset", "hfc",
            "--algo", "yin",
            "--conf", "0.7",
            "--minioi-ms", "60",
            "--tail-ms", "60",
            "--smooth-ms", "150",
            "--min-hz", "82",          # guitar-ish low E2
            "--max-hz", "660",         # ~E5
        ]

if __name__ == "__main__":
    _defaults_when_no_args()
    main()


if __name__ == "__main__":
    main()
