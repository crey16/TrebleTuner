#!/usr/bin/env python3
# notes_by_pitch_changes.py  — prints: time(s)    hz    note
from pathlib import Path
import sys
import argparse
import pathlib
import sys
import math
import numpy as np
import soundfile as sf
import aubio

# --- Your requested file path ---
DEFAULT_AUDIO_FILE = "/Users/collinrey/Desktop/smartmusicclone/TrebleTuner/audios/audio.wav"


def hz_to_midi(hz):
    return 69 + 12*math.log2(hz/440.0) if hz > 0 else None


def midi_to_hz(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))


def midi_to_note(m):
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    n = int(round(m))
    return f"{names[n%12]}{n//12-1}"


def median_midi_in(times, midi_seq, t0, t1):
    i0 = np.searchsorted(times, t0, side="left")
    i1 = np.searchsorted(times, t1, side="left")
    window = [m for m in midi_seq[i0:i1] if m is not None]
    return (float(np.median(window)) if window else None)


def main():
    ap = argparse.ArgumentParser()
    # MODIFIED: Changed default path
    ap.add_argument("path", nargs="?", default=DEFAULT_AUDIO_FILE)
    ap.add_argument("--algo", default="yin",
                    choices=["yin", "yinfft", "yinfast", "mcomb", "schmitt", "fcomb", "default"])
    ap.add_argument("--onset", default="hfc", choices=[
                    "default", "energy", "hfc", "complex", "phase", "specdiff", "kl", "mkl"])
    ap.add_argument("--win", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--min-hz", type=float, default=40.0)
    ap.add_argument("--max-hz", type=float, default=2000.0)
    ap.add_argument("--smooth-ms", type=float, default=150.0)
    ap.add_argument("--minioi-ms", type=float, default=80.0)
    ap.add_argument("--tail-ms", type=float, default=40.0)
    args = ap.parse_args()

    path = pathlib.Path(args.path).expanduser()
    if not path.exists():
        print(f"--- Error: File not found ---")
        print(f"The script is looking for: {path.resolve()}")
        print("Please check the path and try again.")
        sys.exit(f"File not found: {path}")

    # load mono
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        print(f"Error reading audio file {path}: {e}")
        return

    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32)

    # --- ADDED NORMALIZATION ---
    # This is required for 'audio2.wav' to work, as it's too quiet.
    peak_amplitude = np.max(np.abs(x))
    if peak_amplitude > 0:
        x = (x / peak_amplitude) * 0.9
        print(
            f"Note: Audio has been normalized (original peak: {peak_amplitude:.6f}).")
    else:
        print("Warning: Audio file is silent.")
        print("time\thz\tnote")
        return
    # --- END NORMALIZATION ---

    hop, win = args.hop, args.win

    # aubio pitch
    pit = aubio.pitch(args.algo, win, hop, sr)
    pit.set_unit("Hz")
    # MODIFIED: Loosened silence threshold
    pit.set_silence(-60)  # Was -40
    pit.set_tolerance(0.8)

    # aubio onset
    ons = aubio.onset(args.onset, win, hop, sr)
    # MODIFIED: Loosened silence threshold
    ons.set_silence(-60)  # Was -40
    ons.set_minioi_s(max(0.01, args.minioi_ms/1000.0))

    # pass 1: gather per-frame times + pitched frames and onsets
    times, hz_list, confs, onsets = [], [], [], []
    for i in range(0, len(x)-hop, hop):
        frame = x[i:i+hop]
        t = i / sr
        if ons(frame).item() > 0:
            onsets.append(t)
        hz = float(pit(frame).item())
        conf = float(pit.get_confidence())
        if hz > 0 and args.min_hz <= hz <= args.max_hz and conf >= args.conf:
            times.append(t)
            hz_list.append(hz)
            confs.append(conf)

    # ensure we start and end with boundaries
    if not onsets or onsets[0] > 0.05:
        onsets = [0.0] + onsets
    onsets.append((len(x)-1)/sr)

    print("time\thz\tnote")
    if not times:
        print("--- No notes detected ---")
        print("Parameters might be too strict, even with normalization.")
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
    last_valid = None  # remember last non-None pitch to keep octave stable across gaps
    for m in midi_s:
        if m is None:
            corrected.append(None)
            continue
        if last_valid is None:
            corrected.append(m)
            last_valid = m
            continue
        cand = m
        while cand - last_valid > 6:
            cand -= 12
        while last_valid - cand > 6:
            cand += 12
        corrected.append(cand)
        last_valid = cand

    # Track a preferred register (octave placement) for each pitch class so that
    # the same note name always prints in the same octave even if the tracker
    # locks onto a harmonic momentarily.
    register_by_pc = {}
    register_width = 6.0  # snap back if we wander farther than this many semitones
    register_blend = 0.2  # light smoothing so preferred registers can adapt

    # one note per onset-to-onset window
    tail = args.tail_ms/1000.0
    for k in range(len(onsets)-1):
        t0, t1 = onsets[k], onsets[k+1]
        rep = median_midi_in(times, corrected, t0 + tail, t1)
        if rep is None:  # fallback: right at onset
            rep = median_midi_in(times, corrected, t0, min(t1, t0 + 0.25))
        if rep is None:
            continue

        rounded = int(round(rep))
        pitch_class = rounded % 12
        anchor = register_by_pc.get(pitch_class)
        if anchor is not None:
            adj = rep
            while adj - anchor > register_width:
                adj -= 12
            while anchor - adj > register_width:
                adj += 12
            rep = adj
            register_by_pc[pitch_class] = (
                1 - register_blend) * anchor + register_blend * rep
        else:
            register_by_pc[pitch_class] = rep

        print(f"{t0:.3f}\t{midi_to_hz(rep):.1f}\t{midi_to_note(rep)}")


def _defaults_when_no_args():
    if len(sys.argv) == 1:  # no CLI args → use your presets
        print("--- No command-line arguments detected. Using internal defaults. ---")
        sys.argv += [
            # MODIFIED: Changed path and parameters
            DEFAULT_AUDIO_FILE,
            "--onset", "hfc",
            "--algo", "yin",
            "--conf", "0.5",  # Loosened confidence
            "--minioi-ms", "60",
            "--tail-ms", "60",
            "--smooth-ms", "150",
            "--min-hz", "20",  # Widened range
            "--max-hz", "4200",  # Widened range
        ]


if __name__ == "__main__":
    _defaults_when_no_args()
    main()
