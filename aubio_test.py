#!/usr/bin/env python3
# notes_by_pitch_changes.py
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(pathlib.Path.home()/ "Downloads" / "audio.wav"))
    ap.add_argument("--win", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)          # finer time resolution
    ap.add_argument("--algo", type=str, default="yinfft", choices=["yin","yinfft","yinfast","mcomb","schmitt","fcomb","default"])
    ap.add_argument("--min-hz", type=float, default=40.0)
    ap.add_argument("--max-hz", type=float, default=2000.0)
    ap.add_argument("--conf", type=float, default=0.40)      # loose gate so we don't miss quiet notes
    ap.add_argument("--smooth-ms", type=float, default=120.0)# median smoothing window
    ap.add_argument("--min-note-ms", type=float, default=80.0)# ignore super-short blips
    args = ap.parse_args()

    path = pathlib.Path(args.path).expanduser()
    if not path.exists(): sys.exit(f"File not found: {path}")

    # load mono
    sig, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sig.ndim == 2:
        sig = sig.mean(axis=1).astype(np.float32)

    hop, win = args.hop, args.win
    pitch_o = aubio.pitch(args.algo, win, hop, sr); pitch_o.set_unit("Hz")
    pitch_o.set_silence(-40); pitch_o.set_tolerance(0.8)

    times, hz_list, confs = [], [], []
    for i in range(0, len(sig)-hop, hop):
        frame = sig[i:i+hop].astype(np.float32)
        hz = float(pitch_o(frame).item())
        conf = float(pitch_o.get_confidence())
        t = i / sr
        if args.min_hz <= hz <= args.max_hz and conf >= args.conf and hz > 0:
            times.append(t); hz_list.append(hz); confs.append(conf)

    print("time\thz\tnote")
    if not times:
        return

    # median smoothing in MIDI space
    midi_raw = [hz_to_midi(h) for h in hz_list]
    W = max(1, int(round((args.smooth_ms/1000.0) * sr / hop)))
    midi_smooth = []
    for k in range(len(midi_raw)):
        lo, hi = max(0, k-W//2), min(len(midi_raw), k+W//2+1)
        window = [m for m in midi_raw[lo:hi] if m is not None]
        midi_smooth.append(float(np.median(window)) if window else None)

    # light octave correction
    corrected = []
    for m in midi_smooth:
        if not corrected or m is None or corrected[-1] is None:
            corrected.append(m); continue
        prev, cand = corrected[-1], m
        if abs(cand - prev) > 6:
            while cand - prev > 6: cand -= 12
            while prev - cand > 6: cand += 12
        corrected.append(cand)

    # snap to nearest semitone and collapse runs into “notes”
    snapped = [None if m is None else int(round(m)) for m in corrected]
    min_frames = max(1, int(round((args.min_note_ms/1000.0) * sr / hop)))

    start_idx = 0
    while start_idx < len(snapped):
        # skip unvoiced
        if snapped[start_idx] is None:
            start_idx += 1
            continue
        note_m = snapped[start_idx]
        end_idx = start_idx + 1
        while end_idx < len(snapped) and snapped[end_idx] == note_m:
            end_idx += 1
        # only accept segments that last long enough
        if end_idx - start_idx >= min_frames:
            seg_times = times[start_idx:end_idx]
            seg_midi  = [corrected[i] for i in range(start_idx, end_idx) if corrected[i] is not None]
            if seg_times and seg_midi:
                t0 = seg_times[0]
                hz_med = float(np.median([midi_to_hz(m) for m in seg_midi]))
                print(f"{t0:.3f}\t{hz_med:.1f}\t{midi_to_note(note_m)}")
        start_idx = end_idx

if __name__ == "__main__":
    main()
