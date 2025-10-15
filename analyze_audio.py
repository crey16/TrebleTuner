#!/usr/bin/env python3
from dataclasses import dataclass, asdict
import pathlib, sys, math
import numpy as np
import soundfile as sf
import aubio
import json

# Define the dataclass for storing note information
@dataclass
class DetectedNote:
    time: float
    hz: float
    name: str

# Helper functions
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

def analyze_audio(audio_file_path):
    """Analyzes an audio file and returns a list of DetectedNote objects."""
    # Configuration
    algo = "yin"; onset = "hfc"; win = 2048; hop = 256; conf = 0.55
    min_hz = 40.0; max_hz = 2000.0; smooth_ms = 150.0; minioi_ms = 80.0
    tail_ms = 40.0
    
    path = pathlib.Path(audio_file_path).expanduser()
    if not path.is_file():
        print(f"Error: Audio file not found at '{path}'")
        return []

    print(f"Processing audio file: {path.name}...")
    detected_notes = []

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim == 2: x = x.mean(axis=1).astype(np.float32)

    # --- Analysis logic ---
    pit = aubio.pitch(algo, win, hop, sr); pit.set_unit("Hz")
    pit.set_silence(-40); pit.set_tolerance(0.8)
    ons = aubio.onset(onset, win, hop, sr)
    ons.set_silence(-40); ons.set_minioi_s(max(0.01, minioi_ms/1000.0))
    times, hz_list, onsets = [], [], []
    for i in range(0, len(x)-hop, hop):
        frame = x[i:i+hop]; t = i / sr
        if ons(frame).item() > 0: onsets.append(t)
        hz = float(pit(frame).item()); conf_val = float(pit.get_confidence())
        if hz > 0 and min_hz <= hz <= max_hz and conf_val >= conf:
            times.append(t); hz_list.append(hz)
    if not onsets or onsets[0] > 0.05: onsets = [0.0] + onsets
    onsets.append((len(x)-1)/sr)
    if not times: return []
    midi_raw = [hz_to_midi(hz) for hz in hz_list]
    W = max(1, int(round((smooth_ms/1000.0) * sr / hop)))
    midi_s = []
    for k in range(len(midi_raw)):
        lo, hi = max(0, k-W//2), min(len(midi_raw), k+W//2+1)
        vals = [m for m in midi_raw[lo:hi] if m is not None]
        midi_s.append(float(np.median(vals)) if vals else None)
    corrected = []
    for m in midi_s:
        if not corrected or m is None or corrected[-1] is None: corrected.append(m); continue
        prev, cand = corrected[-1], m
        if abs(cand - prev) > 6:
            while cand - prev > 6: cand -= 12
            while prev - cand > 6: cand += 12
        corrected.append(cand)
    tail = tail_ms/1000.0
    for k in range(len(onsets)-1):
        t0, t1 = onsets[k], onsets[k+1]
        rep = median_midi_in(times, corrected, t0 + tail, t1)
        if rep is None: rep = median_midi_in(times, corrected, t0, min(t1, t0 + 0.25))
        if rep is None: continue
        
        note_obj = DetectedNote(time=round(t0, 3), hz=round(midi_to_hz(rep), 1), name=midi_to_note(rep))
        detected_notes.append(note_obj)
    
    print(f"Finished processing. Found {len(detected_notes)} notes.\n")
    return detected_notes

# --- NEW FUNCTION ---
def normalize_note_times(notes: list[DetectedNote]) -> list[DetectedNote]:
    """Adjusts note timestamps so the first note starts at time 0.0."""
    if not notes:
        return []

    # Get the time of the first note to use as the offset
    offset = notes[0].time
    
    # Create a new list with the adjusted times
    normalized_notes = [
        DetectedNote(time=round(note.time - offset, 3), hz=note.hz, name=note.name)
        for note in notes
    ]
    return normalized_notes
# --- END NEW FUNCTION ---

def compare_analyses(notes1, notes2):
    """A simple function to compare two lists of detected notes."""
    print("--- Comparison ---")
    num_to_check = min(len(notes1), len(notes2))
    matches = 0
    
    for i in range(num_to_check):
        note1 = notes1[i]
        note2 = notes2[i]
        
        # Check if the note names are the same and times are close (e.g., within 100ms)
        if note1.name == note2.name and abs(note1.time - note2.time) < 0.1:
            matches += 1
            print(f"Match at index {i}: Note '{note1.name}' at similar relative times (~{note1.time:.2f}s)")
            
    similarity_score = (matches / num_to_check) * 100 if num_to_check > 0 else 0
    print(f"\nFound {matches} matching notes in sequence out of {num_to_check} checked.")
    print(f"Sequence similarity: {similarity_score:.2f}%")


if __name__ == "__main__":
    # Specify the two audio files to compare
    audio_file_1 = "audios/audio2.wav"
    audio_file_2 = "audios/audio7.wav"

    # Run the analysis on both files
    notes_from_audio1 = analyze_audio(audio_file_1)
    notes_from_audio2 = analyze_audio(audio_file_2)
    
    # --- MODIFIED SECTION: Normalize the results before comparing ---
    print("Normalizing timestamps for comparison...")
    normalized_notes1 = normalize_note_times(notes_from_audio1)
    normalized_notes2 = normalize_note_times(notes_from_audio2)
    print("Normalization complete.\n")
    
    # Optional: Print the normalized results to verify
    print("--- Normalized Results for Audio 1 ---")
    print(json.dumps([asdict(n) for n in normalized_notes1], indent=2))
    
    # Compare the two *normalized* lists of results
    if normalized_notes1 and normalized_notes2:
        compare_analyses(normalized_notes1, normalized_notes2)