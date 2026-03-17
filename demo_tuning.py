"""demo_tuning.py
-----------------
Audible tuning comparison: 12-TET vs Just Intonation using the
Salamander Grand Piano samples.

Plays several demonstrations back-to-back:
  1. Major third  — 12-TET (400 ¢)  vs  JI 5/4  (386 ¢)
  2. Minor seventh — 12-TET (1000 ¢) vs  JI 7/4  (969 ¢)  ← the "blue note"
  3. Perfect fifth — 12-TET (700 ¢)  vs  JI 3/2  (702 ¢)  (nearly the same)
  4. Full major chord in 12-TET, then pure JI (1 : 5/4 : 3/2)
  5. A short melodic phrase using 5-limit JI ratios

Run:
    python demo_tuning.py
    python demo_tuning.py --synth piano_sound   # use algorithmic piano instead
"""

import argparse
import numpy as np
import wave_synth as ws

FS   = ws.FS
ROOT = 261.63   # C4 in Hz  (standard 12-TET)


# ── helpers ───────────────────────────────────────────────────────────────

def _hz(ratio: float) -> float:
    """Frequency from ROOT * ratio."""
    return ROOT * ratio


def _tet(semitones: float) -> float:
    """12-TET frequency: ROOT * 2^(semitones/12)."""
    return ROOT * 2 ** (semitones / 12)


def note(freq: float, dur: float, synth_func) -> np.ndarray:
    return synth_func(freq, dur + 1.5)   # +1.5 s tail for natural decay


def chord(freqs: list, dur: float, synth_func) -> np.ndarray:
    """Mix several simultaneous notes."""
    n = int((dur + 2.5) * FS)
    buf = np.zeros(n, dtype=np.float64)
    for f in freqs:
        s = synth_func(f, dur + 2.0)
        buf[:len(s)] += s
    peak = np.max(np.abs(buf))
    if peak > 1e-8:
        buf *= 0.85 / peak
    return buf.astype(np.float32)


def silence(dur: float) -> np.ndarray:
    return np.zeros(int(dur * FS), dtype=np.float32)


def concat(*arrays) -> np.ndarray:
    return np.concatenate(arrays)


def play(audio: np.ndarray) -> None:
    if not ws._HAS_SD:
        raise RuntimeError("pip install sounddevice")
    import sounddevice as sd
    rate = ws._native_fs()
    if rate != FS:
        from scipy.signal import resample_poly
        from fractions import Fraction
        frac = Fraction(rate / FS).limit_denominator(500)
        audio = resample_poly(audio, frac.numerator, frac.denominator).astype(np.float32)
    sd.play(audio, rate, blocksize=2048)
    sd.wait()


# ── demo sections ─────────────────────────────────────────────────────────

def demo_interval(name: str, tet_semitones: float, ji_ratio: float,
                  synth_func, dur: float = 2.5) -> np.ndarray:
    tet_hz = _tet(tet_semitones)
    ji_hz  = _hz(ji_ratio)
    cents_tet = 1200 * np.log2(tet_hz / ROOT)
    cents_ji  = 1200 * np.log2(ji_hz  / ROOT)
    diff      = cents_tet - cents_ji

    print(f"\n  {name}")
    print(f"    12-TET : {tet_hz:.2f} Hz  ({cents_tet:.1f} ¢)")
    print(f"    JI     : {ji_hz:.2f} Hz  ({cents_ji:.1f} ¢)  ratio = {ji_ratio}")
    print(f"    diff   : {diff:+.1f} ¢")

    root_note = note(ROOT, dur, synth_func)
    tet_note  = note(tet_hz, dur, synth_func)
    ji_note   = note(ji_hz,  dur, synth_func)

    def dyad(interval_note):
        n = max(len(root_note), len(interval_note))
        buf = np.zeros(n, dtype=np.float32)
        buf[:len(root_note)]    += root_note  * 0.5
        buf[:len(interval_note)] += interval_note * 0.5
        return buf

    return concat(
        dyad(tet_note), silence(0.6),
        dyad(ji_note),  silence(1.0),
    )


def demo_chord_comparison(synth_func, dur: float = 3.5) -> np.ndarray:
    print("\n  Major chord: 12-TET  then  JI  (1 : 5/4 : 3/2)")
    tet_chord = chord([ROOT, _tet(4), _tet(7)], dur, synth_func)
    ji_chord  = chord([ROOT, _hz(5/4), _hz(3/2)], dur, synth_func)
    return concat(tet_chord, silence(0.8), ji_chord, silence(1.2))


def demo_melody(synth_func) -> np.ndarray:
    """Short phrase using 5-limit JI ratios."""
    print("\n  JI melody  (C D E F G A B C  using 5-limit ratios)")
    # 5-limit JI major scale ratios
    ratios = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2]
    names  = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C\'']
    durs   = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.2]
    notes  = []
    for r, d, n in zip(ratios, durs, names):
        hz = _hz(r)
        cents = 1200 * np.log2(r) if r > 0 else 0
        tet_hz = ROOT * 2 ** (round(1200 * np.log2(r) / 100) / 12) if r > 0 else ROOT
        deviation = 1200 * np.log2(hz / tet_hz) if r > 0 else 0
        print(f"    {n:4s}  {hz:.2f} Hz  ({deviation:+.1f} ¢ from 12-TET)")
        s = synth_func(hz, d + 1.2)
        n_samples = int(d * FS)
        notes.append(s[:n_samples] if len(s) >= n_samples
                     else np.pad(s, (0, n_samples - len(s))))
    return concat(*notes, silence(1.5))


# ── main ──────────────────────────────────────────────────────────────────

def main(synth_name: str = 'sampled_piano') -> None:
    synth_func = ws._get_synth(synth_name)
    using_samples = (synth_func is ws._sampled_piano)

    print("=" * 58)
    print("  Tuning Demo — Salamander Grand Piano")
    print(f"  Synth : {synth_name}"
          + ("  (samples loaded)" if using_samples else
             "  (algorithmic fallback — run download_salamander.py)"))
    print(f"  Root  : C4 = {ROOT} Hz")
    print("=" * 58)
    print("\nFormat: [12-TET dyad]  pause  [JI dyad]")
    print("Listen for beating in the 12-TET versions.")

    sections = [
        demo_interval("Major third   5/4  (12-TET: 400¢, JI: 386¢, diff −14¢)",
                      4, 5/4, synth_func),
        demo_interval("Major sixth   5/3  (12-TET: 900¢, JI: 884¢, diff −16¢)",
                      9, 5/3, synth_func),
        demo_interval("Minor seventh 7/4  (12-TET: 1000¢, JI: 969¢, diff −31¢)",
                      10, 7/4, synth_func),
        demo_interval("Perfect fifth 3/2  (12-TET: 700¢, JI: 702¢, diff +2¢)",
                      7, 3/2, synth_func),
        demo_chord_comparison(synth_func),
        demo_melody(synth_func),
    ]

    full = concat(*sections)

    print(f"\nRendered {len(full)/FS:.1f}s of audio. Playing now…\n")
    play(full)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", default="sampled_piano",
                        help="synth name (default: sampled_piano)")
    args = parser.parse_args()
    main(args.synth)
