"""download_vsco.py
------------------
Downloads VSCO 2 Community Edition (CC0) orchestral samples from GitHub for:

    xylophone  — 8 files  (G3–C7, every 4th/5th)
    organ      — 21 files (C1–C6, every 3 semitones)
    chimes     — 4 files  (tubular bells, C4–F5)
    flute      — sustained non-vibrato articulation
    violin     — arco vibrato (bowed) articulation

All samples are CC0 (public domain).
Source: https://github.com/sgossner/VSCO-2-CE

Usage
-----
    python download_vsco.py                        # all instruments
    python download_vsco.py xylophone chimes       # specific instruments
    python download_vsco.py --check                # show download status
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from urllib.parse import quote as _quote

REPO     = 'sgossner/VSCO-2-CE'
BASE_API = f'https://api.github.com/repos/{REPO}/contents'
DEST_DIR = Path(__file__).parent / 'samples' / 'vsco'
_HEADERS = {
    'User-Agent': 'ScoreBuilder-sample-downloader/1.0',
    'Accept':     'application/vnd.github.v3+json',
}


# ── Instrument definitions ─────────────────────────────────────────────────
#
# Each entry:  name → (github_path, local_subdir, keep_predicate)
#
# keep_predicate(filename) → True  = download this file

def _xylophone_keep(n):   return n.endswith('.wav')
def _organ_keep(n):       return n.startswith('Rode_Man3Open') and n.endswith('.wav')
def _chimes_keep(n):      return n.startswith('TB_hit') and n.endswith('.wav')
def _flute_keep(n):       return n.endswith('.wav')
def _violin_keep(n):      return n.endswith('.wav')
# Concert percussion — rr1 only (fixed-pitch instruments only load one sample
# per pitch level, so rr2 would never be used)
def _snare_keep(n):       return n.startswith('Snare2-HitSN') and 'rr1' in n and n.endswith('.wav')
def _bass_drum_keep(n):   return n.startswith('BDrumNewhit')  and 'rr1' in n and n.endswith('.wav')
def _timpani_keep(n):     return n.startswith('Timpani')      and '_Hit_' in n and 'rr1' in n and n.endswith('.wav')
def _res_tom_keep(n):     return n.startswith('Snare2-HitNS') and 'rr1' in n and n.endswith('.wav')

INSTRUMENTS = {
    'xylophone':    ('Percussion/Xylo',               'xylophone',    _xylophone_keep),
    'organ':        ('Keys/Organ/Loud',               'organ',        _organ_keep),
    'chimes':       ('Percussion',                    'chimes',       _chimes_keep),
    'flute':        ('Woodwinds/Flute/susNV',         'flute',        _flute_keep),
    'violin':       ('Strings/Solo Violin/Arco Vib',  'violin',       _violin_keep),
    # Concert percussion
    'snare':        ('Percussion',                    'snare',        _snare_keep),
    'bass_drum':    ('Percussion',                    'bass_drum',    _bass_drum_keep),
    'timpani':      ('Percussion/Timpani',             'timpani',      _timpani_keep),
    'resonant_tom': ('Percussion',                    'resonant_tom', _res_tom_keep),
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _api_get(gh_path: str) -> list:
    """Call GitHub Contents API, return list of file/dir entries."""
    url = f"{BASE_API}/{_quote(gh_path)}"
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    if isinstance(data, dict) and 'message' in data:
        raise RuntimeError(f"GitHub API error: {data['message']}")
    return data


def _download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={'User-Agent': _HEADERS['User-Agent']})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, 'wb') as f:
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            f.write(chunk)


# ── Per-instrument download ────────────────────────────────────────────────

def download_instrument(name: str) -> None:
    gh_path, local_dir, keep = INSTRUMENTS[name]
    dest = DEST_DIR / local_dir
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n[{name}]  listing {gh_path} …", flush=True)
    try:
        entries = _api_get(gh_path)
    except Exception as e:
        print(f"  ERROR listing files: {e}")
        return

    files = [e for e in entries
             if e.get('type') == 'file' and keep(e['name'])]
    print(f"  Found {len(files)} matching file(s)")

    already = {p.name for p in dest.glob('*.wav')}
    todo    = [e for e in files if e['name'] not in already]
    skip    = len(files) - len(todo)
    if skip:
        print(f"  Skipping {skip} already-downloaded file(s)")
    if not todo:
        print(f"  All present — nothing to download")
        return

    for i, entry in enumerate(todo, 1):
        fname = entry['name']
        url   = entry['download_url']
        size  = entry.get('size', 0)
        sz_kb = f"{size // 1024} KB" if size else "?"
        print(f"  ({i}/{len(todo)}) {fname}  [{sz_kb}]", end=' ', flush=True)
        try:
            _download_file(url, dest / fname)
            print("✓")
        except Exception as e:
            print(f"✗  {e}")


# ── Status check ──────────────────────────────────────────────────────────

def check() -> None:
    print("VSCO 2 CE sample status:")
    for name, (_, local_dir, _) in INSTRUMENTS.items():
        d     = DEST_DIR / local_dir
        count = len(list(d.glob('*.wav'))) if d.exists() else 0
        badge = f"{count} file(s)" if count else "NOT downloaded"
        print(f"  {name:<12}  {badge}   ({d})")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download VSCO 2 Community Edition samples (CC0)")
    parser.add_argument(
        'instruments', nargs='*',
        metavar='INSTRUMENT',
        help=(f"instrument(s) to download: "
              f"{', '.join(INSTRUMENTS)} (default: all)"))
    parser.add_argument(
        '--check', action='store_true',
        help="show download status without downloading anything")
    args = parser.parse_args()

    if args.check:
        check()
        return

    unknown = [n for n in args.instruments if n not in INSTRUMENTS]
    if unknown:
        print(f"Unknown instrument(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(INSTRUMENTS)}")
        sys.exit(1)

    targets = list(INSTRUMENTS) if not args.instruments else args.instruments
    for name in targets:
        download_instrument(name)

    print("\nDone.")
    print("Run  python wave_synth.py <wave.json>  to use the new instruments.")


if __name__ == '__main__':
    main()
