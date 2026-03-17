"""download_salamander.py
------------------------
Downloads the Salamander Grand Piano v8 samples (one per sampled note)
from the official freepats.zenvoid.org archive.

The archive is 707 MB, but we stream it and only write the v8 FLAC files
to disk — roughly 10 MB total for all 30 notes.

Usage
-----
    python download_salamander.py            # download v8 (default)
    python download_salamander.py --vel 12   # download a different velocity layer
    python download_salamander.py --check    # just list what's missing
"""

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

ARCHIVE_URL = (
    "http://freepats.zenvoid.org/Piano/SalamanderGrandPiano/"
    "SalamanderGrandPiano-SFZ+FLAC-V3+20200602.tar.gz"
)
DEST_DIR = Path(__file__).parent / "samples" / "salamander"

# All 30 sampled note names (every minor third A0 → C8, Salamander naming)
SAMPLE_STEMS = [
    "A0",  "C1",  "Ds1", "Fs1",
    "A1",  "C2",  "Ds2", "Fs2",
    "A2",  "C3",  "Ds3", "Fs3",
    "A3",  "C4",  "Ds4", "Fs4",
    "A4",  "C5",  "Ds5", "Fs5",
    "A5",  "C6",  "Ds6", "Fs6",
    "A6",  "C7",  "Ds7", "Fs7",
    "A7",  "C8",
]


def _wanted(velocity: int) -> set:
    return {f"{stem}v{velocity}.flac" for stem in SAMPLE_STEMS}


def _missing(velocity: int) -> list:
    want = _wanted(velocity)
    present = {p.name for p in DEST_DIR.glob("*.flac")} if DEST_DIR.exists() else set()
    return sorted(want - present)


class _ProgressReader:
    """Wraps a URL response to print a download-progress line."""

    def __init__(self, response):
        self._r    = response
        self._read = 0
        try:
            self._total = int(response.headers.get("Content-Length", 0))
        except Exception:
            self._total = 0

    def read(self, size: int = -1) -> bytes:
        data = self._r.read(size)
        self._read += len(data)
        if self._total:
            pct  = self._read / self._total * 100
            done = int(pct / 2)
            bar  = "#" * done + "-" * (50 - done)
            mb   = self._read / 1_000_000
            tot  = self._total / 1_000_000
            print(f"\r  [{bar}] {mb:.0f}/{tot:.0f} MB  ({pct:.1f}%)", end="", flush=True)
        else:
            print(f"\r  {self._read / 1_000_000:.0f} MB streamed", end="", flush=True)
        return data


def download(velocity: int = 8, force: bool = False) -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    missing = _missing(velocity) if not force else sorted(_wanted(velocity))
    if not missing:
        print(f"All v{velocity} samples already present in {DEST_DIR}")
        return

    print(f"Need {len(missing)} file(s):  {', '.join(missing[:4])}"
          + (" …" if len(missing) > 4 else ""))
    print(f"\nStreaming archive from freepats.zenvoid.org (~707 MB).")
    print("Only the selected samples (~10 MB) will be written to disk.\n")

    want   = set(missing)
    found  = 0
    req    = urllib.request.urlopen(ARCHIVE_URL)
    reader = _ProgressReader(req)

    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                fname = Path(member.name).name
                if fname not in want:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                out = DEST_DIR / fname
                out.write_bytes(f.read())
                found += 1
                print(f"\n  ✓ {fname}  ({found}/{len(want)})")
                want.discard(fname)
                if not want:
                    break
    finally:
        req.close()
        print()   # newline after progress bar

    if found:
        print(f"\nSaved {found} file(s) to {DEST_DIR}")
    if want:
        print(f"\nWarning: {len(want)} file(s) not found in archive: {sorted(want)}")


def check(velocity: int = 8) -> None:
    missing = _missing(velocity)
    present = len(SAMPLE_STEMS) - len(missing)
    print(f"v{velocity} samples in {DEST_DIR}:")
    print(f"  present : {present}/{len(SAMPLE_STEMS)}")
    if missing:
        print(f"  missing : {', '.join(missing)}")
    else:
        print("  all present — ready to use _sampled_piano()")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Salamander Grand Piano samples")
    parser.add_argument("--vel",   type=int, default=8,
                        help="velocity layer to download (1–16, default 8)")
    parser.add_argument("--check", action="store_true",
                        help="list missing files without downloading")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if file already exists")
    args = parser.parse_args()

    if args.check:
        check(args.vel)
    else:
        download(args.vel, force=args.force)
