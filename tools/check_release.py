"""Fail if release placeholders are still present. Run before uploading:

    python tools/check_release.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARKERS = ("PLACEHOLDER", "<AUTHOR NAME>", "<YEAR>", "TODO(release)")
FILES = ["LICENSE", "NOTICE", "README.md", "pyproject.toml"]

bad = []
for name in FILES:
    for n, line in enumerate((ROOT / name).read_text().splitlines(), 1):
        if any(m in line for m in MARKERS):
            bad.append(f"{name}:{n}: {line.strip()}")
if "authors" not in (ROOT / "pyproject.toml").read_text():
    bad.append("pyproject.toml: no `authors` entry")
if bad:
    print("Not release-ready:\n  " + "\n  ".join(bad))
    sys.exit(1)
print("ok")
