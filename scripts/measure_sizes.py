"""Measure real tar and gzip sizes for every distribution the demo family needs.

Run with:
    python scripts/measure_sizes.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pullplan import corpus  # noqa: E402
from pullplan.family import FAMILY  # noqa: E402


def main() -> int:
    index = corpus.discover()
    needed: set[str] = set()
    for app, roots in FAMILY.items():
        got, missing = corpus.closure(roots, index)
        needed |= got
        note = ", ".join(sorted(missing)) if missing else "-"
        print(f"{app:18s} {len(got):3d} dists  not installed: {note}", flush=True)

    print(f"\nmeasuring {len(needed)} distributions", flush=True)
    start = time.perf_counter()
    measured = {}
    for i, name in enumerate(sorted(needed), 1):
        tick = time.perf_counter()
        dist = corpus.measure(index[name])
        measured[name] = dist
        print(
            f"[{i:3d}/{len(needed)}] {name:28s} tar={dist.tar_bytes / 1e6:8.2f} MB "
            f"gzip={dist.gzip_bytes / 1e6:8.2f} MB ratio={dist.ratio:4.2f} "
            f"files={dist.file_count:5d} {'sampled' if dist.sampled else '':7s} "
            f"({time.perf_counter() - tick:.1f}s)",
            flush=True,
        )
    out = ROOT / "data" / "dist_sizes.json"
    out.parent.mkdir(exist_ok=True)
    corpus.save_sizes(out, measured)
    print(f"total {time.perf_counter() - start:.1f}s, wrote {out.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
