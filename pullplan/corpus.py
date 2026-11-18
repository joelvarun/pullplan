"""Build image file sets from the Python distributions installed on this machine.

An "image" here is a container image for one service: its content is the transitive
dependency closure of the packages that service imports. Sizes are measured, not
declared: every file of every distribution is read, framed the way tar frames it,
and run through zlib at the level a registry uses for layer blobs.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

try:
    from packaging.requirements import Requirement
except ImportError:  # pragma: no cover - packaging ships with pip environments
    Requirement = None

import importlib.metadata as md

TAR_BLOCK = 512
GZIP_LEVEL = 6
DEFAULT_SAMPLE_CAP = 16 << 20


def normalize(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def tar_framed_size(file_size: int) -> int:
    """Bytes a tar archive spends on one member: header plus padded content."""
    return TAR_BLOCK + TAR_BLOCK * math.ceil(file_size / TAR_BLOCK)


@dataclass
class Distribution:
    name: str
    version: str
    requires: list[str] = field(default_factory=list)
    raw_bytes: int = 0
    tar_bytes: int = 0
    gzip_bytes: int = 0
    file_count: int = 0
    sampled: bool = False

    @property
    def ratio(self) -> float:
        return self.tar_bytes / self.gzip_bytes if self.gzip_bytes else 1.0


def _requires(dist) -> list[str]:
    out = []
    for spec in dist.metadata.get_all("Requires-Dist") or []:
        if Requirement is None:
            out.append(normalize(spec.split(";")[0].split("[")[0].split()[0]))
            continue
        try:
            req = Requirement(spec)
        except Exception:
            continue
        if req.marker is not None:
            # No extras requested: an image installs the plain dependency set.
            try:
                if not req.marker.evaluate({"extra": ""}):
                    continue
            except Exception:
                continue
        out.append(normalize(req.name))
    return out


def discover() -> dict[str, Distribution]:
    """Index every installed distribution by normalized name."""
    found: dict[str, Distribution] = {}
    for dist in md.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        key = normalize(name)
        if key in found:
            continue
        found[key] = Distribution(
            name=key,
            version=dist.version or "0",
            requires=_requires(dist),
        )
        found[key]._dist = dist  # type: ignore[attr-defined]
    return found


def measure(dist: Distribution, sample_cap: int = DEFAULT_SAMPLE_CAP) -> Distribution:
    """Read a distribution's files and record real tar and gzip sizes.

    Distributions larger than sample_cap are measured on a deterministic prefix of
    their file list and the compression ratio is extrapolated to the full tar size.
    """
    handle = getattr(dist, "_dist", None)
    if handle is None:
        return dist
    base = Path(str(handle.locate_file("")))
    entries = []
    for rel in handle.files or []:
        path = base / str(rel)
        try:
            size = path.stat().st_size
        except OSError:
            continue
        entries.append((str(rel), path, size))
    entries.sort()

    dist.file_count = len(entries)
    dist.raw_bytes = sum(size for _, _, size in entries)
    dist.tar_bytes = sum(tar_framed_size(size) for _, _, size in entries)

    comp = zlib.compressobj(GZIP_LEVEL, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    chunks: list[bytes] = []
    read_raw = 0
    read_tar = 0
    for _, path, size in entries:
        if read_raw >= sample_cap:
            dist.sampled = True
            break
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        chunks.append(comp.compress(bytes(TAR_BLOCK)))
        chunks.append(comp.compress(data))
        pad = (-len(data)) % TAR_BLOCK
        if pad:
            chunks.append(comp.compress(bytes(pad)))
        read_raw += len(data)
        read_tar += tar_framed_size(len(data))
    chunks.append(comp.flush())
    sampled_gzip = sum(len(c) for c in chunks)

    if dist.sampled and read_tar:
        dist.gzip_bytes = int(round(sampled_gzip * dist.tar_bytes / read_tar))
    else:
        dist.gzip_bytes = sampled_gzip
    return dist


def closure(roots, index: dict[str, Distribution]):
    """Transitive dependency closure of root package names."""
    seen: set[str] = set()
    missing: set[str] = set()
    stack = [normalize(r) for r in roots]
    while stack:
        key = stack.pop()
        if key in seen:
            continue
        dist = index.get(key)
        if dist is None:
            missing.add(key)
            continue
        seen.add(key)
        stack.extend(dist.requires)
    return seen, missing


def save_sizes(path, dists: dict[str, Distribution]) -> None:
    payload = {
        "gzip_level": GZIP_LEVEL,
        "tar_block": TAR_BLOCK,
        "distributions": {
            name: {
                "version": d.version,
                "files": d.file_count,
                "raw_bytes": d.raw_bytes,
                "tar_bytes": d.tar_bytes,
                "gzip_bytes": d.gzip_bytes,
                "sampled": d.sampled,
                "requires": sorted(d.requires),
            }
            for name, d in sorted(dists.items())
        },
    }
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")


def load_sizes(path) -> dict[str, Distribution]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, row in payload["distributions"].items():
        out[name] = Distribution(
            name=name,
            version=row["version"],
            requires=list(row["requires"]),
            raw_bytes=row["raw_bytes"],
            tar_bytes=row["tar_bytes"],
            gzip_bytes=row["gzip_bytes"],
            file_count=row["files"],
            sampled=row["sampled"],
        )
    return out
