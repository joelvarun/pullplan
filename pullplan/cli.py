"""Command line entry point: measure a host, then plan a family.

    python -m pullplan measure --out data/dist_sizes.json
    python -m pullplan plan --family my_family.json --out plan_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import corpus, emit, plan
from .family import FAMILY


def load_family(path: str | None) -> dict[str, list[str]]:
    if path is None:
        return FAMILY
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_measure(args: argparse.Namespace) -> int:
    family = load_family(args.family)
    index = corpus.discover()
    needed: set[str] = set()
    for roots in family.values():
        got, missing = corpus.closure(roots, index)
        needed |= got
        for name in sorted(missing):
            print(f"not installed, skipped: {name}")
    print(f"measuring {len(needed)} distributions")
    measured = {}
    for i, name in enumerate(sorted(needed), 1):
        dist = corpus.measure(index[name], sample_cap=args.cap << 20)
        measured[name] = dist
        print(
            f"[{i:3d}/{len(needed)}] {name:28s} "
            f"tar={dist.tar_bytes / 1e6:8.2f} MB gzip={dist.gzip_bytes / 1e6:8.2f} MB"
        )
    corpus.save_sizes(args.out, measured)
    print(f"wrote {args.out}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    sizes = Path(args.sizes)
    if not sizes.exists():
        print(f"{sizes} not found: run python -m pullplan measure first")
        return 1
    family = load_family(args.family)
    index = corpus.load_sizes(sizes)
    inst = plan.build_instance(family, index, corpus.closure)
    b = plan.intersection_weights(inst)

    independent = plan.independent_cost(inst, b)
    if args.bases:
        cost, masks = plan.best_k_bases(inst, b, args.bases)
        label = "base image" if args.bases == 1 else "base images"
        print(f"best plan with at most {args.bases} {label}: {cost / 1e6:.1f} MB")
        for mask in masks:
            print(f"  base over: {', '.join(inst.members(mask))}")
        print(f"independent images: {independent / 1e6:.1f} MB")
        return 0

    cost, tree = plan.optimal_hierarchy(inst, b)
    tree = plan.collapse(plan.assign_content(inst, tree))
    bases = plan.base_images(tree)
    print(f"independent images : {independent / 1e6:9.1f} MB")
    print(
        f"exact hierarchy    : {cost / 1e6:9.1f} MB "
        f"({100 * (1 - cost / independent):.1f} percent less, {len(bases)} base images)"
    )
    print(f"non-laminar floor  : {plan.flat_floor(inst) / 1e6:9.1f} MB")
    print()
    print(emit.ascii_tree(inst, tree))
    out = emit.write_plan(inst, tree, args.out)
    print(f"\nwrote the plan to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pullplan", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="measure installed distributions")
    measure.add_argument("--family", help="JSON file mapping image name to packages")
    measure.add_argument("--out", default="data/dist_sizes.json")
    measure.add_argument(
        "--cap", type=int, default=16, help="per distribution sample cap in MB"
    )
    measure.set_defaults(func=cmd_measure)

    solve = sub.add_parser("plan", help="solve the base image hierarchy")
    solve.add_argument("--family", help="JSON file mapping image name to packages")
    solve.add_argument("--sizes", default="data/dist_sizes.json")
    solve.add_argument("--out", default="plan_out")
    solve.add_argument(
        "--bases",
        type=int,
        choices=(1, 2),
        help="restrict the plan to at most this many base images",
    )
    solve.set_defaults(func=cmd_plan)

    args = parser.parse_args(argv)
    return args.func(args)
