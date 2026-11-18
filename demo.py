"""Solve the twelve image family measured on this machine and report every plan.

Run with:
    python demo.py

It prints the family footprint under each strategy, writes the Dockerfiles and the
layer report for the exact plan into plan_out/, records the numbers in
docs/results.json, and renders docs/before_after.png.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from pullplan import corpus, emit, plan
from pullplan.family import FAMILY

ROOT = Path(__file__).resolve().parent
SIZES = ROOT / "data" / "dist_sizes.json"


def mb(value: int) -> float:
    return round(value / 1e6, 1)


def main() -> int:
    if not SIZES.exists():
        print("no measurements yet: run python scripts/measure_sizes.py first")
        return 1

    index = corpus.load_sizes(SIZES)
    inst = plan.build_instance(FAMILY, index, corpus.closure)
    b = plan.intersection_weights(inst)
    sigs = plan.signature_weights(inst)

    print(f"family of {inst.n} images over {len(inst.weight)} distributions, "
          f"{len(sigs)} distinct co-occurrence signatures")
    print()
    print(f"{'image':20s} {'packages':>9s} {'own footprint':>14s}")
    for i, name in enumerate(inst.images):
        print(f"{name:20s} {len(inst.sets[name]):9d} {mb(b[1 << i]):13.1f} MB")
    print()

    independent = plan.independent_cost(inst, b)
    one_base, one_plan = plan.best_k_bases(inst, b, 1)
    two_base, _two_plan = plan.best_k_bases(inst, b, 2)
    greedy, _merges = plan.greedy_hierarchy(inst, b)
    optimal, tree = plan.optimal_hierarchy(inst, b)
    floor = plan.flat_floor(inst)

    tree = plan.collapse(plan.assign_content(inst, tree))
    assert plan.footprint(tree) == optimal, "layer bytes must equal the solved cost"
    bases = plan.base_images(tree)

    strategies = [
        ("independent", "independent\nimages", independent),
        ("one_base", "1 shared\nbase (exact)", one_base),
        ("two_bases", "2 shared\nbases (exact)", two_base),
        ("greedy", "greedy\nhierarchy", greedy),
        ("optimal", "pullplan\nexact plan", optimal),
        ("floor", "no-FROM-chain\nfloor", floor),
    ]

    print(f"{'strategy':34s} {'footprint':>12s} {'saved':>8s}")
    for _key, label, value in strategies:
        flat = label.replace("\n", " ")
        print(f"{flat:34s} {mb(value):11.1f} MB {100 * (1 - value / independent):7.1f}%")
    print()
    print(f"the exact plan uses {len(bases)} shared base images:")
    print(emit.summarize_bases(inst, tree, limit=len(bases)))
    print()
    print("hierarchy:")
    print(emit.ascii_tree(inst, tree))

    out = emit.write_plan(inst, tree, ROOT / "plan_out")
    print()
    print(f"wrote {len(list(out.glob('Dockerfile.*')))} Dockerfiles "
          f"and the layer report to {out.name}/")

    results = {
        "generated_on": dt.date.today().isoformat(),
        "images": inst.n,
        "packages": len(inst.weight),
        "signatures": len(sigs),
        "one_base_images": inst.members(one_plan[0]) if one_plan else [],
        "strategies": [
            {"key": key, "label": label, "bytes": value} for key, label, value in strategies
        ],
        "bases": [
            {
                "name": emit.tag_for(inst, node, emit.tag_index(tree)),
                "images": len(inst.members(node.mask)),
                "members": inst.members(node.mask),
                "packages": len(node.content),
                "bytes": node.bytes,
            }
            for node in bases
        ],
    }
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    import make_chart

    make_chart.render(results, docs / "before_after.png")
    print(f"wrote {docs.name}/results.json and {docs.name}/before_after.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
