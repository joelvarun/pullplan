"""How often does the greedy hierarchy miss the optimum, and by how much?

Draws random families from the distributions already measured on this host, solves
each one exactly, and compares against agglomerative merging. Run with:

    python scripts/greedy_gap.py
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pullplan import corpus, plan  # noqa: E402

TRIALS = 200
IMAGES = 9
SEED = 7


def main() -> int:
    sizes = ROOT / "data" / "dist_sizes.json"
    if not sizes.exists():
        print("no measurements yet: run python scripts/measure_sizes.py first")
        return 1
    index = corpus.load_sizes(sizes)
    pool = sorted(index)
    rng = random.Random(SEED)

    gaps = []
    savings = []
    for _ in range(TRIALS):
        family = {
            f"svc{i}": rng.sample(pool, rng.randint(2, 5)) for i in range(IMAGES)
        }
        inst = plan.build_instance(family, index, corpus.closure)
        b = plan.intersection_weights(inst)
        optimal, _tree = plan.optimal_hierarchy(inst, b)
        greedy, _merges = plan.greedy_hierarchy(inst, b)
        independent = plan.independent_cost(inst, b)
        savings.append(100 * (1 - optimal / independent))
        if greedy > optimal:
            gaps.append(100 * (greedy - optimal) / optimal)

    gaps.sort()
    summary = {
        "trials": TRIALS,
        "images_per_family": IMAGES,
        "seed": SEED,
        "greedy_suboptimal_count": len(gaps),
        "greedy_gap_percent_mean": round(sum(gaps) / len(gaps), 3) if gaps else 0.0,
        "greedy_gap_percent_max": round(gaps[-1], 3) if gaps else 0.0,
        "optimal_saving_percent_mean": round(sum(savings) / len(savings), 2),
        "optimal_saving_percent_min": round(min(savings), 2),
        "optimal_saving_percent_max": round(max(savings), 2),
    }
    for key, value in summary.items():
        print(f"{key:32s} {value}")
    out = ROOT / "docs" / "greedy_gap.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
