"""Correctness tests for the hierarchy optimizer.

The dynamic program claims two things worth checking by exhaustion on small
instances: that it finds the best laminar plan, and that restricting the search to
binary hierarchies loses nothing against general k-ary ones.
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import pytest

from pullplan import corpus, plan

ROOT = Path(__file__).resolve().parents[1]


def instance_from_signatures(n: int, sig_weights: dict[int, int]) -> plan.Instance:
    """Build an instance directly from signature weights, one package per signature."""
    weight = {}
    signature = {}
    for k, (mask, w) in enumerate(sorted(sig_weights.items())):
        name = f"pkg{k}"
        weight[name] = w
        signature[name] = mask
    images = [f"img{i}" for i in range(n)]
    sets = {
        img: {d for d, m in signature.items() if m >> i & 1}
        for i, img in enumerate(images)
    }
    return plan.Instance(images=images, sets=sets, weight=weight, signature=signature)


def partitions(items: list[int]):
    """All set partitions of a list of leaf masks."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for smaller in partitions(rest):
        for i in range(len(smaller)):
            yield smaller[:i] + [[first] + smaller[i]] + smaller[i + 1:]
        yield [[first]] + smaller


def all_hierarchies(masks: list[int]):
    """Every laminar hierarchy over the given leaves, k-ary included."""
    if len(masks) == 1:
        yield masks[0], []
        return
    for part in partitions(masks):
        if len(part) < 2:
            continue
        options = [list(all_hierarchies(block)) for block in part]
        for combo in itertools.product(*options):
            total = 0
            internal: list[tuple[int, int]] = []
            for mask, sub in combo:
                total |= mask
                internal.extend(sub)
            yield total, internal + [(total, len(combo))]


def brute_force_cost(inst: plan.Instance, b: list[int]) -> int:
    leaves = [1 << i for i in range(inst.n)]
    base = plan.independent_cost(inst, b)
    best = base
    for _mask, internal in all_hierarchies(leaves):
        savings = sum((k - 1) * b[m] for m, k in internal)
        best = min(best, base - savings)
    return best


def random_instance(n: int, seed: int) -> plan.Instance:
    rng = random.Random(seed)
    sig_weights: dict[int, int] = {}
    for _ in range(rng.randint(3, 10)):
        mask = rng.randrange(1, 1 << n)
        sig_weights[mask] = sig_weights.get(mask, 0) + rng.randrange(1, 100)
    for i in range(n):
        mask = 1 << i
        sig_weights[mask] = sig_weights.get(mask, 0) + rng.randrange(1, 50)
    return instance_from_signatures(n, sig_weights)


def test_zeta_transform_matches_direct_intersection():
    inst = random_instance(5, seed=11)
    b = plan.intersection_weights(inst)
    for mask in range(1, 1 << inst.n):
        expected = sum(
            w for d, w in inst.weight.items() if not mask & ~inst.signature[d]
        )
        assert b[mask] == expected


@pytest.mark.parametrize("n,seed", [(4, 1), (4, 2), (5, 3), (5, 4), (6, 5), (6, 6)])
def test_dp_matches_exhaustive_search(n, seed):
    inst = random_instance(n, seed)
    b = plan.intersection_weights(inst)
    cost, _tree = plan.optimal_hierarchy(inst, b)
    assert cost == brute_force_cost(inst, b)


@pytest.mark.parametrize("n,seed", [(5, 21), (6, 22), (7, 23)])
def test_reconstructed_tree_costs_what_the_dp_says(n, seed):
    inst = random_instance(n, seed)
    b = plan.intersection_weights(inst)
    cost, tree = plan.optimal_hierarchy(inst, b)
    tree = plan.assign_content(inst, tree)
    assert plan.footprint(tree) == cost


@pytest.mark.parametrize("n,seed", [(5, 31), (6, 32), (7, 33)])
def test_every_image_still_gets_every_package(n, seed):
    inst = random_instance(n, seed)
    b = plan.intersection_weights(inst)
    _cost, tree = plan.optimal_hierarchy(inst, b)
    tree = plan.assign_content(inst, tree)

    def collect(node, mask, acc):
        if node.mask & mask:
            acc.update(node.content)
        for child in node.children:
            collect(child, mask, acc)

    for i in range(inst.n):
        got: set[str] = set()
        collect(tree, 1 << i, got)
        want = {d for d, sig in inst.signature.items() if sig >> i & 1}
        assert got == want


def test_bounds_are_ordered():
    inst = random_instance(6, seed=41)
    b = plan.intersection_weights(inst)
    cost, _tree = plan.optimal_hierarchy(inst, b)
    assert plan.flat_floor(inst) <= cost <= plan.independent_cost(inst, b)
    k1, _ = plan.best_k_bases(inst, b, 1)
    k2, _ = plan.best_k_bases(inst, b, 2)
    assert cost <= k2 <= k1 <= plan.independent_cost(inst, b)


def test_greedy_is_not_always_optimal():
    """Crossing structure: the best pair merge blocks the two merges that pay more."""
    inst = instance_from_signatures(
        4, {0b0110: 10, 0b0011: 9, 0b1100: 9, 0b0001: 1, 0b1000: 1}
    )
    b = plan.intersection_weights(inst)
    cost, _tree = plan.optimal_hierarchy(inst, b)
    greedy, _merges = plan.greedy_hierarchy(inst, b)
    assert cost == brute_force_cost(inst, b)
    assert greedy > cost


@pytest.mark.parametrize("n,seed", [(5, 51), (6, 52)])
def test_exact_two_base_plan_matches_enumeration(n, seed):
    inst = random_instance(n, seed)
    b = plan.intersection_weights(inst)
    cost, _masks = plan.best_k_bases(inst, b, 2)
    base = plan.independent_cost(inst, b)
    best = base
    universe = range(1, 1 << n)
    for outer in universe:
        if plan.popcount(outer) < 2:
            continue
        best = min(best, base - (plan.popcount(outer) - 1) * b[outer])
        for inner in universe:
            if inner == outer or plan.popcount(inner) < 2:
                continue
            if inner & outer == inner:
                gain = (plan.popcount(outer) - plan.popcount(inner)) * b[outer]
                gain += (plan.popcount(inner) - 1) * b[inner]
            elif inner & outer:
                continue
            else:
                gain = (plan.popcount(outer) - 1) * b[outer]
                gain += (plan.popcount(inner) - 1) * b[inner]
            best = min(best, base - gain)
    assert cost == best


def test_measured_corpus_solves_and_balances():
    """The shipped measurement reproduces a plan whose layers sum to the DP cost."""
    sizes = ROOT / "data" / "dist_sizes.json"
    if not sizes.exists():
        pytest.skip("run scripts/measure_sizes.py first")
    from pullplan.family import FAMILY

    index = corpus.load_sizes(sizes)
    inst = plan.build_instance(FAMILY, index, corpus.closure)
    b = plan.intersection_weights(inst)
    cost, tree = plan.optimal_hierarchy(inst, b)
    tree = plan.assign_content(inst, tree)
    assert plan.footprint(tree) == cost
    assert cost < plan.independent_cost(inst, b)
    assert cost >= plan.flat_floor(inst)
