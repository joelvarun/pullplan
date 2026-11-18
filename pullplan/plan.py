"""Exact synthesis of a base-image hierarchy for a family of container images.

Model
-----
Image i needs the distribution set S_i. A layer may be shared by a set of images
T only if every file in it is needed by all of them, so a layer's content is
constrained to the intersection of its users' sets. A Dockerfile build graph can
only express a laminar family: the layers an image inherits form a chain of FROM
parents, so the image sets sharing a layer are nested or disjoint, never crossing.

Write B(L) for the total compressed size of the distributions needed by every
image in L. A layer placed at a node with k children saves (k - 1) * B(L) bytes of
family footprint, and a k-ary node never saves more than a binary cascade over the
same leaves, so binary hierarchies contain an optimum. That gives

    h(L) = min over bipartitions (A, C) of L:  h(A) + h(C) - B(L),
    h({i}) = B({i}) = footprint of image i on its own,

an exact dynamic program over the 2^N subsets, evaluated in O(3^N) by enumerating
each subset's bipartitions. B itself is the superset sum (zeta transform) of the
signature weights, computed in O(2^N * N).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def popcount(mask: int) -> int:
    return bin(mask).count("1")


@dataclass
class Instance:
    images: list[str]
    sets: dict[str, set[str]]
    weight: dict[str, int]
    signature: dict[str, int]

    @property
    def n(self) -> int:
        return len(self.images)

    @property
    def full(self) -> int:
        return (1 << self.n) - 1

    def label(self, mask: int) -> str:
        return "+".join(self.images[i] for i in range(self.n) if mask >> i & 1)

    def members(self, mask: int) -> list[str]:
        return [self.images[i] for i in range(self.n) if mask >> i & 1]


@dataclass
class Node:
    mask: int
    children: list["Node"] = field(default_factory=list)
    content: list[str] = field(default_factory=list)
    bytes: int = 0

    @property
    def is_leaf(self) -> bool:
        return not self.children


def build_instance(family: dict[str, list[str]], index, closure_fn) -> Instance:
    """Resolve each service to its dependency closure and index it by signature."""
    images = list(family)
    sets: dict[str, set[str]] = {}
    for app, roots in family.items():
        got, _missing = closure_fn(roots, index)
        sets[app] = got
    weight = {name: index[name].gzip_bytes for name in set().union(*sets.values())}
    signature = {}
    for dist in weight:
        mask = 0
        for i, app in enumerate(images):
            if dist in sets[app]:
                mask |= 1 << i
        signature[dist] = mask
    return Instance(images=images, sets=sets, weight=weight, signature=signature)


def signature_weights(inst: Instance) -> dict[int, int]:
    """Total bytes per co-occurrence signature: the atoms of the problem."""
    out: dict[int, int] = {}
    for dist, mask in inst.signature.items():
        out[mask] = out.get(mask, 0) + inst.weight[dist]
    return out


def intersection_weights(inst: Instance) -> list[int]:
    """B[L] for every subset L, by superset sum over the signature weights."""
    size = 1 << inst.n
    b = [0] * size
    for mask, w in signature_weights(inst).items():
        b[mask] += w
    for bit in range(inst.n):
        step = 1 << bit
        for mask in range(size):
            if not mask & step:
                b[mask] += b[mask | step]
    return b


def independent_cost(inst: Instance, b: list[int]) -> int:
    """What the family costs today: one self-contained layer set per image."""
    return sum(b[1 << i] for i in range(inst.n))


def flat_floor(inst: Instance) -> int:
    """Every signature as its own layer. Needs a builder that is not a FROM chain."""
    return sum(signature_weights(inst).values())


def optimal_hierarchy(inst: Instance, b: list[int]):
    """Exact minimum family footprint over laminar, Dockerfile-expressible plans."""
    size = 1 << inst.n
    h = [0] * size
    split = [0] * size
    for i in range(inst.n):
        h[1 << i] = b[1 << i]
    for mask in sorted(range(size), key=popcount):
        if popcount(mask) < 2:
            continue
        low = mask & -mask
        best = None
        best_part = 0
        part = (mask - 1) & mask
        while part:
            if part & low:
                value = h[part] + h[mask ^ part]
                if best is None or value < best:
                    best, best_part = value, part
            part = (part - 1) & mask
        h[mask] = best - b[mask]
        split[mask] = best_part
    return h[inst.full], _reconstruct(inst, split)


def _reconstruct(inst: Instance, split: list[int]) -> Node:
    def build(mask: int) -> Node:
        if popcount(mask) == 1:
            return Node(mask=mask)
        part = split[mask]
        return Node(mask=mask, children=[build(part), build(mask ^ part)])

    return build(inst.full)


def assign_content(inst: Instance, root: Node) -> Node:
    """Place each distribution at the highest node whose images all need it."""

    def walk(node: Node, parent_mask: int) -> None:
        for dist, sig in inst.signature.items():
            if node.mask & ~sig:
                continue
            if parent_mask and not (parent_mask & ~sig):
                continue  # an ancestor already carries it
            node.content.append(dist)
        node.content.sort()
        node.bytes = sum(inst.weight[d] for d in node.content)
        for child in node.children:
            walk(child, node.mask)

    walk(root, 0)
    return root


def nodes(root: Node) -> list[Node]:
    out = [root]
    for child in root.children:
        out.extend(nodes(child))
    return out


def collapse(root: Node) -> Node:
    """Drop empty internal nodes: a layer with no content is not a base image."""

    def walk(node: Node) -> Node:
        kids: list[Node] = []
        for child in node.children:
            child = walk(child)
            if child.bytes == 0 and not child.is_leaf:
                kids.extend(child.children)
            else:
                kids.append(child)
        node.children = kids
        return node

    return walk(root)


def base_images(root: Node) -> list[Node]:
    """Internal nodes that actually carry bytes, largest first."""
    out = [n for n in nodes(root) if not n.is_leaf and n.bytes > 0]
    out.sort(key=lambda n: -n.bytes)
    return out


def footprint(root: Node) -> int:
    return sum(n.bytes for n in nodes(root))


def greedy_hierarchy(inst: Instance, b: list[int]):
    """Agglomerative baseline: repeatedly merge the pair that shares the most."""
    parts = [1 << i for i in range(inst.n)]
    cost = independent_cost(inst, b)
    merges = []
    while len(parts) > 1:
        best = (-1, 0, 0)
        for x in range(len(parts)):
            for y in range(x + 1, len(parts)):
                union = parts[x] | parts[y]
                if b[union] > best[0]:
                    best = (b[union], parts[x], parts[y])
        gain, left, right = best
        cost -= gain
        merges.append((left, right, gain))
        parts = [p for p in parts if p not in (left, right)] + [left | right]
    return cost, merges


def best_k_bases(inst: Instance, b: list[int], k: int):
    """Exact best plan using at most k content-carrying base images, for k in (1, 2).

    One base over image set T saves (|T| - 1) * B(T). Two bases are nested or
    disjoint, never crossing. The disjoint case is a maximum over complementary
    subsets, so a subset-maximum transform settles it in O(2^N * N); the nested
    case is a maximum over (outer, inner) pairs, enumerated in O(3^N).
    """
    if k not in (1, 2):
        raise ValueError("exact enumeration is implemented for k in (1, 2)")
    size = 1 << inst.n
    base = independent_cost(inst, b)
    pop = [popcount(m) for m in range(size)]
    solo = [(pop[m] - 1) * b[m] if pop[m] >= 2 else 0 for m in range(size)]

    best_gain = 0
    best_plan: tuple[int, ...] = ()
    for mask in range(size):
        if solo[mask] > best_gain:
            best_gain, best_plan = solo[mask], (mask,)
    if k == 1:
        return base - best_gain, best_plan

    # Disjoint pair: best[M] is the best single base drawn from the images in M.
    best_in = solo[:]
    arg_in = list(range(size))
    for bit in range(inst.n):
        step = 1 << bit
        for mask in range(size):
            if mask & step and best_in[mask ^ step] > best_in[mask]:
                best_in[mask] = best_in[mask ^ step]
                arg_in[mask] = arg_in[mask ^ step]
    for outer in range(size):
        if solo[outer] == 0:
            continue
        rest = inst.full ^ outer
        gain = solo[outer] + best_in[rest]
        if gain > best_gain:
            best_gain = gain
            partner = arg_in[rest]
            best_plan = (outer,) if solo[partner] == 0 else tuple(sorted((outer, partner)))

    # Nested pair: the outer base keeps the inner subtree as one of its children.
    for outer in range(size):
        if pop[outer] < 3 or b[outer] == 0:
            continue
        inner = (outer - 1) & outer
        while inner:
            if pop[inner] >= 2:
                gain = (pop[outer] - pop[inner]) * b[outer] + solo[inner]
                if gain > best_gain:
                    best_gain, best_plan = gain, tuple(sorted((outer, inner)))
            inner = (inner - 1) & outer
    return base - best_gain, best_plan
