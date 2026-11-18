"""Turn a solved hierarchy into Dockerfiles, an ASCII tree, and a Mermaid graph."""

from __future__ import annotations

from pathlib import Path

from .plan import Instance, Node, base_images, nodes


def mb(value: int) -> str:
    return f"{value / 1e6:.1f} MB"


RUNTIME = "python:3.13-slim"


def tag_index(root: Node) -> dict[int, int]:
    """Stable names for the internal nodes that carry content, largest first."""
    internal = [n for n in nodes(root) if not n.is_leaf and n.bytes > 0]
    internal.sort(key=lambda n: (-n.bytes, n.mask))
    return {n.mask: i + 1 for i, n in enumerate(internal)}


def tag_for(inst: Instance, node: Node, index: dict[int, int]) -> str:
    if node.is_leaf:
        return inst.label(node.mask)
    if node.mask not in index:
        return RUNTIME  # an empty node installs nothing: it is just the runtime
    return f"base-{index[node.mask]}"


def ascii_tree(inst: Instance, root: Node) -> str:
    index = tag_index(root)
    lines: list[str] = []

    def walk(node: Node, depth: int) -> None:
        pad = "  " * depth
        name = tag_for(inst, node, index)
        if node.is_leaf:
            lines.append(f"{pad}{name} [image] carries {mb(node.bytes)}")
        else:
            lines.append(
                f"{pad}{name} [{len(inst.members(node.mask))} images] "
                f"carries {mb(node.bytes)} in {len(node.content)} "
                f"package{'' if len(node.content) == 1 else 's'}"
            )
        for child in node.children:
            walk(child, depth + 1)

    walk(root, 0)
    return "\n".join(lines)


def mermaid(inst: Instance, root: Node) -> str:
    index = tag_index(root)
    lines = ["graph TD"]
    for node in nodes(root):
        name = tag_for(inst, node, index)
        node_id = f"n{node.mask}"
        lines.append(f'  {node_id}["{name}<br/>{mb(node.bytes)}"]')
        for child in node.children:
            lines.append(f"  {node_id} --> n{child.mask}")
    return "\n".join(lines)


def dockerfiles(inst: Instance, root: Node, runtime: str = RUNTIME) -> dict[str, str]:
    """One Dockerfile per node: base images install shared packages, leaves the rest.

    An internal node with no content is not a build stage; its children inherit
    straight from the nearest ancestor that does install something.
    """
    index = tag_index(root)
    parent: dict[int, Node] = {}
    for node in nodes(root):
        for child in node.children:
            parent[child.mask] = node

    def inherits_from(node: Node) -> str:
        up = parent.get(node.mask)
        while up is not None and up.mask not in index:
            up = parent.get(up.mask)
        return runtime if up is None else tag_for(inst, up, index)

    out: dict[str, str] = {}
    for node in nodes(root):
        if not node.is_leaf and node.mask not in index:
            continue  # carries nothing, so there is no image to build
        name = tag_for(inst, node, index)
        from_ref = inherits_from(node)
        body = [
            f"# {name}",
            f"# shared by: {', '.join(inst.members(node.mask))}",
            f"# layer content: {len(node.content)} packages, {mb(node.bytes)} compressed",
            f"FROM {from_ref}",
        ]
        if node.content:
            pkgs = " \\\n        ".join(sorted(node.content))
            body.append("RUN pip install --no-cache-dir \\\n        " + pkgs)
        else:
            body.append("# nothing to install at this level")
        out[f"Dockerfile.{name}"] = "\n".join(body) + "\n"
    return out


def write_plan(inst: Instance, root: Node, out_dir: str | Path) -> Path:
    """Write the full plan: Dockerfiles, tree, Mermaid graph, layer report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in dockerfiles(inst, root).items():
        (out / name).write_text(text, encoding="utf-8")
    (out / "hierarchy.txt").write_text(ascii_tree(inst, root) + "\n", encoding="utf-8")
    (out / "hierarchy.mmd").write_text(mermaid(inst, root) + "\n", encoding="utf-8")

    index = tag_index(root)
    rows = ["name,images,packages,compressed_bytes"]
    for node in sorted(nodes(root), key=lambda n: -n.bytes):
        rows.append(
            f"{tag_for(inst, node, index)},"
            f"{len(inst.members(node.mask))},{len(node.content)},{node.bytes}"
        )
    (out / "layers.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out


def summarize_bases(inst: Instance, root: Node, limit: int = 6) -> str:
    """Human readable account of the base images the plan creates."""
    index = tag_index(root)
    lines = []
    for node in base_images(root)[:limit]:
        members = ", ".join(inst.members(node.mask))
        top = sorted(node.content, key=lambda d: -inst.weight[d])[:4]
        lines.append(f"{tag_for(inst, node, index):8s} {mb(node.bytes):>9s}  shared by {members}")
        lines.append(f"{'':8s} {'':>9s}  largest: {', '.join(top)}")
    return "\n".join(lines)
