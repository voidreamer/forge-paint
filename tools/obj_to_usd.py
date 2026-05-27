#!/usr/bin/env python3
"""Small OBJ to USDA converter used by forge-paint test/release workflows.

This intentionally supports the common static-mesh OBJ subset only:
positions, UVs, normals and polygon faces. Faces are triangulated by fan,
vertices are flattened to face-varying data, and material libraries are ignored.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


def parse_float(text: str, line_no: int) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"line {line_no}: non-finite float {text!r}")
    return value


def resolve_index(text: str, length: int, line_no: int, label: str) -> int:
    index = int(text)
    if index == 0:
        raise ValueError(f"line {line_no}: OBJ indices are 1-based")
    resolved = index - 1 if index > 0 else length + index
    if resolved < 0 or resolved >= length:
        raise ValueError(f"line {line_no}: {label} index {text!r} out of range")
    return resolved


def normalize(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if length <= 1e-8:
        return (0.0, 1.0, 0.0)
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def face_normal(a, b, c):
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    return normalize(
        (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
    )


def parse_ref(text: str, counts: tuple[int, int, int], line_no: int):
    fields = text.split("/")
    if not fields or not fields[0] or len(fields) > 3:
        raise ValueError(f"line {line_no}: invalid face vertex {text!r}")
    pos = resolve_index(fields[0], counts[0], line_no, "position")
    uv = resolve_index(fields[1], counts[1], line_no, "texcoord") if len(fields) > 1 and fields[1] else None
    normal = resolve_index(fields[2], counts[2], line_no, "normal") if len(fields) > 2 and fields[2] else None
    return pos, uv, normal


def convert_obj(source: Path):
    positions: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []
    vertices: list[tuple[tuple[float, float, float], tuple[float, float], tuple[float, float, float]]] = []

    for line_no, raw in enumerate(source.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        tag, values = parts[0], parts[1:]
        if tag == "v":
            if len(values) < 3:
                raise ValueError(f"line {line_no}: v needs 3 components")
            positions.append(tuple(parse_float(v, line_no) for v in values[:3]))
        elif tag == "vt":
            if len(values) < 2:
                raise ValueError(f"line {line_no}: vt needs 2 components")
            texcoords.append((parse_float(values[0], line_no), parse_float(values[1], line_no)))
        elif tag == "vn":
            if len(values) < 3:
                raise ValueError(f"line {line_no}: vn needs 3 components")
            normals.append(normalize(tuple(parse_float(v, line_no) for v in values[:3])))
        elif tag == "f":
            refs = [parse_ref(v, (len(positions), len(texcoords), len(normals)), line_no) for v in values]
            if len(refs) < 3:
                raise ValueError(f"line {line_no}: face has fewer than 3 vertices")
            for i in range(1, len(refs) - 1):
                tri = [refs[0], refs[i], refs[i + 1]]
                fallback = face_normal(*(positions[r[0]] for r in tri))
                for pos_i, uv_i, normal_i in tri:
                    vertices.append(
                        (
                            positions[pos_i],
                            texcoords[uv_i] if uv_i is not None else (0.0, 0.0),
                            normals[normal_i] if normal_i is not None else fallback,
                        )
                    )

    if not vertices:
        raise ValueError(f"{source} contains no polygon faces")
    return vertices


def fmt(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.9f}".rstrip("0").rstrip(".")


def ident(path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", path.stem)
    if not name:
        return "mesh"
    if name[0].isdigit():
        name = "_" + name
    return name


def write_usda(source: Path, dest: Path, vertices) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="\n") as f:
        f.write('#usda 1.0\n(\n    defaultPrim = "root"\n    metersPerUnit = 1\n    upAxis = "Y"\n)\n\n')
        f.write('def Xform "root"\n{\n')
        f.write(f'    def Mesh "{ident(source)}"\n    {{\n')
        f.write("        int[] faceVertexCounts = [" + ", ".join("3" for _ in range(len(vertices) // 3)) + "]\n")
        f.write("        int[] faceVertexIndices = [" + ", ".join(str(i) for i in range(len(vertices))) + "]\n")
        f.write("        point3f[] points = [\n")
        for i, (pos, _, _) in enumerate(vertices):
            comma = "," if i + 1 < len(vertices) else ""
            f.write(f"            ({fmt(pos[0])}, {fmt(pos[1])}, {fmt(pos[2])}){comma}\n")
        f.write("        ]\n")
        f.write("        normal3f[] normals = [\n")
        for i, (_, _, normal) in enumerate(vertices):
            comma = "," if i + 1 < len(vertices) else ""
            f.write(f"            ({fmt(normal[0])}, {fmt(normal[1])}, {fmt(normal[2])}){comma}\n")
        f.write('        ] (\n            interpolation = "faceVarying"\n        )\n')
        f.write("        texCoord2f[] primvars:st = [\n")
        for i, (_, uv, _) in enumerate(vertices):
            comma = "," if i + 1 < len(vertices) else ""
            f.write(f"            ({fmt(uv[0])}, {fmt(uv[1])}){comma}\n")
        f.write('        ] (\n            interpolation = "faceVarying"\n        )\n')
        f.write('        uniform token subdivisionScheme = "none"\n')
        f.write("    }\n}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a static OBJ mesh to USDA.")
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    vertices = convert_obj(args.source)
    write_usda(args.source, args.dest, vertices)
    print(f"converted {args.source} -> {args.dest} ({len(vertices)} verts, {len(vertices) // 3} tris)")


if __name__ == "__main__":
    main()
