"""Generate Sphinx example pages from scripts in examples/."""

from __future__ import annotations

import ast
from pathlib import Path


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif")


def _title_from_stem(stem: str) -> str:
    return stem.replace("_", " ").title()


def _savefig_paths(example_file: Path) -> list[str]:
    """Collect string literal savefig targets from a Python script."""
    tree = ast.parse(example_file.read_text(encoding="utf-8"))
    paths: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = None
        if isinstance(func, ast.Attribute):
            func_name = func.attr
        elif isinstance(func, ast.Name):
            func_name = func.id
        if func_name != "savefig" or not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            paths.append(first_arg.value)
    return paths


def _resolve_docs_image_candidates(
    raw_path: str, docs_dir: Path, example_dir: Path
) -> list[Path]:
    """Resolve likely docs image targets from a savefig path string."""
    p = Path(raw_path.replace("\\", "/"))
    candidates: list[Path] = []

    raw = p.as_posix()
    if raw.startswith("../docs/"):
        candidates.append(docs_dir / raw[len("../docs/") :])
    elif raw.startswith("./docs/"):
        candidates.append(docs_dir / raw[len("./docs/") :])
    elif raw.startswith("docs/"):
        candidates.append(docs_dir / raw[len("docs/") :])
    else:
        # Most examples save relative to runtime cwd; docs images are in docs/.
        candidates.append(docs_dir / p.name)
        # Also consider a path relative to the example script location.
        candidates.append((example_dir / p).resolve())
    return candidates


def _find_output_images(example_file: Path, docs_dir: Path) -> list[Path]:
    """Find existing output images for an example script."""
    found: list[Path] = []
    seen: set[Path] = set()

    for savefig_path in _savefig_paths(example_file):
        for candidate in _resolve_docs_image_candidates(
            savefig_path, docs_dir, example_file.parent
        ):
            # Keep only images under docs directory for Sphinx inclusion.
            if not candidate.exists():
                continue
            if candidate.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                rel = candidate.resolve().relative_to(docs_dir.resolve())
            except ValueError:
                continue
            normalized = docs_dir / rel
            if normalized not in seen:
                seen.add(normalized)
                found.append(normalized)

    # Fallback: discover files named like example stem in docs/.
    stem = example_file.stem
    for suffix in IMAGE_SUFFIXES:
        for pattern in (f"{stem}{suffix}", f"{stem}_*{suffix}"):
            for candidate in docs_dir.glob(pattern):
                if candidate not in seen and candidate.exists():
                    seen.add(candidate)
                    found.append(candidate)

    # Generated fallback outputs from docs/build_examples_outputs.py.
    generated_output_dir = docs_dir / "examples" / "output"
    if generated_output_dir.exists():
        for suffix in IMAGE_SUFFIXES:
            for pattern in (f"{stem}{suffix}", f"{stem}_*{suffix}"):
                for candidate in generated_output_dir.glob(pattern):
                    if candidate not in seen and candidate.exists():
                        seen.add(candidate)
                        found.append(candidate)

    return sorted(found)


def _write_example_page(
    output_file: Path, example_file: Path, docs_dir: Path, images: list[Path]
) -> None:
    stem = example_file.stem
    title = _title_from_stem(stem)
    lines: list[str] = [
        f".. _example-{stem}:",
        "",
        title,
        "=" * len(title),
        "",
        f"Source file: ``examples/{example_file.name}``",
        "",
    ]

    if images:
        lines.extend(["Output", "------", ""])
        for image in images:
            rel = image.resolve().relative_to(docs_dir.resolve()).as_posix()
            lines.extend(
                [
                    f".. image:: /{rel}",
                    f"   :alt: Output for {example_file.name}",
                    "   :width: 700px",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                ".. note::",
                "   No output image was found automatically for this example.",
                "",
            ]
        )

    lines.extend(
        [
            "Code",
            "----",
            "",
            f".. literalinclude:: ../../../examples/{example_file.name}",
            "   :language: python",
            "",
        ]
    )
    output_file.write_text("\n".join(lines), encoding="utf-8")


def generate_examples_docs() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    examples_dir = repo_root / "examples"

    out_root = docs_dir / "examples"
    out_generated = out_root / "generated"
    out_generated.mkdir(parents=True, exist_ok=True)

    example_files = sorted(examples_dir.glob("*.py"))
    generated_names: set[str] = set()

    index_lines: list[str] = [
        "Examples",
        "========",
        "",
        "This page is generated automatically from files in ``examples/``.",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]

    for example_file in example_files:
        stem = example_file.stem
        generated_names.add(stem)
        example_rst = out_generated / f"{stem}.rst"
        images = _find_output_images(example_file, docs_dir)
        _write_example_page(example_rst, example_file, docs_dir, images)
        index_lines.append(f"   generated/{stem}")

    index_lines.append("")
    (out_root / "index.rst").write_text("\n".join(index_lines), encoding="utf-8")

    # Remove stale generated pages for deleted example scripts.
    for stale in out_generated.glob("*.rst"):
        if stale.stem not in generated_names:
            stale.unlink()


if __name__ == "__main__":
    generate_examples_docs()
