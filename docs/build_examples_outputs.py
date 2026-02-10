"""Run example scripts and generate output images for docs."""

from __future__ import annotations

import ast
import contextlib
import os
import runpy
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402


IMAGE_EXT = ".png"


def _script_has_savefig_call(script: Path) -> bool:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "savefig":
            return True
        if isinstance(func, ast.Name) and func.id == "savefig":
            return True
    return False


@contextlib.contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _run_one_example(script: Path, out_dir: Path) -> tuple[bool, str]:
    stem = script.stem
    has_savefig = _script_has_savefig_call(script)
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None

    try:
        plt.close("all")
        with _pushd(script.parent):
            # Keep RTD logs compact unless explicitly requested.
            verbose = os.environ.get("OPTABLE_EXAMPLES_VERBOSE", "").strip().lower()
            if verbose in {"1", "true", "yes", "on"}:
                runpy.run_path(str(script), run_name="__main__")
            else:
                with open(os.devnull, "w", encoding="utf-8") as sink:
                    with contextlib.redirect_stdout(sink):
                        with contextlib.redirect_stderr(sink):
                            runpy.run_path(str(script), run_name="__main__")

        # For scripts without savefig, snapshot remaining open figures.
        if not has_savefig:
            fignums = list(plt.get_fignums())
            if not fignums:
                return True, "ok (no figure)"
            for i, num in enumerate(fignums, start=1):
                fig = plt.figure(num)
                name = f"{stem}{IMAGE_EXT}" if len(fignums) == 1 else f"{stem}_{i}{IMAGE_EXT}"
                fig.savefig(out_dir / name, dpi=200, bbox_inches="tight")
            return True, f"ok ({len(fignums)} fallback image(s))"

        return True, "ok"
    except Exception:
        msg = traceback.format_exc(limit=3)
        return False, msg
    finally:
        plt.close("all")
        plt.show = old_show


def build_examples_outputs() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    examples_dir = repo_root / "examples"
    out_dir = repo_root / "docs" / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale generated fallback images.
    for old in out_dir.glob(f"*{IMAGE_EXT}"):
        old.unlink()

    results: list[tuple[str, bool, str]] = []
    for script in sorted(examples_dir.glob("*.py")):
        ok, message = _run_one_example(script, out_dir)
        results.append((script.name, ok, message))

    failed = [r for r in results if not r[1]]
    print("[examples] build summary:")
    for name, ok, message in results:
        status = "PASS" if ok else "FAIL"
        print(f"  - {name}: {status} {message}")
    if failed:
        print(f"[examples] {len(failed)} script(s) failed; continuing docs build.")


if __name__ == "__main__":
    build_examples_outputs()
