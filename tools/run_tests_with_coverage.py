"""Run the test suite with Python's standard-library `trace` module and enforce line coverage >= 90%.

Because the build runs 100% offline without third-party `pytest` or `coverage`
wheels required, this runner discovers and executes all `tests/test_*.py` suites
via `unittest`, measures executable line coverage across `src/churn_flip_audit/`,
and fails the build if any test fails or total line coverage drops below 90.0%.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import trace
import unittest
from typing import Dict, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src" / "churn_flip_audit"
TESTS_DIR = REPO_ROOT / "tests"
MIN_COVERAGE_PCT = 90.0


def _executable_lines(py_file: Path) -> Set[int]:
    """Return the set of executable statement line numbers in a Python source file."""
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))
    exec_lines: Set[int] = set()

    # Collect docstring expr nodes to exclude from executable lines
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring_nodes.add(node.body[0])

    for node in ast.walk(tree):
        if node in docstring_nodes:
            continue
        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.Return,
                ast.Raise,
                ast.Assert,
                ast.Import,
                ast.ImportFrom,
                ast.Call,
                ast.If,
                ast.For,
                ast.While,
                ast.Break,
                ast.Continue,
                ast.FunctionDef,
                ast.ClassDef,
            ),
        ):
            if hasattr(node, "lineno"):
                exec_lines.add(int(node.lineno))

    # Exclude `if __name__ == "__main__":` blocks
    lines_text = source.splitlines()
    for idx, line in enumerate(lines_text, start=1):
        stripped = line.strip()
        if stripped.startswith('if __name__ == "__main__":') or stripped.startswith("if __name__ == '__main__':"):
            for rem in range(idx, len(lines_text) + 1):
                exec_lines.discard(rem)
            break

    return exec_lines


def run_suite_with_coverage() -> Tuple[bool, float, Dict[str, Tuple[int, int]]]:
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(TESTS_DIR), pattern="test_*.py", top_level_dir=str(REPO_ROOT))

    tracer = trace.Trace(count=1, trace=0, ignoredirs=[sys.prefix, sys.exec_prefix])
    runner = unittest.TextTestRunner(verbosity=2)

    result_holder = {}

    def _run() -> None:
        result_holder["res"] = runner.run(suite)

    tracer.runfunc(_run)
    res: unittest.TestResult = result_holder["res"]
    counts = tracer.results().counts

    per_file: Dict[str, Tuple[int, int]] = {}
    total_exec = 0
    total_hit = 0

    for py_file in sorted(SRC_DIR.glob("*.py")):
        resolved = str(py_file.resolve())
        exec_set = _executable_lines(py_file)
        if not exec_set:
            continue
        hit_set = {
            lineno
            for (filename, lineno), cnt in counts.items()
            if cnt > 0 and Path(filename).resolve() == Path(resolved) and lineno in exec_set
        }
        per_file[py_file.name] = (len(hit_set), len(exec_set))
        total_exec += len(exec_set)
        total_hit += len(hit_set)

    pct = (100.0 * total_hit / float(total_exec)) if total_exec > 0 else 100.0
    return res.wasSuccessful(), pct, per_file


def main() -> int:
    ok, pct, per_file = run_suite_with_coverage()
    print("\n" + "=" * 68)
    print("SOURCE LINE COVERAGE REPORT (src/churn_flip_audit/)")
    print("=" * 68)
    for fname, (hit, total) in per_file.items():
        file_pct = 100.0 * hit / float(total) if total > 0 else 100.0
        print(f"  {fname:24s}  {hit:4d} / {total:4d} lines  ({file_pct:6.2f}%)")
    print("-" * 68)
    print(f"  TOTAL COVERAGE: {pct:.2f}% (minimum enforced threshold: {MIN_COVERAGE_PCT:.2f}%)")
    print("=" * 68)

    if not ok:
        print("FAIL: Unit test suite reported failures.", file=sys.stderr)
        return 1
    if pct < MIN_COVERAGE_PCT:
        print(f"FAIL: Coverage {pct:.2f}% is below minimum {MIN_COVERAGE_PCT:.2f}%.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
