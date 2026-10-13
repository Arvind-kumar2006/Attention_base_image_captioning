"""Static checks for a .ipynb: Python cells must parse, and names used across
cells must be defined by an earlier cell.

Shell magics (!cmd, %magic) and their backslash continuations are stripped
before parsing, since they are not Python.
"""

from __future__ import annotations

import ast
import builtins
import json
import sys
from pathlib import Path

MAGIC_PREFIXES = ("!", "%", "from IPython", "FileLink(")
SHELL_COMMANDS = ("cd ", "git ", "pip ", "zip ", "unzip ", "export ")
# continuation fragments of a shell escape, e.g. "!cmd \\\n  || echo hi"
SHELL_FRAGMENTS = ("||", "&&", "|", ">")


def is_shell_line(stripped: str) -> bool:
    """True when a line is a shell escape, a magic, or part of one."""
    if stripped.startswith(("!", "%")):
        return True
    if stripped.startswith(MAGIC_PREFIXES):
        return True
    if stripped.startswith(SHELL_COMMANDS):
        return True
    return stripped.startswith(SHELL_FRAGMENTS)


def strip_magics(source: str) -> str:
    """Drop shell-escaped lines and IPython display helpers.

    Handles a command split over several lines with trailing backslashes, and
    fragments such as ``|| echo ...`` that continue a shell escape.
    """
    kept: list[str] = []
    in_shell = False
    for line in source.split("\n"):
        stripped = line.strip()

        if in_shell:
            if stripped.endswith("\\") or stripped == "":
                continue
            in_shell = False
            if is_shell_line(stripped) or stripped.startswith("2>/dev/null"):
                continue

        if stripped.startswith(("!git ", "!pip ", "!unzip", "%cd ")):
            in_shell = stripped.endswith("\\")
            continue
        if is_shell_line(stripped):
            continue
        kept.append(line)
    return "\n".join(kept)


def defined_names(tree: ast.AST) -> set[str]:
    """Every name the cell binds or imports, at any scope."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                for arg in (
                    *args.posonlyargs,
                    *args.args,
                    *args.kwonlyargs,
                    args.vararg,
                    args.kwarg,
                ):
                    if arg:
                        names.add(arg.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def function_nodes(tree: ast.AST) -> set[int]:
    """ids of nodes inside a function body, so their locals are not flagged."""
    inside: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            inside.update(id(child) for child in ast.walk(node))
    return inside


def cross_cell_names(tree: ast.AST) -> set[str]:
    """Names a cell exports for later cells to use."""
    exported: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exported.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                exported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    exported.add(target.id)
    return exported


def check(path: Path) -> int:
    """Run the checks; returns the number of problems found."""
    notebook = json.loads(path.read_text())
    known = set(dir(builtins))
    available: dict[str, int] = {}
    problems: list[tuple[int, str]] = []
    order: list[tuple[int, str, int]] = []
    pure_magic_cells = 0

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        python = strip_magics(source)
        if not python.strip():
            pure_magic_cells += 1
            continue
        try:
            tree = ast.parse(python)
        except SyntaxError as error:
            problems.append((index, f"syntax error: {error}"))
            continue

        for name in defined_names(tree):
            known.add(name)
        inside = function_nodes(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and id(node) not in inside
                and node.id not in known
            ):
                problems.append(
                    (index, f"undefined global {node.id!r} on line {node.lineno}")
                )

        for name in cross_cell_names(tree):
            available.setdefault(name, index)

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        python = strip_magics("".join(cell["source"]))
        if not python.strip():
            continue
        tree = ast.parse(python)
        inside = function_nodes(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and id(node) not in inside
            ):
                source_cell = available.get(node.id)
                if source_cell is not None and source_cell > index:
                    order.append((index, node.id, source_cell))

    print(f"{path}: {len(notebook['cells'])} cells, {pure_magic_cells} shell-only")
    if problems:
        print("  undefined / syntax problems:")
        for index, message in problems:
            print(f"    cell {index}: {message}")
    if order:
        print("  use before definition:")
        for index, name, source_cell in order:
            print(f"    cell {index} uses {name!r} from cell {source_cell}")
    if not problems and not order:
        print("  ok: every code cell parses and every name is available")
    return len(problems) + len(order)


if __name__ == "__main__":
    total = 0
    for argument in sys.argv[1:]:
        total += check(Path(argument))
    raise SystemExit(1 if total else 0)
