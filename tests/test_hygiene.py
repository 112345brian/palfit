"""The package must stand alone: stdlib only, no repo imports, no I/O."""

import ast
import pathlib
import sys

import palfit

SRC = pathlib.Path(palfit.__file__).parent
FORBIDDEN_CALLS = {"open", "print", "input", "exec", "eval", "__import__"}


def test_imports_are_stdlib_or_self():
    assert len(list(SRC.glob("*.py"))) >= 3
    allowed = set(sys.stdlib_module_names) | {"palfit", "__future__"}
    for path in SRC.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module.split(".")[0]]
            for n in names:
                assert n in allowed, f"{path.name} imports non-stdlib module {n}"


def test_no_io_calls():
    assert len(list(SRC.glob("*.py"))) >= 3
    for path in SRC.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in FORBIDDEN_CALLS, f"{path.name}: {node.func.id}()"
