#!/usr/bin/env python3
"""Regression tests for the typeclass-generalization runner.

Run with:

    python3 scripts/test_generalize_typeclasses.py
"""

import pathlib
import tempfile
import unittest

import generalize_typeclasses as g

HEADER = """/-
Copyright (c) 2026 The Tau Ceti contributors. All rights reserved.
-/
module

-- Public: a comment between imports.
public import Mathlib.Algebra.Group.Defs
import all TauCeti.B
public meta import TauCeti.C

/-!
# Title

import TauCeti.D
-/

theorem x : True := trivial
"""


class HeaderTest(unittest.TestCase):
    def test_scan_header(self):
        is_module, imports, end = g.scan_header(HEADER)
        self.assertTrue(is_module)
        self.assertEqual(imports, ["Mathlib.Algebra.Group.Defs", "TauCeti.B", "TauCeti.C"])
        self.assertEqual(HEADER.split("\n")[end - 1], "public meta import TauCeti.C")

    def test_inject_non_module(self):
        src, is_module, line = g.inject(
            "import Mathlib.Order.Basic\n\ntheorem x : True := trivial\n")
        self.assertFalse(is_module)
        self.assertEqual(line, 2)
        self.assertEqual(src.split("\n")[:2],
                         ["import Mathlib.Order.Basic", "import GeneralizationLinter"])

    def test_inject_module(self):
        src, is_module, line = g.inject(HEADER)
        self.assertTrue(is_module)
        self.assertEqual(src.split("\n")[line - 1], "meta import GeneralizationLinter")


class LevelTest(unittest.TestCase):
    def test_longest_path(self):
        graph = {"A": [], "B": ["A"], "C": ["A", "B"], "D": ["A"], "E": []}
        self.assertEqual(g.levels(graph), {"A": 0, "B": 1, "C": 2, "D": 1, "E": 0})

    def test_cycle(self):
        with self.assertRaises(ValueError):
            g.levels({"A": ["B"], "B": ["A"]})

    def test_import_graph_ignores_external_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "TauCeti"
            (root / "X").mkdir(parents=True)
            (root / "A.lean").write_text("import Mathlib.Order.Basic\n")
            (root / "X" / "B.lean").write_text("module\n\npublic import TauCeti.A\n")
            graph = g.import_graph(root)
        self.assertEqual(graph, {"TauCeti.A": [], "TauCeti.X.B": ["TauCeti.A"]})

    def test_keys_follow_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "TauCeti"
            root.mkdir()
            (root / "A.lean").write_text("import Mathlib.Order.Basic\n")
            (root / "B.lean").write_text("import TauCeti.A\n")
            (root / "C.lean").write_text("import Mathlib.Order.Basic\n")
            before = g.source_keys(g.import_graph(root), "salt", root)
            (root / "A.lean").write_text("import Mathlib.Order.Basic\n-- edited\n")
            after = g.source_keys(g.import_graph(root), "salt", root)
            salted = g.source_keys(g.import_graph(root), "other", root)
        self.assertNotEqual(before["TauCeti.A"], after["TauCeti.A"])
        self.assertNotEqual(before["TauCeti.B"], after["TauCeti.B"])
        self.assertEqual(before["TauCeti.C"], after["TauCeti.C"])
        self.assertNotEqual(after["TauCeti.C"], salted["TauCeti.C"])

    def test_linter_sources_compile_imports_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            (checkout / "GeneralizationLinter" / "Graph").mkdir(parents=True)
            (checkout / "GeneralizationLinter.lean").write_text(
                "import GeneralizationLinter.Frontend\n")
            (checkout / "GeneralizationLinter" / "Frontend.lean").write_text(
                "import GeneralizationLinter.Graph.Vertex\n")
            (checkout / "GeneralizationLinter" / "Graph" / "Vertex.lean").write_text(
                "import Lean.Expr\n")
            order = [str(p.relative_to(checkout)) for p in g.linter_sources(checkout)]
        self.assertEqual(order, ["GeneralizationLinter/Graph/Vertex.lean",
                                 "GeneralizationLinter/Frontend.lean",
                                 "GeneralizationLinter.lean"])


def message(text: str, line: int = 7) -> dict:
    return {"kind": "linter.generalizeTypeclasses", "sourceLine": line,
            "data": text + "\n\nNote: This linter can be disabled with `set_option ...`"}


class FindingTest(unittest.TestCase):
    RESULT = {
        "module": "TauCeti.A", "source": "TauCeti/A.lean", "returncode": 0, "stats": [],
        "messages": [
            message("the `[CommSemiring k]` hypothesis of "
                    "`_private.TauCeti.A.0.TauCeti.Foo.bar` can be weakened to "
                    "`NonAssocSemiring k`."),
            message("the `[Group G]` hypothesis of `Foo.baz` can be removed (dropped), "
                    "but its proof may have to be modified.", 9),
            message("the `[Field K]` hypothesis of `Foo.qux` can be split into "
                    "`Nontrivial K`, `CommRing K`.", 11),
            {"kind": "linter.style.longLine", "sourceLine": 3, "data": "the `x` hypothesis"},
        ],
    }

    def test_parse(self):
        first, second, split = g.findings(self.RESULT)
        self.assertEqual(first.key, ("TauCeti/A.lean", "TauCeti.Foo.bar", "[CommSemiring k]"))
        self.assertEqual((first.target, first.unconfirmed), ("NonAssocSemiring k", False))
        self.assertEqual((second.target, second.unconfirmed, second.line), (None, True, 9))
        self.assertEqual((split.target, split.unconfirmed), ("Nontrivial K + CommRing K", False))
        self.assertEqual(str(first), "FINDING TauCeti/A.lean:7 TauCeti.Foo.bar "
                                     "[CommSemiring k] -> `NonAssocSemiring k`")

    def test_rejection_round_trip(self):
        finding = g.findings(self.RESULT)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "rejected.txt"
            path.write_text(f"# Needs commutativity downstream.\n{finding.rejection}\n")
            self.assertEqual(g.load_rejections(path), {finding.key: 2})

    def test_incomplete(self):
        self.assertIsNone(g.incomplete(self.RESULT))
        self.assertEqual(g.incomplete({**self.RESULT, "returncode": None}), "timed out")
        truncated = {**self.RESULT, "stats": [{"decl": "x", "truncated": True}]}
        self.assertEqual(g.incomplete(truncated), "1 declaration(s) truncated")
        aborted = {**self.RESULT, "stats": [{"decl": "x", "outcome": "aborted"}]}
        self.assertEqual(g.incomplete(aborted), "1 declaration(s) aborted")


if __name__ == "__main__":
    unittest.main()
