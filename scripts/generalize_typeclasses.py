#!/usr/bin/env python3
"""Run the typeclass-generalization linter over TauCeti one import level at a time.

The linter is `GeneralizationLinter` from https://github.com/nvlang/generalization. For each
typeclass hypothesis that a declaration does not need in full, it suggests a weaker class, or
dropping the hypothesis. It is not part of this repository: pass a checkout of its source with
`--linter` or `GENERALIZATION_LINTER`, by default the sibling checkout `../generalization-tauceti`.
The run compiles it against this project's Lake environment, and recompiles it whenever its source
or the Lake pins change, so it always matches the pinned Mathlib.

This runner relies on the `tauceti` branch of the fork https://github.com/harahu/generalization,
the fork's integration branch: it is never proposed upstream itself, but merges changes that are
not yet upstream, each developed on its own branch there for its own upstream pull request:

- `fix/zero-budget-truncation`: with a heartbeat budget of 0, which this runner passes by default,
  upstream aborts a declaration that exhausts Lean's own `maxHeartbeats`, discarding the
  weakenings it already verified, instead of flagging it as truncated.
- `fix/prefer-fallback`: under `splitPolicy` `allow` or `prefer`, upstream suggests nothing for a
  hypothesis whose preferred replacement fails verification, instead of trying the other one.
- `feat/strictness-guard`: upstream suggests weakenings that the other hypotheses undo, such as
  `AddCommGroup M` to `AddCommMonoid M` next to `Module R M` with `R` a ring, since a module over a
  ring is already an additive group.
- `fix/split-binder-order`: upstream drops a split whose replacements need data from one another,
  such as `[Field F]` into `[NoZeroDivisors F]` and `[CommRing F]`, which gives `NoZeroDivisors F`
  its multiplication.
- `feat/declaration-fixpoint`: upstream reports weakenings after a single pass, which can leave a
  hypothesis stronger than the weakened declaration needs; the fork weakens until nothing more can
  be, reporting each hypothesis once.
- `feat/bridge-rules`: upstream suggests weakenings undone by facts Mathlib states as no
  declaration, such as `NeZero n` to `Fintype (ZMod n)`, which holds only for `n ≠ 0`.
- `feat/galois-bridge`: upstream suggests weakening `IsGalois K L` to `IsGaloisGroup Gal(L/K) K L`
  over a finite-dimensional extension, which restates it (`IsGaloisGroup.isGalois`).
- `fix/pi-binder-bridges`: upstream applies no vacuity bridge to a family of instances, such as
  `[∀ i, AddCommGroup (N i)]` next to `[∀ i, Module R (N i)]`.
- `feat/parts-assembly`: upstream suggests splitting a class into the parents it bundles, such as
  `IsTopologicalGroup` into `ContinuousMul` and `ContinuousInv`, which restates it.
- `feat/more-bridge-rules`: proved rules for facts that no declaration states, such as a T0
  seminormed group being normed and a commutative simple ring being a field.
- `fix/resynth-memo`: upstream rebuilds each shared subterm of a weakened signature once per
  occurrence, which on some modules runs for hours.
- `feat/build-budget`: upstream re-elaborates a proof, and the conclusion's source, against a
  weakened statement under the linter's own budget, so lifting that budget, as this runner does,
  lets one that will never fit a build run unbounded, and exhausting it counts as a truncated
  analysis rather than as a proof that does not compile. The fork adds `buildHeartbeats`, which
  this runner sets to twice the default `maxHeartbeats` that Tau Ceti builds use, allowing no
  overrides. A weakening whose proof lands in between is still reported, as one worth making once
  the proof is made faster, and `lake build` then flags the proof.

The run warns when the checkout is on another branch. Drop an entry here once its change is
upstream, and point the runner back at upstream once none remain.

Levels
------

A module's level is the length of its longest import path through TauCeti modules: a module
importing no TauCeti module (only Mathlib and other dependencies) has level 0, and otherwise its
level is one more than the largest level among the TauCeti modules it imports. Levels are
computed from the source tree on every run.

Two modules on the same level never import each other, directly or indirectly, and the linter
elaborates a module from source against the build of its imports. So editing a module changes
the suggestions for that module and for higher levels, never for another module on its level.

Workflow
--------

Work up from level 0, finishing each level before starting the next:

1. Run `python3 scripts/generalize_typeclasses.py N`.
2. Implement the `FINDING`s it prints, or record the ones you decline in the rejections file
   (below).
3. Repeat from 1 until it exits with status 0. Results are cached per module, keyed on the
   module's source, the sources of every TauCeti module it imports, the Lake pins, the linter
   build and the options, so each run only re-lints the modules you changed since the last one
   (fixes can unlock further suggestions in the same module) and reports the whole level.
4. Run `lake build`: a weaker hypothesis can still break callers on higher levels, for example
   ones passing instances explicitly.
5. Run `python3 scripts/generalize_typeclasses.py --cumulative N`, which also re-checks every
   lower level. Edits can remove TauCeti imports and so move a module down into a finished
   level, as can rebasing onto a newer `main`. Unchanged modules come from the cache.
6. Commit, and move on to level N + 1.

A module reported as truncated has declarations that exhausted Lean's own `maxHeartbeats`; rerun
it with `--lean-option maxHeartbeats=0` and a longer `--timeout` until it completes. That lifts
only the limit on the linter's analysis: it still re-elaborates each proof within twice the budget
of a build.

Before linting, the run builds the TauCeti imports of the modules it is about to lint, so that
the linter sees the current state of the lower levels (`--no-build` skips this).

The exit status is 0 when every selected module was linted to completion and left no
suggestion outside the rejections file; 1 when any suggestion remains, or a module failed,
timed out, or had declarations whose analysis was truncated or aborted, or the build failed.

Rejections
----------

`scripts/generalize-typeclasses-rejected.txt` lists the suggestions deliberately not taken, one
per line as `FILE DECLARATION HYPOTHESIS`, copied from the `FINDING` line, with a `#` comment
line above saying why. A rejection stops matching once the hypothesis or declaration changes,
and the run warns about entries that no longer match anything.

Usage
-----

    python3 scripts/generalize_typeclasses.py 0
    python3 scripts/generalize_typeclasses.py 0 --linter /path/to/generalization
    python3 scripts/generalize_typeclasses.py --cumulative 3
    python3 scripts/generalize_typeclasses.py --list 3
    python3 scripts/generalize_typeclasses.py --files TauCeti/A.lean TauCeti/B.lean

Per-module sources, logs and results live in `<out>/modules/`, the compiled linter in
`<out>/linter/`, and a summary of the run in `<out>/<selection>.json`.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SOURCE_ROOT = pathlib.Path("TauCeti")
REJECTIONS = pathlib.Path("scripts/generalize-typeclasses-rejected.txt")
LINTER = "GeneralizationLinter"
LINTER_BRANCH = "tauceti"
PINS = [pathlib.Path("lake-manifest.json"), pathlib.Path("lean-toolchain")]
# Bump when the stored result format or the way results are produced changes.
CACHE_VERSION = 1

IMPORT = re.compile(r"^(?:public\s+)?(?:meta\s+)?import\s+(?:all\s+)?([\w'.]+)\s*$")
FINDING = re.compile(
    r"the `(?P<hypothesis>.+?)` hypothesis of `(?P<decl>.+?)` can be "
    r"(?:weakened to `(?P<target>.+?)`|split into (?P<split>`.+?`(?:, `.+?`)*)"
    r"|removed \(dropped\))(?P<rest>.*)", re.DOTALL)

# The linter's default heartbeat budgets truncate the analysis of many declarations; `--timeout`
# bounds the run instead, and `buildHeartbeats` each re-elaborated proof, at twice what a build
# allows.
LINTER_OPTIONS = [
    "weak.linter.generalizeTypeclasses=true",
    "weak.generalizeTypeclasses.stats=true",
    "weak.generalizeTypeclasses.generationHeartbeats=0",
    "weak.generalizeTypeclasses.perCandidateHeartbeats=0",
    "weak.generalizeTypeclasses.buildHeartbeats=400000",
    "weak.linter.mathlibStandardSet=true",
    "weak.linter.style.longFile=1500",
    "weak.linter.style.longFileDefValue=1500",
    "weak.linter.style.header=true",
]


def module_name(path: pathlib.Path) -> str:
    return ".".join(path.with_suffix("").parts)


def module_path(module: str, root: pathlib.Path = SOURCE_ROOT) -> pathlib.Path:
    return root.parent.joinpath(*module.split(".")).with_suffix(".lean")


def scan_header(src: str) -> tuple[bool, list[str], int]:
    """Return whether `src` is a `module` file, its imports, and the line index after them."""
    is_module, imports, end, depth = False, [], 0, 0
    for i, line in enumerate(src.split("\n")):
        if depth:
            depth += line.count("/-") - line.count("-/")
            continue
        code = line.split("--", 1)[0].strip()
        if code == "module":
            is_module, end = True, i + 1
        elif match := IMPORT.match(code):
            imports.append(match.group(1))
            end = i + 1
        elif code and not code.startswith("/-"):
            break
        depth += code.count("/-") - code.count("-/")
    return is_module, imports, end


def import_graph(root: pathlib.Path = SOURCE_ROOT) -> dict[str, list[str]]:
    """Map each TauCeti module to the TauCeti modules it imports."""
    sources = {module_name(p.relative_to(root.parent)): p for p in sorted(root.rglob("*.lean"))}
    graph = {}
    for module, path in sources.items():
        _, imports, _ = scan_header(path.read_text())
        graph[module] = [m for m in imports if m in sources]
    return graph


def levels(graph: dict[str, list[str]]) -> dict[str, int]:
    """Longest TauCeti import path to each module."""
    result: dict[str, int] = {}
    active: set[str] = set()

    def visit(module: str) -> int:
        if module in result:
            return result[module]
        if module in active:
            raise ValueError(f"import cycle through {module}")
        active.add(module)
        level = max((visit(m) + 1 for m in graph[module]), default=0)
        active.discard(module)
        result[module] = level
        return level

    for module in graph:
        visit(module)
    return result


def source_keys(graph: dict[str, list[str]], salt: str,
                root: pathlib.Path = SOURCE_ROOT) -> dict[str, str]:
    """Hash of each module's source together with the keys of the TauCeti modules it imports."""
    keys: dict[str, str] = {}

    def visit(module: str) -> str:
        if module not in keys:
            h = hashlib.sha256(salt.encode())
            h.update(module_path(module, root).read_bytes())
            for dep in sorted(graph[module]):
                h.update(visit(dep).encode())
            keys[module] = h.hexdigest()
        return keys[module]

    for module in graph:
        visit(module)
    return keys


def linter_sources(checkout: pathlib.Path) -> list[pathlib.Path]:
    """The linter's source files, in an order that compiles each after its imports."""
    graph = import_graph(checkout / LINTER)
    level_of = levels(graph)
    order = sorted(graph, key=lambda m: (level_of[m], m))
    return [module_path(m, checkout / LINTER) for m in order] + [checkout / f"{LINTER}.lean"]


def linter_key(checkout: pathlib.Path) -> str:
    """Hash of the linter's source and of the Lake pins it is compiled against."""
    h = hashlib.sha256(f"{CACHE_VERSION}\n".encode())
    for pin in PINS:
        h.update(pin.read_bytes())
    for path in linter_sources(checkout):
        h.update(str(path.relative_to(checkout)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def check_linter_branch(checkout: pathlib.Path) -> None:
    """Warn unless `checkout` is on the fork branch this runner relies on."""
    p = subprocess.run(["git", "-C", str(checkout), "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True)
    branch = p.stdout.strip() if p.returncode == 0 else None
    if branch != LINTER_BRANCH:
        print(f"warning: the linter checkout {checkout} is on {branch or 'no git branch'}, not "
              f"{LINTER_BRANCH}; see this script's docstring", file=sys.stderr)


def setup_file(path: pathlib.Path, module: str, is_module: bool) -> pathlib.Path:
    """Write a `lean --setup` file that resolves `module`'s imports through `LEAN_PATH`."""
    path.write_text(json.dumps(dict(
        name=module, isModule=is_module, importArts={}, dynlibs=[], plugins=[], options={})))
    return path


def compile_linter(checkout: pathlib.Path, dest: pathlib.Path, key: str,
                   env: dict[str, str]) -> bool:
    """Compile the linter into `dest` unless the build there already has `key`."""
    stamp = dest / "key"
    if stamp.is_file() and stamp.read_text() == key:
        return True
    print(f"compiling the linter from {checkout}", file=sys.stderr, flush=True)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for source in linter_sources(checkout):
        module = module_name(source.relative_to(checkout))
        target = dest.joinpath(*module.split("."))
        target.parent.mkdir(parents=True, exist_ok=True)
        is_module, _, _ = scan_header(source.read_text())
        setup = setup_file(dest / f"{module}.setup.json", module, is_module)
        p = subprocess.run(["lake", "env", "lean", "--setup", str(setup),
                            "-o", f"{target}.olean", "-i", f"{target}.ilean", str(source)],
                           env=env, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout + p.stderr, file=sys.stderr)
            print(f"compiling {module} failed", file=sys.stderr)
            return False
    stamp.write_text(key)
    return True


def inject(src: str) -> tuple[str, bool, int]:
    """Add the linter import; return the new source, the `module` flag, and its 1-based line."""
    is_module, _, end = scan_header(src)
    lines = src.split("\n")
    lines.insert(end, ("meta " if is_module else "") + "import GeneralizationLinter")
    return "\n".join(lines), is_module, end + 1


@dataclasses.dataclass(frozen=True)
class Finding:
    source: str
    line: int
    decl: str
    hypothesis: str
    target: str | None
    unconfirmed: bool

    @property
    def key(self) -> tuple[str, str, str]:
        return self.source, self.decl, self.hypothesis

    @property
    def rejection(self) -> str:
        """The line recording this suggestion in the rejections file."""
        return " ".join(self.key)

    def __str__(self) -> str:
        target = f"`{self.target}`" if self.target else "dropped"
        caveat = " (unconfirmed: may need other changes)" if self.unconfirmed else ""
        return (f"FINDING {self.source}:{self.line} {self.decl} {self.hypothesis} "
                f"-> {target}{caveat}")


def findings(result: dict) -> list[Finding]:
    """The linter's suggestions in `result`, with private declaration names unmangled."""
    private = f"_private.{result['module']}.0."
    found = []
    for m in result["messages"]:
        if m.get("kind") != "linter.generalizeTypeclasses":
            continue
        text = m.get("data", "").split("\n\nNote:")[0]
        match = FINDING.search(text)
        if not match:
            continue
        decl = match["decl"].removeprefix(private)
        unconfirmed = text.startswith("[UNVERIFIED]") or bool(match["rest"].strip(". "))
        target = match["target"]
        if match["split"]:
            target = " + ".join(t.strip("`") for t in match["split"].split("`, `"))
        found.append(Finding(result["source"], m["sourceLine"], decl, match["hypothesis"],
                             target, unconfirmed))
    return found


def incomplete(result: dict) -> str | None:
    """Why `result` does not cover its whole module, if it does not."""
    if result["returncode"] is None:
        return "timed out"
    if result["returncode"] != 0:
        return f"exited with {result['returncode']}"
    truncated = sum(1 for s in result["stats"] if s.get("truncated"))
    aborted = sum(1 for s in result["stats"] if s.get("outcome") == "aborted")
    problems = [f"{n} declaration(s) {what}" for n, what in
                ((truncated, "truncated"), (aborted, "aborted")) if n]
    return ", ".join(problems) or None


def load_rejections(path: pathlib.Path) -> dict[tuple[str, str, str], int]:
    """Map each rejected (file, declaration, hypothesis) to its line in `path`."""
    rejected = {}
    if not path.exists():
        return rejected
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"{path}:{number}: expected FILE DECLARATION HYPOTHESIS")
        rejected[tuple(parts)] = number
    return rejected


def lint(module: str, key: str, out: pathlib.Path, env: dict[str, str], options: list[str],
         args) -> dict:
    path = module_path(module)
    src, is_module, inserted = inject(path.read_text())
    shadow = out / f"{module}.lean"
    shadow.write_text(src)
    setup = setup_file(out / f"{module}.setup.json", module, is_module)
    command = ["lake", "env", "lean", "--json", f"--threads={args.threads}",
               "--setup", str(setup), *(f"-D{o}" for o in options), str(shadow)]
    started = time.monotonic()
    try:
        p = subprocess.run(command, env=env, capture_output=True, text=True,
                           timeout=args.timeout)
        stdout, stderr, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout or ""
        stderr, rc = f"timeout after {args.timeout} seconds", None
    (out / f"{module}.log").write_text(stdout + stderr)
    messages, stats = [], []
    for line in stdout.splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "GL_STATS " in m.get("data", ""):
            stats.append(json.loads(m["data"].split("GL_STATS ", 1)[1]))
        else:
            pos = m.get("pos", {}).get("line", 0)
            m["sourceLine"] = pos - (1 if pos > inserted else 0)
            messages.append(m)
    result = dict(module=module, source=str(path), key=key, returncode=rc,
                  seconds=round(time.monotonic() - started, 2), stats=stats, messages=messages)
    (out / f"{module}.result.json").write_text(json.dumps(result, indent=2) + "\n")
    problem = incomplete(result)
    print(f"{module}: {len(stats)} declarations, {len(findings(result))} suggestions, "
          f"{result['seconds']}s{f' ({problem})' if problem else ''}", flush=True)
    return result


def cached(module: str, key: str, out: pathlib.Path) -> dict | None:
    """The stored result for `module` if it is current and ran to the end."""
    try:
        result = json.loads((out / f"{module}.result.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("key") != key or result.get("returncode") != 0:
        return None
    return result


def build(modules: set[str]) -> bool:
    if not modules:
        return True
    print(f"building {len(modules)} imported modules", file=sys.stderr, flush=True)
    return subprocess.run(["lake", "build", *sorted(modules)]).returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("level", nargs="?", type=int, help="import level to lint")
    selection.add_argument("--files", nargs="+", type=pathlib.Path, metavar="FILE",
                           help="lint these source files instead of a level")
    parser.add_argument("--cumulative", action="store_true",
                        help="also lint every level below LEVEL")
    parser.add_argument("--list", action="store_true",
                        help="print the selected modules and exit")
    parser.add_argument("--linter", type=pathlib.Path,
                        default=os.environ.get("GENERALIZATION_LINTER",
                                               "../generalization-tauceti"),
                        help="checkout of the linter's source "
                             "(default: $GENERALIZATION_LINTER, or %(default)s)")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(".lake/generalize-typeclasses"),
                        help="output and cache directory (default: %(default)s)")
    parser.add_argument("--no-cache", action="store_true",
                        help="lint every selected module, even those with a current result")
    parser.add_argument("--no-build", action="store_true",
                        help="do not build the imports of the modules to lint")
    parser.add_argument("--jobs", type=int, default=4, help="parallel modules (default: 4)")
    parser.add_argument("--threads", type=int, default=2,
                        help="Lean threads per module (default: 2)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="seconds per module (default: %(default)s)")
    parser.add_argument("--option", action="append", default=[], metavar="NAME=VALUE",
                        help="extra weak.generalizeTypeclasses.NAME=VALUE option, overriding the "
                             "defaults, e.g. splitPolicy=allow (repeatable)")
    parser.add_argument("--lean-option", action="append", default=[], metavar="NAME=VALUE",
                        help="extra Lean option, e.g. maxHeartbeats=0 to let declarations the "
                             "linter reports as truncated run to completion (repeatable)")
    args = parser.parse_args(argv)
    if args.cumulative and args.files:
        parser.error("--cumulative needs a level")

    graph = import_graph()
    if args.files:
        selected = []
        for path in args.files:
            module = module_name(path)
            if module not in graph:
                parser.error(f"not a TauCeti source file: {path}")
            selected.append(module)
        name = "files"
    else:
        level_of = levels(graph)
        wanted = range(args.level + 1) if args.cumulative else [args.level]
        selected = sorted(m for m, level in level_of.items() if level in wanted)
        name = f"level{'0-' if args.cumulative else ''}{args.level}"
        print(f"{name}: {len(selected)} of {len(level_of)} modules "
              f"(max level {max(level_of.values())})", file=sys.stderr, flush=True)
    if args.list:
        for module in selected:
            print(module_path(module))
        return 0
    if not (args.linter / f"{LINTER}.lean").is_file():
        parser.error(f"no linter checkout at {args.linter}; clone "
                     "https://github.com/nvlang/generalization there, or pass --linter")
    rejected = load_rejections(REJECTIONS)

    options = (LINTER_OPTIONS + [f"weak.generalizeTypeclasses.{o}" for o in args.option]
               + args.lean_option)
    linter = linter_key(args.linter)
    keys = source_keys(graph, hashlib.sha256(f"{linter}\n{options}".encode()).hexdigest())
    out = args.out / "modules"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    if not args.no_cache:
        for module in selected:
            if (result := cached(module, keys[module], out)) is not None:
                results[module] = result
    stale = [m for m in selected if m not in results]
    print(f"{len(stale)} to lint, {len(results)} cached", file=sys.stderr, flush=True)

    if stale and not args.no_build and not build({d for m in stale for d in graph[m]}):
        print("build failed; fix it before linting", file=sys.stderr)
        return 1
    build_dir = args.out / "linter"
    env = os.environ.copy()
    env["LEAN_PATH"] = os.pathsep.join(
        p for p in (str(build_dir.resolve()), env.get("LEAN_PATH")) if p)
    if stale:
        check_linter_branch(args.linter)
        if not compile_linter(args.linter, build_dir, linter, env):
            return 1
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(lint, m, keys[m], out, env, options, args) for m in stale]
        for future in as_completed(futures):
            result = future.result()
            results[result["module"]] = result

    ordered = [results[m] for m in selected]
    (args.out / f"{name}.json").write_text(json.dumps(ordered, indent=2) + "\n")
    remaining, matched = [], set()
    for result in ordered:
        for finding in findings(result):
            if finding.key in rejected:
                matched.add(finding.key)
            else:
                remaining.append(finding)
    problems = [(r["source"], p) for r in ordered if (p := incomplete(r))]
    linted = {r["source"] for r in ordered if not incomplete(r)}
    unused = sorted((n, k) for k, n in rejected.items() if k[0] in linted and k not in matched)

    for finding in sorted(remaining, key=lambda f: (f.source, f.line)):
        print(finding)
    for source, problem in problems:
        print(f"INCOMPLETE {source}: {problem}")
    for number, key in unused:
        print(f"warning: {REJECTIONS}:{number} matches no suggestion: {' '.join(key)}",
              file=sys.stderr)
    print(f"{name}: {len(remaining)} suggestions, {len(matched)} rejected, "
          f"{len(problems)} incomplete modules", flush=True)
    return 1 if remaining or problems else 0


if __name__ == "__main__":
    sys.exit(main())
