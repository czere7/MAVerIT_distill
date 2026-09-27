"""The collaborator index: one JSON record per project class, built once per project.

Retrieval needs to answer, for a class under test (CUT): what does it take, return and
throw; what does it inherit; how is it constructed; who implements its parameter types.
Every one of those is either a fact about ONE class, or the inverse of such a fact. So the
index is flat: the builder records each class's own facts, then inverts the ones retrieval
walks backwards (implemented_by, extended_by, constructed_by) in a second pass. Retrieval is
then lookups -- the focal record, then the records it points to -- not a graph traversal.

Sources of truth:
  bytecode (target/classes, via javap)  every resolved reference, member signatures with
                                        generics, calls, `new` sites, supertypes
  source   (src/main/java)             what bytecode loses: inlined constants, javadoc links

Output: collaborators/<project>_collaborators.jsonl. The first line is a header carrying a
hash of target/classes, so a stale index is rebuilt instead of silently used.

    python collaborators.py --project C:/path/to/project [--name NAME] [--compile]
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from utils import parser

INDEX_DIR = Path(__file__).resolve().parent / "collaborators"
JAVAP_BATCH = 150
MAX_ENTRIES_PER_PRODUCER = 8

_HEADER = re.compile(r"^(?P<mods>(?:[a-z-]+\s+)*)(?P<kw>class|interface|enum|record)\s+"
                     r"(?P<name>[\w.$]+)(?P<tail>.*?)\s*\{?$")
_MEMBER = re.compile(r"^  (?P<decl>\S.*);$")
_REF = re.compile(r"//\s+(Method|InterfaceMethod|Field|class)\s+(\S+)")
_DESC_TYPE = re.compile(r"L([\w/$]+);")
_DOTTED = re.compile(r"\b[a-zA-Z_$][\w$]*(?:\.[a-zA-Z_$][\w$]*)+")
_MODS = {"public", "protected", "private", "static", "final", "abstract", "synchronized",
         "native", "transient", "volatile", "default", "strictfp", "sealed", "non-sealed"}


# ============================================================== bytecode
@dataclass
class _Method:
    name: str
    visibility: str                     # public / protected / package / private
    static: bool
    decl: str
    kind: str                           # method / ctor / field
    param_types: set = field(default_factory=set)
    return_types: set = field(default_factory=set)
    throws_types: set = field(default_factory=set)
    abstract: bool = False
    params: list = field(default_factory=list)   # erased parameter types, in order
    ret: str = ""                                # erased return type
    calls: set = field(default_factory=set)      # (owner binary name, member name)
    news: set = field(default_factory=set)       # binary names instantiated


@dataclass
class _BClass:
    name: str                           # binary name: dotted, nested as Outer$Inner
    visibility: str
    kind: str                           # class / abstract / interface / enum / record / annotation
    extends: list = field(default_factory=list)
    implements: list = field(default_factory=list)
    refs: set = field(default_factory=set)
    members: list = field(default_factory=list)


def _strip_generics(s: str) -> str:
    out, depth = [], 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _split_top(s: str) -> list[str]:
    """Split on commas outside generics."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def _visibility(words) -> str:
    return next((v for v in ("public", "protected", "private") if v in words), "package")


def _parse_member(decl: str, owner: str) -> _Method | None:
    if decl.startswith("static {}"):
        return None
    if "(" not in decl:                                       # a field
        words = _strip_generics(decl).split()
        return _Method(words[-1] if words else "", _visibility(words), "static" in words,
                       decl, "field", return_types=set(_DOTTED.findall(decl)),
                       ret=words[-2] if len(words) > 1 else "")
    head, rest = decl.split("(", 1)
    params, _, after = rest.partition(")")
    head_plain = _strip_generics(head).split()
    name = head_plain[-1] if head_plain else ""
    kind = "ctor" if name == owner else "method"
    returns = set(_DOTTED.findall(head)) - {name}
    throws = set(_DOTTED.findall(after.split("throws", 1)[1])) if "throws" in after else set()
    erased = [_strip_generics(p).strip() for p in _split_top(params)]
    return _Method("<init>" if kind == "ctor" else name, _visibility(head_plain),
                   "static" in head_plain, decl, kind, param_types=set(_DOTTED.findall(params)),
                   return_types=set() if kind == "ctor" else returns, throws_types=throws,
                   abstract="abstract" in head_plain, params=erased,
                   ret="" if kind == "ctor" or len(head_plain) < 2 else head_plain[-2])


def _parse_javap(text: str) -> dict[str, _BClass]:
    classes: dict[str, _BClass] = {}
    cls: _BClass | None = None
    method: _Method | None = None
    for line in text.splitlines():
        if line.startswith("Compiled from"):
            cls, method = None, None
            continue
        if cls is None:
            h = _HEADER.match(line)
            if not h:
                continue
            words = h.group("mods").split()
            tail = h.group("tail")
            plain = _strip_generics(tail)
            ext = re.search(r"\bextends\s+(.*?)(?:\s+implements\b|$)", plain)
            imp = re.search(r"\bimplements\s+(.*)$", plain)
            extends = [t.strip() for t in ext.group(1).split(",")] if ext else []
            implements = [t.strip() for t in imp.group(1).split(",")] if imp else []
            kind = h.group("kw")
            if kind == "interface":
                kind = "annotation" if "java.lang.annotation.Annotation" in extends else "interface"
                implements, extends = extends, []            # an interface "extends" interfaces
            elif "java.lang.Enum" in extends:
                kind = "enum"
            elif "java.lang.Record" in extends:
                kind = "record"
            elif "abstract" in words:
                kind = "abstract"
            cls = classes.setdefault(h.group("name"), _BClass(
                h.group("name"), _visibility(words), kind, extends, implements))
            cls.refs |= set(_DOTTED.findall(tail))
            continue
        m = _MEMBER.match(line)
        if m:
            method = _parse_member(m.group("decl"), cls.name)
            if method is not None:
                cls.members.append(method)
                cls.refs |= method.param_types | method.return_types | method.throws_types
            continue
        ref = _REF.search(line)
        if ref is None:
            continue
        kind, target = ref.groups()
        for t in _DESC_TYPE.findall(target):
            cls.refs.add(t.replace("/", "."))
        if kind == "class":
            owner = target.strip('"').lstrip("[").removeprefix("L").removesuffix(";").replace("/", ".")
            cls.refs.add(owner)
            if method is not None and re.search(r"\bnew\s+#", line):
                method.news.add(owner)
            continue
        head = target.split(":", 1)[0]
        if "." in head.replace('"<init>"', "").replace('"<clinit>"', ""):
            owner, name = head.rsplit(".", 1)
            owner = owner.replace("/", ".")
        else:
            owner, name = cls.name, head
        cls.refs.add(owner)
        if method is not None and kind != "Field":
            method.calls.add((owner, name.strip('"')))
    return classes


def _javap(classes_dir: Path) -> dict[str, _BClass]:
    names = sorted(str(p.relative_to(classes_dir))[:-6].replace("\\", ".").replace("/", ".")
                   for p in classes_dir.rglob("*.class")
                   if p.stem not in ("package-info", "module-info"))
    chunks = []
    for i in range(0, len(names), JAVAP_BATCH):
        result = subprocess.run(["javap", "-c", "-p", "-cp", str(classes_dir), *names[i:i + JAVAP_BATCH]],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0 and not result.stdout:
            raise RuntimeError(f"javap failed: {result.stderr[:500]}")
        chunks.append(result.stdout)
    return _parse_javap("\n".join(chunks))


# ============================================================== source
def _package(code: str) -> str:
    m = re.search(r"^\s*package\s+([\w.]+)\s*;", code, re.M)
    return m.group(1) if m else ""


def _source_type_names(code: str) -> set[str]:
    """Type names used in code (not comments, strings or imports), from the parse tree."""
    src = code.encode("utf-8")
    names, stack = set(), [parser.parse(src).root_node]
    while stack:
        n = stack.pop()
        if n.type in ("import_declaration", "package_declaration", "line_comment", "block_comment"):
            continue
        if n.type == "type_identifier":
            names.add(src[n.start_byte:n.end_byte].decode())
        elif n.type in ("method_invocation", "field_access"):
            obj = n.child_by_field_name("object")
            if obj is not None and obj.type == "identifier":
                text = src[obj.start_byte:obj.end_byte].decode()
                if text[:1].isupper():
                    names.add(text)
        stack.extend(n.children)
    return names


def _resolver(by_fqn: dict):
    def resolve(simple: str, code: str) -> str | None:
        single, wild = {}, []
        for m in re.finditer(r"^\s*import\s+(static\s+)?([\w.]+)(\.\*)?\s*;", code, re.M):
            if m.group(1):
                continue
            if m.group(3):
                wild.append(m.group(2))
            else:
                single[m.group(2).rsplit(".", 1)[-1]] = m.group(2)
        if simple in single:
            return single[simple] if single[simple] in by_fqn else None
        pkg = _package(code)
        for candidate in [pkg] + wild:
            fqn = f"{candidate}.{simple}" if candidate else simple
            if fqn in by_fqn:
                return fqn
        return None
    return resolve


# ============================================================== build
def _outer(binary: str) -> str:
    return binary.split("$", 1)[0]


def _hash_classes(classes_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(classes_dir.rglob("*.class")):
        h.update(str(p.relative_to(classes_dir)).replace("\\", "/").encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def compile_project(project_dir: Path) -> None:
    mvn = shutil.which("mvn") or shutil.which("mvn.cmd")
    if mvn is None:
        raise RuntimeError("Maven executable 'mvn' was not found on PATH.")
    result = subprocess.run([mvn, "-q", "compile"], cwd=project_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mvn compile failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def build(project_dir: Path, name: str) -> tuple[Path, dict]:
    started = time.time()
    classes_dir = project_dir / "target" / "classes"
    source_root = project_dir / "src" / "main" / "java"
    bclasses = _javap(classes_dir)

    # Source files, keyed by top-level FQN.
    sources: dict[str, Path] = {}
    for path in source_root.rglob("*.java"):
        if path.stem in ("package-info", "module-info"):
            continue
        code = path.read_text(encoding="utf-8", errors="replace")
        pkg = _package(code)
        sources[f"{pkg}.{path.stem}" if pkg else path.stem] = path
    project = {fqn for fqn in sources if fqn in bclasses}
    resolve = _resolver(sources)

    def proj(names) -> set[str]:
        return {_outer(n) for n in names if _outer(n) in project}

    by_outer: dict[str, list[_BClass]] = {}
    for b in bclasses.values():
        by_outer.setdefault(_outer(b.name), []).append(b)

    # Binary-level ancestry (nested classes too): needed to follow `this.m()` calls, which
    # the bytecode attributes to whichever supertype declares m.
    def binary_ancestors(name: str) -> set[str]:
        seen, todo = {name}, [name]
        while todo:
            c = bclasses.get(todo.pop())
            for s in (c.extends + c.implements if c else []):
                s = s.split("<")[0]
                if s not in seen:
                    seen.add(s)
                    todo.append(s)
        return seen

    callers: dict[tuple[str, str], list[tuple[str, _Method]]] = {}
    for b in bclasses.values():
        for m in b.members:
            for call in m.calls:
                callers.setdefault(call, []).append((b.name, m))

    records: dict[str, dict] = {}
    for fqn in sorted(project):
        top = bclasses[fqn]
        own = by_outer.get(fqn, [])
        # The surface: the top-level class and its non-private, named nested classes.
        surface_classes = [top] + [b for b in own if b is not top and b.visibility != "private"
                                   and not b.name.split("$", 1)[1][:1].isdigit()]
        surface = [m for b in surface_classes for m in b.members if m.visibility != "private"]
        code = sources[fqn].read_text(encoding="utf-8", errors="replace")
        src_refs = {r for r in (resolve(n, code) for n in _source_type_names(code)) if r} - {fqn}
        byte_refs = proj(r for b in own for r in b.refs) - {fqn}
        links = set(re.findall(r"\{@link(?:plain)?\s+([A-Z]\w*)", code)) | \
            set(re.findall(r"@see\s+([A-Z]\w*)", code))
        calls: dict[str, set[str]] = {}
        for b in own:
            for m in b.members:
                for owner, member in m.calls:
                    if _outer(owner) in project and _outer(owner) != fqn:
                        calls.setdefault(_outer(owner), set()).add(member)
        records[fqn] = {
            "class": fqn,
            "file": sources[fqn].relative_to(project_dir).as_posix(),
            "kind": top.kind,
            "public": top.visibility == "public",
            "extends": [t.split("<")[0] for t in top.extends],
            "implements": [t.split("<")[0] for t in top.implements],
            "ctor_params": sorted(proj(t for m in surface if m.kind == "ctor" for t in m.param_types) - {fqn}),
            "method_params": sorted(proj(t for m in surface if m.kind == "method" for t in m.param_types) - {fqn}),
            "returns": sorted(proj(t for m in surface if m.kind == "method" for t in m.return_types) - {fqn}),
            "throws": sorted(proj(t for m in surface for t in m.throws_types) - {fqn}),
            "fields": sorted(proj(t for m in surface if m.kind == "field" for t in m.return_types) - {fqn}),
            "factories": sorted(m.decl for m in top.members
                                if m.kind == "method" and m.static and m.visibility != "private"
                                and fqn in {_outer(t) for t in m.return_types}),
            "calls": {k: sorted(v) for k, v in sorted(calls.items())},
            "instantiates": sorted(proj(n for b in own for m in b.members for n in m.news) - {fqn}),
            "references": sorted(byte_refs | src_refs),
            "source_only_refs": sorted(src_refs - byte_refs),
            "javadoc_links": sorted({r for r in (resolve(n, code) for n in links) if r} - {fqn}),
            "package": fqn.rsplit(".", 1)[0] if "." in fqn else "",
            # Every member of the top-level class and its named nested classes, with what a
            # renderer needs to filter: visibility, erased signature, and (below) overrides.
            "members": [
                {"name": m.name, "kind": m.kind, "visibility": m.visibility, "static": m.static,
                 "abstract": m.abstract, "params": m.params, "returns": m.ret,
                 "declared_in": b.name.split("$", 1)[1] if "$" in b.name else "",
                 "decl": m.decl}
                for b in [top] + [b for b in own if b is not top
                                  and not b.name.split("$", 1)[1][:1].isdigit()]
                for m in b.members if "$" not in m.name],
        }

    # Ancestors at the binary level, THROUGH nested classes: JsonFactoryBuilder extends
    # DecorableTSFactory$DecorableTSFBuilder, which extends TSFBuilder. `ancestors` below
    # only follows top-level records and stops at the nested class; this one does not.
    # Nested entries keep their $ name; their members are the outer record's members with
    # `declared_in` set.
    for fqn, rec in records.items():
        rec["binary_ancestors"] = sorted(
            a for a in binary_ancestors(fqn) - {fqn}
            if _outer(a) in records and _outer(a) != fqn)
        # Direct supertypes of every class in this file, nested ones included, so a
        # renderer can walk the chain nearest-first (overrides are resolved in that order).
        rec["type_supers"] = {b.name: [s.split("<")[0] for s in b.extends + b.implements]
                              for b in by_outer.get(fqn, [])
                              if not b.name.split("$", 1)[-1][:1].isdigit()}

    # Project ancestors, transitive, and what they expose to a caller of the CUT.
    for fqn, rec in records.items():
        seen, todo = set(), [fqn]
        while todo:
            r = records.get(todo.pop())
            for s in (r["extends"] + r["implements"] if r else []):
                if s in records and s not in seen:
                    seen.add(s)
                    todo.append(s)
        rec["ancestors"] = sorted(seen)
    for rec in records.values():
        inherited = {"method_params": set(), "returns": set(), "throws": set(), "fields": set()}
        for a in rec["ancestors"]:
            for k in inherited:
                inherited[k] |= set(records[a][k])
        rec["inherited_api"] = {k: sorted(v - {rec["class"]}) for k, v in inherited.items()}

    # Overrides, by erased signature: a supertype's renderer skips what the CUT redefines.
    for rec in records.values():
        inherited_sigs = {}
        for a in rec["ancestors"]:
            for m in records[a]["members"]:
                if m["kind"] == "method" and not m["static"] and m["visibility"] != "private" \
                        and not m["declared_in"]:
                    inherited_sigs.setdefault((m["name"], tuple(m["params"])), a)
        for m in rec["members"]:
            m["overrides"] = None
            if m["kind"] == "method" and not m["static"] and not m["declared_in"]:
                m["overrides"] = inherited_sigs.get((m["name"], tuple(m["params"])))

    # Inversions: what retrieval walks backwards.
    for rec in records.values():
        rec["implemented_by"], rec["extended_by"], rec["constructed_by"] = [], [], []
        rec["referenced_by_count"] = 0
    for fqn, rec in records.items():
        for a in rec["ancestors"]:
            if rec["kind"] in ("class", "enum", "record"):
                records[a]["implemented_by"].append(fqn)
        for s in rec["extends"]:
            if s in records:
                records[s]["extended_by"].append(fqn)
        for r in rec["references"]:
            records[r]["referenced_by_count"] += 1

    # Construction sites, followed to a public entry point through the constructing class's
    # own lineage (JsonFactory._createParser is reached from the inherited createParser).
    producers: dict[str, dict[str, set[str]]] = {}
    for b in bclasses.values():
        for m in b.members:
            for target in {_outer(n) for n in m.news} & project:
                if target == _outer(b.name):
                    continue
                entries = producers.setdefault(target, {}).setdefault(_outer(b.name), set())
                if m.visibility == "public":
                    entries.add(m.decl)
                    continue
                lineage = binary_ancestors(b.name)
                for owner in lineage:
                    for caller, cm in callers.get((owner, m.name), []):
                        if cm.visibility == "public" and caller in lineage:
                            entries.add(cm.decl)
    for target, by_class in producers.items():
        if target not in records:
            continue
        records[target]["constructed_by"] = [
            {"class": c, "entries": sorted(e)[:MAX_ENTRIES_PER_PRODUCER]}
            for c, e in sorted(by_class.items()) if c in records and e]

    for rec in records.values():
        rec["implemented_by"] = sorted(set(rec["implemented_by"]))
        rec["extended_by"] = sorted(set(rec["extended_by"]))

    INDEX_DIR.mkdir(exist_ok=True)
    out = INDEX_DIR / f"{name}_collaborators.jsonl"
    header = {"_header": True, "project": name, "project_dir": str(project_dir),
              "classes_hash": _hash_classes(classes_dir), "records": len(records),
              "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "build_seconds": round(time.time() - started, 1)}
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for fqn in sorted(records):
            f.write(json.dumps(records[fqn]) + "\n")
    return out, header


def load(project_dir: Path, name: str | None = None, rebuild_if_stale: bool = True) -> dict[str, dict]:
    """FQN -> record. Rebuilds when target/classes changed since the index was written."""
    project_dir = Path(project_dir)
    name = name or project_dir.name
    path = INDEX_DIR / f"{name}_collaborators.jsonl"
    classes_dir = project_dir / "target" / "classes"
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        if not rebuild_if_stale or header.get("classes_hash") == _hash_classes(classes_dir):
            return {r["class"]: r for r in map(json.loads, lines[1:])}
    build(project_dir, name)
    return load(project_dir, name, rebuild_if_stale=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", required=True, type=Path, help="Maven project directory")
    ap.add_argument("--name", help="index name; defaults to the project directory's name")
    ap.add_argument("--compile", action="store_true",
                    help="run `mvn compile` first (done anyway when target/classes is missing)")
    args = ap.parse_args()
    project_dir = args.project.resolve()
    if args.compile or not (project_dir / "target" / "classes").is_dir():
        print(f"compiling {project_dir} ...", flush=True)
        compile_project(project_dir)
    out, header = build(project_dir, args.name or project_dir.name)
    print(f"wrote {out} ({header['records']} records, {header['build_seconds']}s, "
          f"{out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
