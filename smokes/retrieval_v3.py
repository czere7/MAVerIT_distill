"""Collaborator retrieval, v3: v2 plus what the compiler already worked out.

`mvn compile` leaves target/classes, and the bytecode in it records every class a class
references, resolved by javac, and every `new` instruction. Two additions over v2:

1. BYTECODE REFERENCES. Project types the CUT's bytecode references that v2's source
   resolution missed: return types of chained calls, `var` locals, nested classes. Added as
   internal views, keeping the members the bytecode shows the CUT calling.
2. CONSTRUCTION SITES. The classes whose bytecode does `new <CUT>` -- the question a test
   has to answer first is how to get an instance, and a class the CUT never names is
   often the answer (JsonFactory constructs ReaderBasedJsonParser). A site in a
   non-public method is followed one call up to the public method a test could call
   (JsonFactory._createParser -> createParser). Capped at MAX_PRODUCERS files.

Needs target/classes; without it, v3 is v2.
"""
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from smokes import retrieval_v2 as v2
from utils import SourceCodeFileData, extract_methods, get_current_class_under_test

MAX_PRODUCERS = 2
MAX_INHERITED = 6
LAST_REASONS: dict[str, str] = {}
JAVAP_BATCH = 150

_HEADER = re.compile(r"^(?:\S.*\s)?(?:class|interface|enum|record)\s+([\w.$]+)")
_MEMBER = re.compile(r"^  (?!Code:)(?P<decl>[^\s].*?);$")
_REF = re.compile(r"//\s+(Method|InterfaceMethod|Field|class)\s+(\S+)")
_DESC_TYPE = re.compile(r"L([\w/$]+);")
_DOTTED = re.compile(r"\b[a-zA-Z_$][\w$]*(?:\.[a-zA-Z_$][\w$]*)+")
_MODS = {"public", "protected", "private", "static", "final", "abstract", "synchronized",
         "native", "transient", "volatile", "default", "strictfp"}


@dataclass
class _Method:
    name: str
    visibility: str                                   # public / protected / package / private
    static: bool
    decl: str = ""
    calls: set = field(default_factory=set)           # (owner binary name, member name)
    news: set = field(default_factory=set)            # binary names instantiated


@dataclass
class _Class:
    name: str                                         # binary name, dotted, with $
    refs: set = field(default_factory=set)
    supers: set = field(default_factory=set)
    methods: list = field(default_factory=list)


def _javap(classes_dir: Path) -> dict[str, _Class]:
    names = sorted(str(p.relative_to(classes_dir))[:-6].replace("\\", ".").replace("/", ".")
                   for p in classes_dir.rglob("*.class"))
    out = []
    for i in range(0, len(names), JAVAP_BATCH):
        result = subprocess.run(["javap", "-c", "-p", "-cp", str(classes_dir), *names[i:i + JAVAP_BATCH]],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        out.append(result.stdout)
    return _parse("\n".join(out))


def _parse(text: str) -> dict[str, _Class]:
    classes: dict[str, _Class] = {}
    cls: _Class | None = None
    method: _Method | None = None
    for line in text.splitlines():
        if line.startswith("Compiled from"):
            cls, method = None, None
            continue
        if cls is None:
            m = _HEADER.match(line)
            if m:
                cls = classes.setdefault(m.group(1), _Class(m.group(1)))
                tail = line.split(m.group(1), 1)[1]
                cls.refs |= set(_DOTTED.findall(tail))
                cls.supers |= set(_DOTTED.findall(re.sub(r"<[^{]*?>", "", tail)))
            continue
        member = _MEMBER.match(line)
        if member:
            decl = member.group("decl")
            cls.refs |= set(_DOTTED.findall(decl))
            if "(" in decl:
                words = decl.split("(")[0].split()
                name = words[-1] if words else ""
                name = "<init>" if name == cls.name else name
                mods = {w for w in words if w in _MODS}
                vis = next((v for v in ("public", "protected", "private") if v in mods), "package")
                method = _Method(name, vis, "static" in mods, decl)
                cls.methods.append(method)
            else:
                method = None
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
        name = name.strip('"')
        cls.refs.add(owner)
        if method is not None and kind != "Field":
            method.calls.add((owner, name))
    return classes


class _Bytecode:
    _cache: dict[str, "_Bytecode | None"] = {}

    def __init__(self, classes: dict[str, _Class]):
        self.classes = classes
        self._ancestors: dict[str, set[str]] = {}
        self.callers: dict[tuple[str, str], list[tuple[str, _Method]]] = {}
        for c in classes.values():
            for m in c.methods:
                for call in m.calls:
                    self.callers.setdefault(call, []).append((c.name, m))

    def ancestors(self, name: str) -> set[str]:
        """The class and its project supertypes, transitively."""
        if name not in self._ancestors:
            seen, todo = {name}, [name]
            while todo:
                c = self.classes.get(todo.pop())
                for s in (c.supers if c else ()):
                    if s not in seen:
                        seen.add(s)
                        todo.append(s)
            self._ancestors[name] = seen
        return self._ancestors[name]

    @classmethod
    def for_file(cls, source_path: str) -> "_Bytecode | None":
        p = source_path.replace("\\", "/")
        i = p.find("/src/main/java/")
        if i < 0:
            return None
        module = p[:i]
        if module not in cls._cache:
            classes_dir = Path(module) / "target" / "classes"
            cls._cache[module] = cls(_javap(classes_dir)) if classes_dir.is_dir() else None
        return cls._cache[module]


def _outer(binary: str) -> str:
    return binary.split("$", 1)[0]


def get_relevant_source_files_v3(agent_state: Mapping[str, Any]) -> list[SourceCodeFileData]:
    views = {f.file_path: f for f in v2.get_relevant_source_files_v2(agent_state)}
    LAST_REASONS.clear()
    LAST_REASONS.update(v2.LAST_REASONS)

    bytecode = None
    cut = get_current_class_under_test(agent_state)
    try:
        bytecode = _Bytecode.for_file(cut.file_path)
    except OSError:
        pass
    if bytecode is None:
        return list(views.values())

    all_files = agent_state["all_files"]
    index = v2._Index.of(all_files)
    pkg = v2._package(cut.file_content)
    cut_fqn = f"{pkg}.{Path(cut.file_path).stem}" if pkg else Path(cut.file_path).stem
    root, src = index.parse(cut)
    cut_ids = v2._identifiers(root, src)
    own = [c for n, c in bytecode.classes.items() if _outer(n) == cut_fqn]
    called = {name for c in own for m in c.methods for _, name in m.calls}

    # 1. What the CUT's bytecode references that v2 did not resolve from source.
    for binary in sorted({r for c in own for r in c.refs}):
        f = index.by_fqn.get(_outer(binary))
        if f is None or f.file_path == cut.file_path or f.file_path in views:
            continue
        view = v2._view(f, index, "internal", cut_ids | called)
        if view.file_content.count("\n") > 1:
            views[f.file_path] = view
            LAST_REASONS[f.file_path] = "bytecode ref"

    # 2. Where the CUT is constructed, followed up to a public entry point.
    #    The entry point is credited to the CONSTRUCTING class even when inherited:
    #    JsonFactory._createParser is reached from TextualTSFactory.createParser through
    #    `this._createParser`, and the bytecode names the superclass as the call's owner.
    entries: dict[str, dict[str, set[str]]] = {}      # constructing class -> {name: decls}
    for c in bytecode.classes.values():
        if _outer(c.name) == cut_fqn:
            continue
        for m in c.methods:
            if not any(_outer(n) == cut_fqn for n in m.news):
                continue
            found = entries.setdefault(_outer(c.name), {})
            if m.visibility == "public":
                found.setdefault(m.name, set()).add(m.decl)
                continue
            lineage = bytecode.ancestors(c.name)
            for owner in lineage:
                for caller, cm in bytecode.callers.get((owner, m.name), []):
                    if cm.visibility == "public" and caller in lineage:
                        found.setdefault(cm.name, set()).add(cm.decl)
    ranked = sorted(((fqn, names) for fqn, names in entries.items() if names
                     if fqn in index.by_fqn and index.by_fqn[fqn].file_path not in views),
                    key=lambda e: (-len(e[1]), len(index.by_fqn[e[0]].file_content)))
    for fqn, names in ranked[:MAX_PRODUCERS]:
        f = index.by_fqn[fqn]
        views[f.file_path] = _producer_view(f, index, names)
        LAST_REASONS[f.file_path] = "constructs the CUT"
    return list(views.values())


def _producer_view(f: SourceCodeFileData, index, entries: dict[str, set[str]]) -> SourceCodeFileData:
    """Public constructors (to build the producer) and the entry methods, as signatures.
    Entry methods it inherits are listed from the bytecode, since its source lacks them."""
    root, src = index.parse(f)
    entry_names = set(entries)

    def keep(member, name):
        if member.type == "constructor_declaration":
            return True
        return member.type == "method_declaration" and name in entry_names

    parts = [f"package {v2._package(f.file_content)};"] if v2._package(f.file_content) else []
    for decl in v2._top_types(root):
        parts.append(v2._render_type(decl, src, set(), keep))
    declared = {m["name"] for m in extract_methods(f.file_content)}
    inherited = sorted(d for n, ds in entries.items() if n not in declared for d in ds)[:MAX_INHERITED]
    if inherited:
        parts.append("// Inherited entry points that construct the class under test:\n"
                     + "\n".join(f"//   {d};" for d in inherited))
    return SourceCodeFileData(f.file_path, "\n\n".join(parts))
