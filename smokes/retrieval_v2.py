"""Collaborator retrieval, v2: resolve what the class under test (CUT) references, instead of
substring-matching class names against its raw text.

Drop-in for utils.get_relevant_source_files: same state in, same list of SourceCodeFileData
out. What changes (numbers from smokes/retrieval_audit.py over 249 classes):

1. Names come from the PARSED code, resolved through package, imports and wildcards. A name
   inside a comment, inside a longer identifier, or belonging to another package no longer
   matches -- about 31% of the files v1 retrieved.
2. The CUT is never retrieved, and an implementer is added only for an interface/abstract type
   the tests must SUPPLY (a parameter type), never behind the CUT's own supertype.
3. A second hop for what tests need but the CUT never names: the exceptions its supertypes
   throw (EncoderException behind StringEncoder), and the types its javadoc links to.
4. Files are rendered as views, not method bags. Types in the CUT's API get their public
   surface -- constructors, static factories, nested builders, constants -- as signatures,
   plus full bodies of the methods the CUT calls. Types used only privately get signatures of
   the members the CUT calls, nothing more. Every view keeps its class header, so a
   constructor still reads as one.
"""
import re
from pathlib import Path
from typing import Any, Mapping

from utils import SourceCodeFileData, get_current_class_under_test, parser

TYPE_DECLS = {"class_declaration", "interface_declaration", "enum_declaration",
              "record_declaration", "annotation_type_declaration"}
BODIES = {"class_body", "interface_body", "enum_body", "annotation_type_body",
          "enum_body_declarations"}
MEMBERS = {"method_declaration", "constructor_declaration", "compact_constructor_declaration",
           "field_declaration", "constant_declaration", "annotation_type_element_declaration"}
LAST_REASONS: dict[str, str] = {}   # path -> why the last call retrieved it (for the audit)
MAX_FIELD_CHARS = 200
OMITTED = " { /* body omitted */ }"


# ---------------------------------------------------------------- parsing helpers
def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8")


def _package(code: str) -> str:
    m = re.search(r"^\s*package\s+([\w.]+)\s*;", code, re.M)
    return m.group(1) if m else ""


def _imports(code: str) -> tuple[dict[str, str], list[str]]:
    single, wild = {}, []
    for m in re.finditer(r"^\s*import\s+(static\s+)?([\w.]+)(\.\*)?\s*;", code, re.M):
        if m.group(1):
            continue
        if m.group(3):
            wild.append(m.group(2))
        else:
            single[m.group(2).rsplit(".", 1)[-1]] = m.group(2)
    return single, wild


def _is_private(decl, src: bytes) -> bool:
    for child in decl.children:
        if child.type == "modifiers":
            return "private" in _text(src, child).split()
    return False


def _is_static(decl, src: bytes) -> bool:
    for child in decl.children:
        if child.type == "modifiers":
            return "static" in _text(src, child).split()
    return False


def _type_refs(node, src: bytes, skip_bodies: bool = False) -> set[str]:
    """Type names used under `node`: type identifiers, Upper.member / Upper::m static access."""
    names, stack = set(), [node]
    while stack:
        n = stack.pop()
        if n.type in ("import_declaration", "package_declaration", "line_comment", "block_comment"):
            continue
        if skip_bodies and n is not node and n.type in ("block", "constructor_body"):
            continue
        if n.type == "type_identifier":
            names.add(_text(src, n))
        elif n.type in ("method_invocation", "field_access"):
            obj = n.child_by_field_name("object")
            if obj is not None and obj.type == "identifier" and _text(src, obj)[:1].isupper():
                names.add(_text(src, obj))
        elif n.type == "method_reference" and n.children and n.children[0].type == "identifier":
            if _text(src, n.children[0])[:1].isupper():
                names.add(_text(src, n.children[0]))
        stack.extend(n.children)
    return names


def _identifiers(root, src: bytes) -> set[str]:
    """Every identifier in actual code: what the CUT calls, reads or constructs."""
    out, stack = set(), [root]
    while stack:
        n = stack.pop()
        if n.type in ("import_declaration", "package_declaration", "line_comment", "block_comment"):
            continue
        if n.type in ("identifier", "type_identifier"):
            out.add(_text(src, n))
        stack.extend(n.children)
    return out


def _top_types(root):
    return [n for n in root.children if n.type in TYPE_DECLS]


def _members(decl):
    body = decl.child_by_field_name("body")
    if body is None:
        return []
    out = []
    for child in body.children:
        if child.type == "enum_body_declarations":
            out.extend(child.children)
        else:
            out.append(child)
    return out


def _surface(root, src: bytes) -> tuple[set[str], set[str], set[str]]:
    """(api, params, supertypes): type names in the non-private surface, the subset that
    are parameter types (a test must supply those), and the declared supertypes."""
    api, params, supers = set(), set(), set()

    def visit(decl, top: bool):
        for child in decl.children:
            if child.type in ("superclass", "super_interfaces", "extends_interfaces"):
                refs = _type_refs(child, src)
                api.update(refs)
                if top:
                    supers.update(refs)
        for member in _members(decl):
            if member.type in TYPE_DECLS:
                if not _is_private(member, src):
                    visit(member, False)
            elif member.type in MEMBERS and not _is_private(member, src):
                api.update(_type_refs(member, src, skip_bodies=True))
                formal = member.child_by_field_name("parameters")
                if formal is not None:
                    params.update(_type_refs(formal, src))

    for top in _top_types(root):
        visit(top, True)
    return api, params, supers


# ---------------------------------------------------------------- the project index
class _Index:
    """FQN -> file, for resolving a simple name the way javac would."""
    _cache: dict[int, "_Index"] = {}

    def __init__(self, all_files):
        self.by_fqn: dict[str, SourceCodeFileData] = {}
        self.parsed: dict[str, tuple[Any, bytes]] = {}
        for f in all_files:
            simple = Path(f.file_path).stem
            pkg = _package(f.file_content)
            self.by_fqn[f"{pkg}.{simple}" if pkg else simple] = f

    @classmethod
    def of(cls, all_files) -> "_Index":
        key = id(all_files)
        if key not in cls._cache:
            cls._cache = {key: cls(all_files)}
        return cls._cache[key]

    def parse(self, f: SourceCodeFileData):
        if f.file_path not in self.parsed:
            src = f.file_content.encode("utf-8")
            self.parsed[f.file_path] = (parser.parse(src).root_node, src)
        return self.parsed[f.file_path]

    def resolve(self, simple: str, code: str) -> SourceCodeFileData | None:
        single, wild = _imports(code)
        if simple in single:
            return self.by_fqn.get(single[simple])
        pkg = _package(code)
        for candidate in [pkg] + wild:
            hit = self.by_fqn.get(f"{candidate}.{simple}" if candidate else simple)
            if hit is not None:
                return hit
        return None

    def resolve_all(self, names: set[str], code: str, exclude: str) -> dict[str, SourceCodeFileData]:
        out = {}
        for name in names:
            hit = self.resolve(name, code)
            if hit is not None and hit.file_path != exclude:
                out[hit.file_path] = hit
        return out


# ---------------------------------------------------------------- rendering
def _signature(member, src: bytes) -> str:
    body = member.child_by_field_name("body")
    if body is None:                                  # abstract / interface method
        return _text(src, member)
    return _text(src, member)[:body.start_byte - member.start_byte].rstrip() + OMITTED


def _field(member, src: bytes) -> str:
    text = _text(src, member)
    if len(text) <= MAX_FIELD_CHARS:
        return text
    head = text.split("=", 1)[0].rstrip()
    return f"{head} = /* initializer omitted */;"


def _render_type(decl, src: bytes, full_bodies: set[str], keep, depth: int = 0) -> str:
    """Header, then the kept members; `keep(member, name)` decides, `full_bodies` names the
    methods whose bodies stay."""
    body = decl.child_by_field_name("body")
    header = _text(src, decl)[:body.start_byte - decl.start_byte].rstrip() if body else _text(src, decl)
    pad = "    " * (depth + 1)
    lines = []
    constants = []
    for member in _members(decl):
        if member.type == "enum_constant":
            cbody = member.child_by_field_name("body")
            text = _text(src, member)
            constants.append(text[:cbody.start_byte - member.start_byte].rstrip() if cbody else text)
            continue
        if member.type in TYPE_DECLS:
            if not _is_private(member, src):
                nested = _render_type(member, src, full_bodies, keep, depth + 1)
                if nested.count("\n") > 1:            # a nested type with nothing kept is noise
                    lines.append(nested)
            continue
        if member.type not in MEMBERS or _is_private(member, src):
            continue
        name_node = member.child_by_field_name("name")
        name = _text(src, name_node) if name_node is not None else ""
        if member.type in ("field_declaration", "constant_declaration"):
            declarator = member.child_by_field_name("declarator")
            name = _text(src, declarator.child_by_field_name("name")) if declarator else name
            if keep(member, name):
                lines.append(pad + _field(member, src))
            continue
        if not keep(member, name):
            continue
        text = _text(src, member) if name in full_bodies else _signature(member, src)
        lines.append(pad + text.replace("\n", "\n" + "    " * depth))
    out = ["    " * depth + header + " {"]
    if constants:
        out.append(pad + ",\n".join(pad + c if i else c for i, c in enumerate(constants)) + ";")
    out.extend(lines)
    out.append("    " * depth + "}")
    return "\n".join(out)


def _view(f: SourceCodeFileData, index: _Index, kind: str, cut_ids: set[str]) -> SourceCodeFileData:
    """kind: "api" (public surface as signatures + bodies of what the CUT calls),
    "internal" (signatures of what the CUT calls), "hint" (constructors and static factories)."""
    root, src = index.parse(f)
    simple = Path(f.file_path).stem

    def keep(member, name):
        if kind == "api":
            return True                                # non-private surface, filtered upstream
        if kind == "hint":
            return member.type == "constructor_declaration" or (
                member.type == "method_declaration" and _is_static(member, src))
        return name in cut_ids or (member.type == "constructor_declaration" and simple in cut_ids)

    full = cut_ids if kind == "api" else set()
    parts = [f"package {_package(f.file_content)};"] if _package(f.file_content) else []
    for decl in _top_types(root):
        parts.append(_render_type(decl, src, full, keep))
    return SourceCodeFileData(f.file_path, "\n\n".join(parts))


def _kind_of(f: SourceCodeFileData, index: _Index) -> str:
    root, src = index.parse(f)
    tops = _top_types(root)
    if not tops:
        return "unknown"
    t = tops[0]
    if t.type == "interface_declaration":
        return "interface"
    if t.type == "enum_declaration":
        return "enum"
    for child in t.children:
        if child.type == "modifiers" and "abstract" in _text(src, child).split():
            return "abstract"
    return "concrete"


# ---------------------------------------------------------------- the retriever
def get_relevant_source_files_v2(agent_state: Mapping[str, Any]) -> list[SourceCodeFileData]:
    all_files = agent_state["all_files"]
    cut = get_current_class_under_test(agent_state)
    index = _Index.of(all_files)
    code = cut.file_content
    root, src = index.parse(cut)
    cut_simple = Path(cut.file_path).stem
    cut_ids = _identifiers(root, src)

    api_names, param_names, super_names = _surface(root, src)
    referenced = index.resolve_all(_type_refs(root, src), code, cut.file_path)
    api = index.resolve_all(api_names, code, cut.file_path)
    params = index.resolve_all(param_names, code, cut.file_path)
    supers = index.resolve_all(super_names, code, cut.file_path)

    views: dict[str, SourceCodeFileData] = {}
    LAST_REASONS.clear()

    # 1. Types in the CUT's surface: the test calls, builds and catches these.
    for path, f in api.items():
        views[path] = _view(f, index, "api", cut_ids)
        LAST_REASONS.setdefault(path, "api")

    # 3a. Second hop: what the CUT's supertypes THROW. A test calling an inherited method has
    #     to catch or declare it (EncoderException behind StringEncoder). Only the throws:
    #     the supertypes' whole surface was 229 files at a 20% hit rate.
    for f in supers.values():
        sroot, ssrc = index.parse(f)
        for path, g in index.resolve_all(_throws_refs(sroot, ssrc), f.file_content,
                                         cut.file_path).items():
            views.setdefault(path, _view(g, index, "api", cut_ids))
            LAST_REASONS.setdefault(path, "supertype throws")

    # 2. An implementer only where the test must SUPPLY an instance: an interface or abstract
    #    parameter type. Behind the CUT's own supertype the CUT is the implementation.
    for path, f in params.items():
        if path in supers or _kind_of(f, index) not in ("interface", "abstract"):
            continue
        name = Path(path).stem
        pattern = re.compile(rf"\b(?:extends|implements)\b[^{{]*\b{re.escape(name)}\b")
        candidates = [g for g in all_files
                      if g.file_path not in (cut.file_path, path) and g.file_path not in views
                      and _kind_of(g, index) == "concrete" and pattern.search(g.file_content)]
        if candidates:
            current = [g for g in candidates if "@Deprecated" not in g.file_content[:g.file_content.find("class ")]]
            candidates = current or candidates
            same_pkg = [g for g in candidates if _package(g.file_content) == _package(f.file_content)]
            pick = min(same_pkg or candidates, key=lambda g: len(g.file_content))
            views[pick.file_path] = _view(pick, index, "api", cut_ids)
            LAST_REASONS.setdefault(pick.file_path, "implementer")

    # (Producers -- classes with a method returning the CUT -- were tried and dropped: 68
    #  files, one of them used by a test.)

    # 4. Types the CUT uses only privately: signatures of what it calls.
    for path, f in referenced.items():
        views.setdefault(path, _view(f, index, "internal", cut_ids))
        LAST_REASONS.setdefault(path, "internal")

    # 5. Types the CUT's javadoc links to ({@link X}, @see X). The author pointing at a type
    #    is a strong hint: 66% of them were used by tests, and a third of those the CUT
    #    never names in code (StringEncoderComparator -> Soundex). Constructors and static
    #    factories only: enough to build one.
    linked = set(re.findall(r"\{@link(?:plain)?\s+([A-Z]\w*)", code)) | \
        set(re.findall(r"@see\s+([A-Z]\w*)", code))
    for path, f in index.resolve_all(linked, code, cut.file_path).items():
        if path not in views:
            views[path] = _view(f, index, "hint", cut_ids)
            LAST_REASONS.setdefault(path, "javadoc link")

    return [v for v in views.values() if v.file_content.count("\n") > 1 or v.file_path in api]


def _throws_refs(root, src: bytes) -> set[str]:
    """Exception types in the throws clauses of the non-private methods."""
    names = set()
    for n in _walk_decls(root):
        if n.type in ("method_declaration", "constructor_declaration") and not _is_private(n, src):
            for child in n.children:
                if child.type == "throws":
                    names |= _type_refs(child, src)
    return names


def _walk_decls(root):
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        if n.type not in ("block", "constructor_body"):
            stack.extend(n.children)
