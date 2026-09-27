"""Rendering the walk's candidates into prompt context: what a test can USE, nothing else.

Rules (each one answers a failure seen in the teacher logs):
  * Accessibility is judged from the TEST's package, which is the CUT's. Public members
    always; protected and package-private only when the declaring class shares that package;
    private never. (Logs: 197 compiles failed on protected collaborator members, 67 on
    private ones.)
  * The CUT itself is shown in full -- its private logic is what the tests must cover -- but
    every private declaration is tagged PRIVATE_TAG. (Logs: 316 compiles failed on the
    CUT's own private members.) The tag contains utils.KEEP_COMMENT_MARKER, so the prompt's
    comment stripper keeps it.
  * Supertypes contribute only what the CUT inherits and does not override; no constructors.
    Bodies only for inherited methods the CUT calls.
  * Types the test only has to BUILD or OBTAIN (constructor args, implementations, subclasses,
    producers) become one-line recipes, not files: "new X(A, B)", "X.of(A)",
    "new F().create(A)". Only the how, never the implementation.
  * Collaborators in the CUT's surface: accessible members, bodies only for what the CUT calls.
    Collaborators used privately: only the members the CUT calls. Returned types: their
    accessors.
"""
import re
from pathlib import Path

from utils import KEEP_COMMENT_MARKER, SourceCodeFileData, parser

PRIVATE_TAG = f"/* PRIVATE: {KEEP_COMMENT_MARKER} */"
OMITTED = " { /* body omitted */ }"
MAX_RECIPES_PER_TYPE = 3
MAX_ENTRIES_PER_PRODUCER = 3

TYPE_DECLS = {"class_declaration", "interface_declaration", "enum_declaration",
              "record_declaration", "annotation_type_declaration"}
MEMBERS = {"method_declaration", "constructor_declaration", "field_declaration",
           "constant_declaration", "compact_constructor_declaration"}
BUILD_ROLES = {"ctor_args", "implementations", "subclasses"}


# ============================================================== small helpers
def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8")


def _modifiers(node, src: bytes) -> list[str]:
    for child in node.children:
        if child.type == "modifiers":
            return _text(src, child).split()
    return []


def _visibility(node, src: bytes) -> str:
    mods = _modifiers(node, src)
    for v in ("public", "protected", "private"):
        if v in mods:
            return v
    body = node.parent
    if body is not None and body.type in ("interface_body", "annotation_type_body"):
        return "public"                                   # interface members are implicitly public
    return "package"


def accessible(visibility: str, same_package: bool) -> bool:
    return visibility == "public" or (visibility != "private" and same_package)


def _simple(type_name: str) -> str:
    """java.util.List -> List, a.b.Outer$Inner -> Inner, keeping [] and ..."""
    base = re.sub(r"<.*>", "", type_name).strip()
    suffix = ""
    while base.endswith("[]") or base.endswith("..."):
        cut = 2 if base.endswith("[]") else 3
        suffix = base[-cut:] + suffix
        base = base[:-cut]
    return re.split(r"[.$]", base)[-1] + suffix


def _source_sig(member, src: bytes) -> tuple[str, tuple[str, ...]]:
    name_node = member.child_by_field_name("name")
    name = _text(src, name_node) if name_node is not None else ""
    params = []
    formal = member.child_by_field_name("parameters")
    for p in (formal.children if formal is not None else []):
        if p.type == "formal_parameter":
            params.append(_simple(_text(src, p.child_by_field_name("type"))))
        elif p.type == "spread_parameter":
            t = next((c for c in p.children if c.type not in ("modifiers", "variable_declarator", "...")), None)
            params.append(_simple(_text(src, t)) + "..." if t is not None else "...")
    return name, tuple(params)


def _index_sig(m: dict) -> tuple[str, tuple[str, ...]]:
    return m["name"], tuple(_simple(p) for p in m["params"])


def _members(decl):
    body = decl.child_by_field_name("body")
    out = []
    for child in (body.children if body is not None else []):
        out.extend(child.children if child.type == "enum_body_declarations" else [child])
    return out


def _signature(member, src: bytes) -> str:
    body = member.child_by_field_name("body")
    if body is None:
        return _text(src, member)
    return _text(src, member)[:body.start_byte - member.start_byte].rstrip() + OMITTED


def _render_type(decl, src: bytes, keep, body_for, depth: int = 0) -> str | None:
    """Header plus kept members; None when nothing is kept. `keep(member, name, vis)` and
    `body_for(name)` decide membership and whether a method keeps its body."""
    body = decl.child_by_field_name("body")
    header = _text(src, decl)[:body.start_byte - decl.start_byte].rstrip() if body else _text(src, decl)
    pad = "    " * (depth + 1)
    lines, constants = [], []
    for member in _members(decl):
        if member.type == "enum_constant":
            cbody = member.child_by_field_name("body")
            text = _text(src, member)
            constants.append(text[:cbody.start_byte - member.start_byte].rstrip() if cbody else text)
            continue
        if member.type in TYPE_DECLS:
            if keep(member, "", _visibility(member, src)):
                nested = _render_type(member, src, keep, body_for, depth + 1)
                if nested:
                    lines.append(nested)
            continue
        if member.type not in MEMBERS:
            continue
        vis = _visibility(member, src)
        if member.type in ("field_declaration", "constant_declaration"):
            declarator = member.child_by_field_name("declarator")
            name_node = declarator.child_by_field_name("name") if declarator is not None else None
            name = _text(src, name_node) if name_node is not None else ""
            if keep(member, name, vis):
                text = _text(src, member)
                if len(text) > 200:
                    text = text.split("=", 1)[0].rstrip() + " = /* initializer omitted */;"
                lines.append(pad + text)
            continue
        name_node = member.child_by_field_name("name")
        name = _text(src, name_node) if name_node is not None else ""
        if not keep(member, name, vis):
            continue
        text = _text(src, member) if body_for(name) else _signature(member, src)
        lines.append(pad + text)
    if not lines and not constants:
        return None
    out = ["    " * depth + header + " {"]
    if constants:
        out.append(pad + ", ".join(constants) + ";")
    out.extend(lines)
    out.append("    " * depth + "}")
    return "\n".join(out)


def _render_file(f: SourceCodeFileData, keep, body_for) -> str | None:
    src = f.file_content.encode("utf-8")
    root = parser.parse(src).root_node
    parts = [p for d in root.children if d.type in TYPE_DECLS
             for p in [_render_type(d, src, keep, body_for)] if p]
    return "\n\n".join(parts) if parts else None


# ============================================================== the CUT
def identifiers(root, src: bytes) -> set[str]:
    """Every identifier in the CUT's code (not comments or imports): what it calls, reads,
    constructs. Decides which collaborator methods keep their bodies."""
    out, stack = set(), [root]
    while stack:
        n = stack.pop()
        if n.type in ("import_declaration", "package_declaration", "line_comment", "block_comment"):
            continue
        if n.type in ("identifier", "type_identifier"):
            out.add(_text(src, n))
        stack.extend(n.children)
    return out


def tag_private_members(code: str) -> str:
    """The CUT in full, with PRIVATE_TAG in front of every private declaration."""
    src = code.encode("utf-8")
    root = parser.parse(src).root_node
    inserts = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in MEMBERS | TYPE_DECLS and "private" in _modifiers(n, src):
            mods = next(c for c in n.children if c.type == "modifiers")
            inserts.append(mods.start_byte)
        if n.type not in ("block", "constructor_body"):
            stack.extend(n.children)
    tag = (PRIVATE_TAG + " ").encode("utf-8")
    for at in sorted(inserts, reverse=True):
        src = src[:at] + tag + src[at:]
    return src.decode("utf-8")


def inaccessible_inherited(index: dict, cut: str, cut_ids: set[str]) -> dict[str, list[str]]:
    """ancestor -> names of its members the CUT uses but a test in the CUT's package cannot:
    protected or package-private, declared in another package, not redeclared by the CUT.
    (One gpt-oss run hit `_formatWriteFeatures has protected access in TSFBuilder` 84 times:
    the CUT's own code used it, so the model copied it into the test.)"""
    rec = index.get(cut)
    if rec is None:
        return {}
    own = {m["name"] for m in rec["members"] if not m["declared_in"]}
    found: dict[str, set[str]] = {}
    # binary_ancestors follows nested superclasses (older indexes lack it: fall back).
    for anc in rec.get("binary_ancestors", rec["ancestors"]):
        outer, _, nested = anc.partition("$")
        arec = index.get(outer)
        if arec is None or arec["package"] == rec["package"]:
            continue
        for m in arec["members"]:
            if (m["declared_in"] != nested or m["kind"] == "ctor"
                    or m["visibility"] not in ("protected", "package")
                    or m["name"] in own or m["name"] not in cut_ids):
                continue
            found.setdefault(anc.replace("$", "."), set()).add(m["name"])
    return {anc: sorted(names) for anc, names in sorted(found.items())}


def tag_inherited_members(code: str, inaccessible: dict[str, list[str]]) -> str:
    """One tag line after the CUT's opening brace naming the inherited members a test cannot
    use. Contains KEEP_COMMENT_MARKER, so the prompt's comment stripper keeps it."""
    if not inaccessible:
        return code
    src = code.encode("utf-8")
    root = parser.parse(src).root_node
    top = next((n for n in root.children if n.type in TYPE_DECLS), None)
    body = top.child_by_field_name("body") if top is not None else None
    if body is None:
        return code
    parts = "; ".join(f"{', '.join(names)} (from {anc.rsplit('.', 1)[-1]})"
                      for anc, names in inaccessible.items())
    line = (f"\n    /* INHERITED, {KEEP_COMMENT_MARKER}: protected or package-private in another "
            f"package: {parts} */").encode("utf-8")
    at = body.start_byte + 1                      # just after the opening brace
    return (src[:at] + line + src[at:]).decode("utf-8")


# ============================================================== views
def surface_view(f, same_package: bool, cut_ids: set[str]) -> str | None:
    """A collaborator in the CUT's surface: its accessible members; bodies of what the CUT calls."""
    return _render_file(f, lambda m, n, v: accessible(v, same_package), lambda n: n in cut_ids)


def called_view(f, same_package: bool, called: set[str]) -> str | None:
    """A collaborator the CUT uses privately: only the accessible members it calls."""
    simple = Path(f.file_path).stem
    return _render_file(
        f, lambda m, n, v: accessible(v, same_package) and (
            m.type in TYPE_DECLS or n in called or (m.type == "constructor_declaration" and simple in called)),
        lambda n: n in called)


def results_view(f, same_package: bool) -> str | None:
    """A type the CUT returns: its accessors (no-argument, non-void instance methods)."""
    src = f.file_content.encode("utf-8")

    def keep(m, n, v):
        if m.type != "method_declaration" or not accessible(v, same_package):
            return False
        params = m.child_by_field_name("parameters")
        rtype = m.child_by_field_name("type")
        return (params is not None and params.named_child_count == 0 and rtype is not None
                and rtype.type != "void_type" and "static" not in _modifiers(m, src))
    return _render_file(f, keep, lambda n: False)


def _find_type(root, src: bytes, binary: str):
    """The declaration of `binary` (a.b.Outer or a.b.Outer$Inner) inside its file."""
    names = binary.rsplit(".", 1)[-1].split("$")
    nodes = [n for n in root.children if n.type in TYPE_DECLS]
    node = None
    for name in names:
        node = next((n for n in nodes if (n.child_by_field_name("name") is not None and
                     _text(src, n.child_by_field_name("name")) == name)), None)
        if node is None:
            return None
        nodes = [m for m in _members(node) if m.type in TYPE_DECLS]
    return node


def supertype_views(index: dict, sources: dict, cut: str, test_pkg: str, cut_ids: set[str],
                    nested: bool = False) -> dict[str, str]:
    """What the CUT inherits and does not override, nearest supertype first.

    nested (v6): walk the chain through nested superclasses too (index `type_supers`), and
    render a nested supertype from inside its outer file. Keyed by the outer file's FQN;
    several supertypes from one file are joined."""
    rec = index[cut]
    seen = {_index_sig(m) for m in rec["members"] if m["kind"] == "method" and not m["declared_in"]}
    if nested and "type_supers" in rec:
        return _nested_supertype_views(index, sources, cut, test_pkg, cut_ids, seen)
    order, todo = [], list(rec["extends"]) + list(rec["implements"])
    while todo:                                   # breadth-first: nearest wins a signature
        a = todo.pop(0)
        if a in index and a not in order:
            order.append(a)
            todo.extend(index[a]["extends"] + index[a]["implements"])
    out = {}
    for a in order:
        f = sources.get(a)
        if f is None:
            continue
        same = index[a]["package"] == test_pkg
        rendered_here = set()
        src = f.file_content.encode("utf-8")

        def keep(m, n, v, same=same, rendered_here=rendered_here, src=src):
            if m.type in TYPE_DECLS or m.type == "constructor_declaration":
                return False
            if not accessible(v, same):
                return False
            if m.type == "method_declaration":
                sig = _source_sig(m, src)
                if sig in seen:
                    return False
                rendered_here.add(sig)
            return True
        text = _render_file(f, keep, lambda n: n in cut_ids)
        seen |= rendered_here
        if text:
            out[a] = text
    return out


def _nested_supertype_views(index, sources, cut, test_pkg, cut_ids, seen) -> dict[str, str]:
    def supers_of(binary: str) -> list[str]:
        return index.get(binary.split("$", 1)[0], {}).get("type_supers", {}).get(binary, [])

    order, todo = [], list(supers_of(cut))
    while todo:                                   # breadth-first: nearest wins a signature
        a = todo.pop(0)
        if a.split("$", 1)[0] in index and a not in order and a.split("$", 1)[0] != cut:
            order.append(a)
            todo.extend(supers_of(a))
    out: dict[str, list[str]] = {}
    for a in order:
        outer = a.split("$", 1)[0]
        f = sources.get(outer)
        if f is None:
            continue
        src = f.file_content.encode("utf-8")
        decl = _find_type(parser.parse(src).root_node, src, a)
        if decl is None:
            continue
        same = index[outer]["package"] == test_pkg
        rendered_here = set()

        def keep(m, n, v, same=same, rendered_here=rendered_here, src=src):
            if m.type in TYPE_DECLS or m.type == "constructor_declaration":
                return False
            if not accessible(v, same):
                return False
            if m.type == "method_declaration":
                sig = _source_sig(m, src)
                if sig in seen:
                    return False
                rendered_here.add(sig)
            return True
        text = _render_type(decl, src, keep, lambda n: n in cut_ids)
        seen |= rendered_here
        if text:
            out.setdefault(outer, []).append(text)
    return {outer: "\n\n".join(texts) for outer, texts in out.items()}


# ============================================================== recipes
def _call(name: str, params: list[str]) -> str:
    return f"{name}({', '.join(_simple(p) for p in params)})"


def build_recipes(index: dict, t: str, test_pkg: str, enum_constants: bool = False) -> list[str]:
    """How to build a T: accessible constructors and static factories, fewest arguments first.

    enum_constants (v5): for an enum, "building one" means naming a constant, so list them
    all. v4 gave only valueOf(String), and a weak teacher guessed the names
    (AUTOCLOSE_SOURCE for AUTO_CLOSE_SOURCE, 20 failed compiles in one run)."""
    rec = index[t]
    same = rec["package"] == test_pkg
    simple = t.rsplit(".", 1)[-1]
    if enum_constants and rec["kind"] == "enum":
        return [f"{simple}.{m['name']}" for m in rec["members"]
                if m["kind"] == "field" and m["static"] and not m["declared_in"]
                and m["returns"] == t and accessible(m["visibility"], same)]
    ctors = sorted((m for m in rec["members"] if m["kind"] == "ctor" and not m["declared_in"]
                    and accessible(m["visibility"], same)), key=lambda m: len(m["params"]))
    # A factory returns the type or one of its nested types: Base64OutputStream.builder()
    # returns Base64OutputStream$Builder.
    factories = sorted((m for m in rec["members"] if m["kind"] == "method" and m["static"]
                        and not m["declared_in"] and accessible(m["visibility"], same)
                        and m["returns"].split("$", 1)[0] == t), key=lambda m: len(m["params"]))
    recipes = [f"new {_call(simple, m['params'])}" for m in ctors[:MAX_RECIPES_PER_TYPE]]
    recipes += [f"{simple}.{_call(m['name'], m['params'])}" for m in factories[:MAX_RECIPES_PER_TYPE]]
    return recipes


def producer_recipes(index: dict, cut: str, test_pkg: str) -> list[str]:
    """How to obtain an instance of the CUT through the classes that construct it."""
    out = []
    for p in index[cut]["constructed_by"]:
        producer = p["class"]
        if producer not in index:
            continue
        simple = producer.rsplit(".", 1)[-1]
        make = []
        if index[producer]["kind"] in ("abstract", "interface"):
            # An abstract producer cannot be built: start from its smallest concrete subclass.
            for sub in sorted(index[producer]["implemented_by"],
                              key=lambda s: len(index[s]["members"]) if s in index else 1 << 30):
                if index.get(sub, {}).get("kind") in ("class", "enum", "record"):
                    make = build_recipes(index, sub, test_pkg)
                    if make:
                        break
        else:
            make = build_recipes(index, producer, test_pkg)
        receiver = make[0] if make else f"a{'n' if simple[0] in 'AEIOU' else ''} {simple}"
        own = {(m["name"], tuple(m["params"])) for m in index[producer]["members"]}
        for decl in p["entries"][:MAX_ENTRIES_PER_PRODUCER]:
            head, rest = decl.split("(", 1)
            name = head.split()[-1]
            params = [x.strip() for x in rest.split(")", 1)[0].split(",") if x.strip()]
            origin = ""
            if (name, tuple(params)) not in own:
                owner = next((a for a in index[producer]["ancestors"]
                              if (name, tuple(params)) in {(m["name"], tuple(m["params"]))
                                                           for m in index[a]["members"]}), None)
                origin = f"   (inherited from {owner.rsplit('.', 1)[-1]})" if owner else ""
            call = _call(name, params)
            out.append(f"{simple}.{call}" if " static " in f" {head} " else f"{receiver}.{call}{origin}")
    return out


# ============================================================== one candidate
def render_candidate(cand, f: SourceCodeFileData, index: dict, cut: str, test_pkg: str,
                     cut_ids: set[str], supers: dict[str, str], producer_lines: list[str],
                     enum_constants: bool = False, nested: bool = False) -> str:
    """The text one walk candidate contributes, by its role. Types shown as a partial view
    (supertype, used privately, returned) also get their build recipes: a test often has to
    construct them even when the CUT only calls or returns them.
    nested (v6): a file holding a NESTED superclass counts as a supertype too."""
    rec = index.get(cand.fqn, {})
    ancestors = set(index.get(cut, {}).get("ancestors", []))
    nested_only = False
    if nested:
        binary = index.get(cut, {}).get("binary_ancestors", [])
        ancestors |= {a for a in binary if "$" not in a}
        # A file that is an ancestor ONLY through a nested class (BufferRecycler implements
        # RecyclerPool.WithPool) keeps its normal role -- RecyclerPool is also a type the CUT
        # uses -- and gets the nested supertype's inherited members appended. Treating the
        # whole file as a supertype dropped RecyclerPool's API entirely.
        nested_only = cand.fqn not in ancestors and any(a.split("$", 1)[0] == cand.fqn for a in binary)
    same = rec.get("package") == test_pkg
    simple = cand.fqn.rsplit(".", 1)[-1]
    called = set(index.get(cut, {}).get("calls", {}).get(cand.fqn, []))

    def with_recipes(text: str | None) -> str:
        recipes = build_recipes(index, cand.fqn, test_pkg, enum_constants) \
            if rec.get("kind") in ("class", "enum", "record") else []
        # Plain text, not // comments: the prompt's comment stripper would delete those.
        how = f"How to build {article(simple)}:\n" + "\n".join(f"    {r}" for r in recipes) if recipes else ""
        return "\n".join(p for p in [text or "", how] if p)

    if cand.fqn in ancestors:
        return with_recipes(supers.get(cand.fqn))
    if cand.pattern == "producers":
        text = "\n".join(l for l in producer_lines if l.startswith(("new " + simple + "(", simple + ".")))
    elif cand.pattern in BUILD_ROLES:
        text = "\n".join(build_recipes(index, cand.fqn, test_pkg, enum_constants))
    elif cand.pattern == "results":
        text = with_recipes(results_view(f, same))
    elif cand.pattern == "used_privately":
        text = with_recipes(called_view(f, same, called))
    else:
        text = surface_view(f, same, cut_ids) or ""
    if nested_only and supers.get(cand.fqn) and supers[cand.fqn] not in text:
        text = "\n".join(p for p in (text, supers[cand.fqn]) if p)
    return text


def article(name: str) -> str:
    """'an EncoderException', 'a Soundex'."""
    return ("an " if name[:1] in "AEIO" else "a ") + name      # "a UTF8...", "a Utils"
