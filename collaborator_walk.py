"""The walk: from a class under test (CUT) to candidate collaborators, by lookups in the
collaborator index (collaborators.py). No selection happens here -- every candidate comes
back tagged with the pattern(s) that found it, and scoring/budgeting is a separate step.

Each pattern is a question a test has to answer, mapped to index fields:

  own_surface        what the CUT takes, returns, throws, exposes     own signature fields
  supertype          what the CUT inherits from                      ancestors
  inherited_api      what inherited methods take/return/throw         inherited_api
  ctor_args          how to build what the CUT's constructors take    ctor_params -> their ctor_params
  implementations    a concrete class for an interface parameter      params -> implemented_by
  producers          how to get an instance of the CUT at all         constructed_by
  subclasses         concrete forms of a CUT that is a base class     implemented_by / extended_by
  results            how to inspect what the CUT returns              returns -> their returns
  used_privately     what the CUT's code calls or references          calls, instantiates, references

Each pattern also names the VIEW its candidates should be rendered with, since a type
reached as "something to construct" needs its constructors, not its whole API.

Measured and dropped (smokes/walk_audit.py, 163 classes with logged suites):
  stub_obligations   abstract params -> their surface   218 candidates, 0 needed, 870k chars
  javadoc_link       javadoc_links                       26 candidates, 1 needed
"""
from dataclasses import dataclass, field

# pattern -> view. Order is the priority when one type is reached by several patterns: the
# earlier pattern's view wins (own_surface's full API view beats a constructor-only view).
PATTERNS = {
    "own_surface": "api",
    "producers": "producer",
    "inherited_api": "api",
    "supertype": "inherited",
    "ctor_args": "ctor",
    "implementations": "ctor",
    "subclasses": "ctor",
    "results": "signatures",
    "used_privately": "internal",
}
_SURFACE = ("ctor_params", "method_params", "returns", "throws", "fields")
_CONCRETE = ("class", "enum", "record")


@dataclass
class Candidate:
    fqn: str
    patterns: list = field(default_factory=list)      # [(pattern, hops)], in discovery order
    entries: list = field(default_factory=list)       # producer entry points, if any

    @property
    def pattern(self) -> str:
        """The highest-priority pattern that reached this type."""
        order = list(PATTERNS)
        return min((p for p, _ in self.patterns), key=order.index)

    @property
    def view(self) -> str:
        return PATTERNS[self.pattern]

    @property
    def hops(self) -> int:
        return min(h for _, h in self.patterns)


def _erased_outer(type_name: str) -> str:
    """'a.b.Outer$Inner[]' -> 'a.b.Outer': the file a member type lives in."""
    return type_name.replace("...", "").replace("[]", "").split("$", 1)[0]


def walk(index: dict[str, dict], cut: str, nested_ancestors: bool = False) -> list[Candidate]:
    """Candidates in a FIXED order: by pattern priority, then fully-qualified name. Every
    set is iterated sorted, so the same class always yields the same prompt (Python
    randomises set order per process, which used to reshuffle sections run to run).

    nested_ancestors (v6): follow the superclass chain through nested classes via the
    index's binary_ancestors. `ancestors` stops at a nested superclass, so e.g.
    JsonFactoryBuilder -> DecorableTSFactory$DecorableTSFBuilder -> TSFBuilder saw no
    supertype and no inherited API at all."""
    rec = index.get(cut)
    if rec is None:
        return []
    found: dict[str, Candidate] = {}

    def add(fqn: str, pattern: str, hops: int, entries=None):
        if fqn == cut or fqn not in index:
            return
        c = found.setdefault(fqn, Candidate(fqn))
        if (pattern, hops) not in c.patterns:
            c.patterns.append((pattern, hops))
        if entries:
            c.entries.extend(e for e in entries if e not in c.entries)

    params = set(rec["ctor_params"]) | set(rec["method_params"]) | \
        set(rec["inherited_api"]["method_params"])
    returns = set(rec["returns"]) | set(rec["inherited_api"]["returns"])
    for key in _SURFACE:
        for t in rec[key]:
            add(t, "own_surface", 1)
    for t in rec["ancestors"]:
        add(t, "supertype", 1)
    for values in rec["inherited_api"].values():
        for t in values:
            add(t, "inherited_api", 1)
    if nested_ancestors:
        for anc in rec.get("binary_ancestors", []):
            outer, _, nested = anc.partition("$")
            add(outer, "supertype", 1)
            for m in index.get(outer, {}).get("members", []):
                if m["declared_in"] != nested or m["visibility"] == "private" or m["kind"] == "ctor":
                    continue
                for t in m["params"] + ([m["returns"]] if m["returns"] else []):
                    add(_erased_outer(t), "inherited_api", 1)
                params |= {_erased_outer(t) for t in m["params"]}
                if m["kind"] == "method" and m["returns"]:
                    returns.add(_erased_outer(m["returns"]))
    for p in rec["constructed_by"]:
        add(p["class"], "producers", 1, p["entries"])
    if rec["kind"] in ("abstract", "interface") or rec["extended_by"]:
        for t in sorted(set(rec["implemented_by"]) | set(rec["extended_by"])):
            if index.get(t, {}).get("kind") in _CONCRETE:
                add(t, "subclasses", 1)
    for t in sorted(set(rec["calls"]) | set(rec["instantiates"]) | set(rec["references"])):
        add(t, "used_privately", 1)

    for p in sorted(params):
        prec = index.get(p)
        if prec is None:
            continue
        for t in prec["ctor_params"]:
            add(t, "ctor_args", 2)
        if prec["kind"] in ("interface", "abstract"):
            for t in prec["implemented_by"]:
                if index.get(t, {}).get("kind") in _CONCRETE:
                    add(t, "implementations", 2)
    for r in sorted(returns):
        rrec = index.get(r)
        if rrec is None:
            continue
        for t in rrec["returns"]:
            add(t, "results", 2)

    order = list(PATTERNS)
    return sorted(found.values(), key=lambda c: (order.index(c.pattern), c.fqn))
