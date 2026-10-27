"""
Atomic, idempotent bulk import of named graphs into a
[`Dataset`][rdflib.graph.Dataset].

Moving a batch of named graphs between environments with plain
[`Dataset.parse`][rdflib.graph.Dataset.parse] has two failure modes:

* parsing can fail halfway through a document, leaving a half-loaded
  dataset in the target;
* blank node identifiers from the source (e.g. parser generated
  ``_:n<random>b1`` labels) can collide with identifiers already used
  by the target, silently merging unrelated nodes.

This module implements a two phase import instead:

1.  [`stage_import`][rdflib.bulk.stage_import] parses the source into an
    isolated, throwaway in-memory [`Dataset`][rdflib.graph.Dataset] and
    produces an [`ImportCheck`][rdflib.bulk.ImportCheck] describing the
    graphs, statements and blank nodes found. The target is not touched,
    so a parse failure can never leave a partial dataset behind.
2.  [`StagedImport.commit`][rdflib.bulk.StagedImport.commit] writes
    everything at once. Blank nodes are remapped to deterministic,
    content derived labels that cannot collide with unrelated existing
    data, and named graphs are replaced wholesale by default. Stores
    with native transaction support commit as one unit; other stores get
    a compensating rollback that restores the previous state if a write
    fails.

Re-running the same import is a no-op: identical input always remaps to
identical blank node labels, and only statements that are actually new
or replaced are written. [`bulk_import`][rdflib.bulk.bulk_import] runs
both phases in one call and returns an [`ImportReport`][rdflib.bulk.ImportReport]
listing the added and replaced statements per graph.

```python
>>> from rdflib import Dataset
>>> from rdflib.bulk import bulk_import, stage_import
>>> target = Dataset()
>>> staged = stage_import(
...     target,
...     data="@prefix ex: <http://example.org/> . ex:g1 { ex:s ex:p 'x' . }",
...     format="trig",
... )
>>> staged.check.ok
True
>>> report = staged.commit()
>>> report.total_added
1
>>> bulk_import(  # re-running the same import changes nothing
...     target,
...     data="@prefix ex: <http://example.org/> . ex:g1 { ex:s ex:p 'x' . }",
...     format="trig",
... ).is_noop
True

```
"""

from __future__ import annotations

__all__ = [
    "BulkImportError",
    "GraphChange",
    "GraphCheck",
    "ImportCheck",
    "ImportReport",
    "BNodeCheck",
    "StagedImport",
    "bulk_import",
    "inspect_import",
    "stage_import",
]

from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from typing import Literal as StrLiteral

from rdflib.compare import Color, _TripleCanonicalizer
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Dataset, Graph
from rdflib.term import BNode, IdentifiedNode, Literal, Node, URIRef

_BNodePolicy = StrLiteral["canonical", "keep"]
_Triple = tuple[Node, Node, Node]
_Quad = tuple[Node, Node, Node, IdentifiedNode]

#: Predicate of the internal marker triples that record which graph(s) a
#: blank node occurs in, so blank node canonicalization is scoped per graph
#: while preserving nodes shared between graphs.
_GRAPH_MEMBERSHIP_MARKER = URIRef("urn:rdflib:bulk-import:graph-membership")


class BulkImportError(Exception):
    """Raised when an import cannot be staged, fails its checks, or fails
    while writing to the target (in which case the target has been rolled
    back to its previous state when the store permits it)."""


@dataclass(frozen=True)
class GraphCheck:
    """Check result for one graph found in the import source.

    Attributes:
        identifier: Graph identifier, or ``None`` for the default graph.
        is_default: Whether statements go into the dataset default graph.
        exists: Whether a graph with this identifier already has statements
            or is registered in the target.
        statement_count: Number of distinct statements this import places
            in the graph.
        existing_statement_count: Number of statements the target currently
            holds for the graph.
    """

    identifier: IdentifiedNode | None
    is_default: bool
    exists: bool
    statement_count: int
    existing_statement_count: int


@dataclass(frozen=True)
class BNodeCheck:
    """Check result for the blank nodes found in the import source.

    Attributes:
        labels: Distinct blank node labels occurring in the source
            (including blank node graph names).
        collisions: Source labels that are already in use as blank node
            labels in the target. These are harmless with the default
            ``canonical`` blank node policy (the labels are remapped), but
            fatal with the ``keep`` policy.
        mapping: Planned deterministic remapping from original label to the
            label that will actually be written.
    """

    labels: frozenset[str]
    collisions: frozenset[str]
    mapping: Mapping[str, str]

    @property
    def count(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class ImportCheck:
    """Result of inspecting an import source against a target dataset.

    Attributes:
        graphs: One [`GraphCheck`][rdflib.bulk.GraphCheck] per graph found
            in the source, in first-seen order.
        raw_statement_count: Number of statements parsed, counting repeated
            statements and statements repeated across graphs separately.
        statement_count: Number of distinct statements to be written, one
            per distinct quad.
        bnodes: The blank node check result.
        errors: Problems that prevent the import from being committed.
    """

    graphs: tuple[GraphCheck, ...]
    raw_statement_count: int
    statement_count: int
    bnodes: BNodeCheck
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Whether the staged import passed every check."""
        return not self.errors


@dataclass(frozen=True)
class GraphChange:
    """Statements actually changed in one graph during a commit.

    Attributes:
        identifier: Graph identifier, or ``None`` for the default graph.
        created: Whether the graph did not exist in the target before the
            commit.
        added: Statements that were not in the target and were written.
        replaced: Statements previously in the target that were removed
            (always empty in merge mode).
        unchanged: Number of statements that were already present and were
            left as-is.
    """

    identifier: IdentifiedNode | None
    created: bool
    added: frozenset[_Triple]
    replaced: frozenset[_Triple]
    unchanged: int


@dataclass(frozen=True)
class ImportReport:
    """Result of committing a staged import.

    Attributes:
        check: The check result the commit was based on.
        changes: One [`GraphChange`][rdflib.bulk.GraphChange] per imported
            graph, in first-seen order.
        dry_run: Whether this report was produced by a dry run without
            touching the target.
    """

    check: ImportCheck
    changes: tuple[GraphChange, ...]
    dry_run: bool

    @property
    def total_added(self) -> int:
        return sum(len(change.added) for change in self.changes)

    @property
    def total_replaced(self) -> int:
        return sum(len(change.replaced) for change in self.changes)

    @property
    def is_noop(self) -> bool:
        """Whether the commit wrote or removed nothing."""
        return self.total_added == 0 and self.total_replaced == 0


def _display_identifier(identifier: IdentifiedNode) -> IdentifiedNode | None:
    return None if identifier == DATASET_DEFAULT_GRAPH_ID else identifier


def _target_contexts(target: Dataset) -> set[IdentifiedNode]:
    return {
        context.identifier
        for context in target.store.contexts()
        if isinstance(context, Graph)
    }


def _target_bnode_labels(target: Dataset) -> set[str]:
    labels: set[str] = set()
    for s, p, o, g in target.quads((None, None, None, None)):
        for term in (s, o, g):
            if isinstance(term, BNode):
                labels.add(str(term))
    return labels


def _triples_of(target: Dataset, identifier: IdentifiedNode) -> set[_Triple]:
    context = target.get_context(identifier)
    return set(context.triples((None, None, None)))


def _ground_color(term: Node) -> str:
    return "G\x00" + term.n3()


def _wl_seeds(quads: list[_Quad]) -> dict[BNode, str]:
    """Seed colour for every statement blank node.

    Runs a 1-dimensional Weisfeiler-Lehman colour refinement over the union
    of all statements, augmented with marker edges tying a blank node to
    every graph it occurs in. The seeds are stable across parses (they only
    depend on statement structure and graph identifiers, not on the random
    labels parsers assign) and distinguish a single blank node by its
    contents, unlike a plain canonical graph digest.
    """

    bnodes: set[BNode] = set()
    adjacency: dict[BNode, list[tuple[str, str, Node]]] = defaultdict(list)
    for s, p, o, g in quads:
        if isinstance(s, BNode):
            bnodes.add(s)
            adjacency[s].append(("+", p.n3(), o))
        if isinstance(o, BNode):
            bnodes.add(o)
            adjacency[o].append(("-", p.n3(), s))
        graph_marker = Literal(str(g))
        if isinstance(s, BNode):
            adjacency[s].append(("~", _GRAPH_MEMBERSHIP_MARKER.n3(), graph_marker))
        if isinstance(o, BNode):
            adjacency[o].append(("~", _GRAPH_MEMBERSHIP_MARKER.n3(), graph_marker))

    if not bnodes:
        return {}

    def signature(node: BNode, colors: Mapping[BNode, str]) -> str:
        # Sort the incident edge descriptions: quad iteration order from the
        # store depends on parser generated blank node labels, so the
        # signature must not depend on it.
        parts = [
            direction + "\x1f" + predicate + "\x1f" + neighbor
            for direction, predicate, other in adjacency[node]
            for neighbor in (
                [colors[other]] if isinstance(other, BNode) else [_ground_color(other)]
            )
        ]
        return colors[node] + "\x1e" + "\x1e".join(sorted(parts))

    def partition(members_colors: Mapping[BNode, str]) -> set[frozenset[BNode]]:
        groups: dict[str, set[BNode]] = defaultdict(set)
        for node, color in members_colors.items():
            groups[color].add(node)
        return {frozenset(group) for group in groups.values()}

    colors = {node: "B0" for node in bnodes}
    current_partition = partition(colors)
    while True:
        updated = {
            node: "B:" + sha256(signature(node, colors).encode("utf-8")).hexdigest()
            for node in bnodes
        }
        next_partition = partition(updated)
        if next_partition == current_partition:
            return updated
        colors = updated
        current_partition = next_partition


class _SeededTripleCanonicalizer(_TripleCanonicalizer):
    """Variant of the RGDA1 canonicalizer whose blank nodes start out with
    pre-computed Weisfeiler-Lehman seed colours instead of a single empty
    colour. Blank nodes with different structural environments are already
    distinguished by their seed; the standard refinement and trace
    algorithms only split structurally symmetric blank nodes."""

    def __init__(self, graph: Graph, seeds: Mapping[BNode, str]) -> None:
        super().__init__(graph)
        self._seeds = dict(seeds)

    def _initial_color(self) -> list[Color]:
        bnodes: set[BNode] = set()
        others = set()
        for s, p, o in self.graph:
            nodes = {s, p, o}
            blank = {x for x in nodes if isinstance(x, BNode)}
            if blank:
                others |= nodes - blank
                bnodes |= blank
        blank_colors = [
            Color(
                [node],
                self.hashfunc,
                (self._seeds[node],),  # type: ignore[arg-type]
                hash_cache=self._hash_cache,
            )
            for node in sorted(bnodes, key=lambda node: self._seeds[node])
        ]
        ground_colors = [
            # The parent implementation carries the same type ignores: the
            # ground node itself is used as the initial colour.
            Color([x], self.hashfunc, x, hash_cache=self._hash_cache)  # type: ignore[list-item, arg-type]
            for x in sorted(others, key=lambda node: node.n3())
        ]
        return blank_colors + ground_colors


def _canonical_bnode_mapping(
    quads: list[_Quad],
) -> dict[BNode, BNode]:
    """Map every statement blank node to a deterministic, content derived
    ``bulk<color>`` label.

    Weisfeiler-Lehman seeds distinguish blank nodes by their structural
    environment and graph membership; the canonicalizer from
    [`rdflib.compare`][rdflib.compare] then resolves structurally symmetric
    blank nodes deterministically. The resulting label does not depend on
    the random labels parsers assign.
    """

    seeds = _wl_seeds(quads)
    if not seeds:
        return {}

    union = Graph()
    for s, p, o, g in quads:
        union.add((s, p, o))
        marker_object = Literal(str(g))
        if isinstance(s, BNode):
            union.add((s, _GRAPH_MEMBERSHIP_MARKER, marker_object))
        if isinstance(o, BNode):
            union.add((o, _GRAPH_MEMBERSHIP_MARKER, marker_object))

    canonicalizer = _SeededTripleCanonicalizer(union, seeds)
    colors: dict[BNode, str] = {}
    canonical_triples = canonicalizer.canonical_triples()
    for original_triple, canonical_triple in zip(union, canonical_triples):
        for original, canonical in zip(original_triple, canonical_triple):
            if isinstance(original, BNode):
                color = str(canonical)[len("cb") :]
                previous = colors.setdefault(original, color)
                if previous != color:
                    msg = (
                        "internal error: inconsistent canonical blank node "
                        "label produced by rdflib.compare"
                    )
                    raise BulkImportError(msg)

    return {node: BNode("bulk" + color) for node, color in colors.items()}


#: Anchor URI used as a stand-in graph identifier while fingerprinting
#: graphs whose name is itself a blank node (whose parser label is random).
_BNODE_GRAPH_ANCHOR = "urn:rdflib:bulk-import:bnode-graph#"


def _bnode_graph_fingerprint(triples: list[_Triple]) -> str:
    """Stable content fingerprint of a graph with a blank node name.

    The statements are canonicalized in isolation against a fixed anchor
    graph, so the fingerprint does not depend on parser generated labels.
    Two blank-node-named graphs with identical statements get the same
    fingerprint and are treated as the same graph.
    """

    anchor = URIRef(_BNODE_GRAPH_ANCHOR)
    pseudo_quads: list[_Quad] = [(s, p, o, anchor) for s, p, o in triples]
    mapping = _canonical_bnode_mapping(pseudo_quads)
    lines = sorted(
        " ".join(
            (mapping.get(term, term) if isinstance(term, BNode) else term).n3()
            for term in triple
        )
        for triple in set(triples)
    )
    return sha256(("\n".join(lines)).encode("utf-8")).hexdigest()


class StagedImport:
    """An import source that has been parsed and checked but not committed.

    Instances are created with
    [`stage_import`][rdflib.bulk.stage_import]. The
    [`check`][rdflib.bulk.StagedImport.check] attribute describes graphs,
    statements and blank nodes; [`commit`][rdflib.bulk.StagedImport.commit]
    performs the atomic write.
    """

    def __init__(
        self,
        target: Dataset,
        parse_kwargs: Mapping[str, Any],
    ) -> None:
        self._target = target
        self._parse_kwargs = dict(parse_kwargs)
        self._stage()

    def _stage(self) -> None:
        staged = Dataset()
        try:
            staged.parse(**self._parse_kwargs)
        except Exception as error:
            msg = (
                "failed to parse import source; target dataset was not "
                f"modified: {error}"
            )
            raise BulkImportError(msg) from error

        raw_quads: list[_Quad] = []
        graph_triples: OrderedDict[IdentifiedNode, list[_Triple]] = OrderedDict()
        for s, p, o, g in staged.quads((None, None, None, None)):
            identifier = DATASET_DEFAULT_GRAPH_ID if g is None else g
            raw_quads.append((s, p, o, identifier))
            graph_triples.setdefault(identifier, []).append((s, p, o))

        # Blank node graph names carry random parser labels, so give them
        # stable content derived identifiers and use the stable identifier
        # as graph membership anchor while canonicalizing statement nodes.
        graph_anchors: dict[BNode, URIRef] = {}
        bnode_graph_names: dict[BNode, BNode] = {}
        for identifier, triples in graph_triples.items():
            if isinstance(identifier, BNode):
                fingerprint = _bnode_graph_fingerprint(triples)
                graph_anchors[identifier] = URIRef(_BNODE_GRAPH_ANCHOR + fingerprint)
                bnode_graph_names[identifier] = BNode("bulkgn" + fingerprint)
        analysis_quads = [
            (
                s,
                p,
                o,
                graph_anchors[g] if isinstance(g, BNode) else g,
            )
            for s, p, o, g in raw_quads
        ]
        bnode_mapping = _canonical_bnode_mapping(analysis_quads)
        bnode_mapping.update(bnode_graph_names)

        source_labels = {
            str(term)
            for s, p, o, g in raw_quads
            for term in (s, o, g)
            if isinstance(term, BNode)
        }
        target_labels = _target_bnode_labels(self._target)
        collisions = frozenset(source_labels & target_labels)

        existing_contexts = _target_contexts(self._target)
        graph_checks: list[GraphCheck] = []
        distinct_statements = 0
        for identifier, triples in graph_triples.items():
            unique_triples = set(triples)
            distinct_statements += len(unique_triples)
            exists = identifier in existing_contexts or (
                identifier == DATASET_DEFAULT_GRAPH_ID
            )
            existing_count = len(_triples_of(self._target, identifier)) if exists else 0
            graph_checks.append(
                GraphCheck(
                    identifier=_display_identifier(identifier),
                    is_default=identifier == DATASET_DEFAULT_GRAPH_ID,
                    exists=exists,
                    statement_count=len(unique_triples),
                    existing_statement_count=existing_count,
                )
            )

        mapping = OrderedDict(
            (str(source), str(mapped))
            for source, mapped in sorted(
                bnode_mapping.items(), key=lambda item: str(item[0])
            )
            if str(source) != str(mapped)
        )
        self._raw_quads = raw_quads
        self._bnode_mapping = bnode_mapping
        self.check = ImportCheck(
            graphs=tuple(graph_checks),
            raw_statement_count=len(raw_quads),
            statement_count=distinct_statements,
            bnodes=BNodeCheck(
                labels=frozenset(source_labels),
                collisions=collisions,
                mapping=mapping,
            ),
            errors=(),
        )

    @property
    def target(self) -> Dataset:
        """The dataset the import will be written to."""
        return self._target

    def _effective_quads(self, policy: _BNodePolicy) -> list[_Quad]:
        quads: list[_Quad] = []
        for s, p, o, g in self._raw_quads:
            if policy == "canonical":
                mapping = self._bnode_mapping
                s, p, o, g = (
                    mapping.get(s, s) if isinstance(s, BNode) else s,
                    p,
                    mapping.get(o, o) if isinstance(o, BNode) else o,
                    mapping.get(g, g) if isinstance(g, BNode) else g,
                )
            quads.append((s, p, o, g))
        return quads

    def commit(
        self,
        *,
        replace: bool = True,
        bnode_policy: _BNodePolicy = "canonical",
        dry_run: bool = False,
    ) -> ImportReport:
        """Write the staged import into the target dataset.

        Args:
            replace: When ``True`` (the default) each graph present in the
                source fully replaces the target graph of the same name, so
                statements missing from the source are removed. When
                ``False`` statements are only merged in and nothing is
                removed.
            bnode_policy: ``"canonical"`` (the default) remaps blank nodes
                to deterministic, content derived labels. ``"keep"`` keeps
                the source labels as parsed and makes blank node label
                collisions a fatal error.
            dry_run: When ``True`` compute and return the report without
                modifying the target.

        Returns:
            An [`ImportReport`][rdflib.bulk.ImportReport] listing the
            statements actually added and replaced per graph.

        Raises:
            BulkImportError: If the staged import failed its checks, the
                blank node policy cannot be satisfied, or writing fails. In
                the last case the target is rolled back when the store
                permits it.
        """

        if bnode_policy not in ("canonical", "keep"):
            msg = f"unknown blank node policy: {bnode_policy!r}"
            raise BulkImportError(msg)
        if not self.check.ok:
            msg = "refusing to commit an import with check errors: " + "; ".join(
                self.check.errors
            )
            raise BulkImportError(msg)
        if bnode_policy == "keep":
            # Re-check against the current target: the dataset may have
            # changed between staging and committing.
            current_collisions = self.check.bnodes.labels & _target_bnode_labels(
                self._target
            )
            if current_collisions:
                labels = ", ".join(sorted(current_collisions))
                msg = (
                    "blank node labels from the import already exist in the "
                    f"target: {labels}; use bnode_policy='canonical' to remap "
                    "them"
                )
                raise BulkImportError(msg)

        quads = self._effective_quads(bnode_policy)
        incoming: OrderedDict[IdentifiedNode, set[_Triple]] = OrderedDict()
        for s, p, o, identifier in quads:
            incoming.setdefault(identifier, set()).add((s, p, o))

        existing_contexts = _target_contexts(self._target)
        changes: list[GraphChange] = []
        plan: list[tuple[IdentifiedNode, set[_Triple], set[_Triple], bool]] = []
        for identifier, new_triples in incoming.items():
            created = identifier not in existing_contexts and not (
                identifier == DATASET_DEFAULT_GRAPH_ID
            )
            existing = set() if created else _triples_of(self._target, identifier)
            added = new_triples - existing
            replaced = existing - new_triples if replace else set()
            unchanged = len(new_triples & existing)
            changes.append(
                GraphChange(
                    identifier=_display_identifier(identifier),
                    created=created,
                    added=frozenset(added),
                    replaced=frozenset(replaced),
                    unchanged=unchanged,
                )
            )
            plan.append((identifier, added, replaced, created))

        report = ImportReport(
            check=self.check,
            changes=tuple(changes),
            dry_run=dry_run,
        )
        if dry_run:
            return report

        self._apply(plan)
        return report

    def _apply(
        self,
        plan: list[tuple[IdentifiedNode, set[_Triple], set[_Triple], bool]],
    ) -> None:
        target = self._target
        store = target.store

        if store.transaction_aware:
            try:
                for identifier, added, replaced, created in plan:
                    context = target.get_context(identifier)
                    if created:
                        store.add_graph(context)
                    for triple in replaced:
                        store.remove(triple, context=context)
                    store.addN((s, p, o, context) for s, p, o in added)
                store.commit()
            except Exception as error:
                store.rollback()
                msg = (
                    "import failed while writing; the transaction was rolled "
                    f"back: {error}"
                )
                raise BulkImportError(msg) from error
            return

        removed: list[tuple[Node, Node, Node, Graph]] = []
        added_records: list[tuple[Node, Node, Node, Graph]] = []
        created_contexts: list[Graph] = []

        def record(
            quads: Iterator[tuple[Node, Node, Node, Graph]],
        ) -> Iterator[tuple[Node, Node, Node, Graph]]:
            for quad in quads:
                added_records.append(quad)
                yield quad

        try:
            for identifier, added, replaced, created in plan:
                context = target.get_context(identifier)
                if created:
                    store.add_graph(context)
                    created_contexts.append(context)
                for triple in replaced:
                    store.remove(triple, context=context)
                    removed.append((triple[0], triple[1], triple[2], context))
                store.addN(record((s, p, o, context) for s, p, o in added))
        except Exception as error:
            # Best effort compensating rollback: undo the additions,
            # restore every statement that was removed and deregister any
            # graph that only existed because of this import.
            for s, p, o, context in added_records:
                try:
                    store.remove((s, p, o), context=context)
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
            if removed:
                try:
                    store.addN(iter(removed))
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
            for context in created_contexts:
                try:
                    if len(context) == 0:
                        store.remove_graph(context)
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
            msg = (
                "import failed while writing; changes were rolled back as "
                f"far as the store allows: {error}"
            )
            raise BulkImportError(msg) from error

    def __repr__(self) -> str:
        return (
            f"<StagedImport target={self._target!r} "
            f"graphs={len(self.check.graphs)} "
            f"statements={self.check.statement_count} ok={self.check.ok}>"
        )


def stage_import(
    target: Dataset,
    source: Any = None,
    *,
    publicID: str | None = None,  # noqa: N803
    format: str | None = None,
    location: str | None = None,
    file: Any = None,
    data: str | bytes | None = None,
    **parser_args: Any,
) -> StagedImport:
    """Parse and check an import source without modifying the target.

    The source arguments mirror
    [`Dataset.parse`][rdflib.graph.Dataset.parse]. Parsing happens in an
    isolated in-memory dataset: if it fails, the target is guaranteed to be
    untouched.

    Args:
        target: The dataset the source would be committed to. Used for
            collision and existing content checks only.
        source: File name, URL, file-like object or path to parse.
        publicID: Logical URI / document base, as in ``Dataset.parse``.
        format: Source format, e.g. ``"trig"`` or ``"nquads"``. Guessed when
            omitted.
        location: Relative or absolute URL of the source.
        file: File-like object to read from.
        data: Inline source data.
        **parser_args: Extra keyword arguments forwarded to the parser.

    Returns:
        A [`StagedImport`][rdflib.bulk.StagedImport] whose
        [`check`][rdflib.bulk.StagedImport.check] attribute reports on the
        graphs, statements and blank nodes found.

    Raises:
        BulkImportError: If the target is not a
            [`Dataset`][rdflib.graph.Dataset] or the source cannot be
            parsed.
    """

    if not isinstance(target, Dataset):
        msg = (  # type: ignore[unreachable]
            "bulk import requires a rdflib.Dataset target (a store "
            "supporting named graphs), got "
            f"{type(target).__module__}.{type(target).__name__}"
        )
        raise TypeError(msg)
    parse_kwargs = {
        "source": source,
        "publicID": publicID,
        "format": format,
        "location": location,
        "file": file,
        "data": data,
        **parser_args,
    }
    return StagedImport(target, parse_kwargs)


def inspect_import(
    target: Dataset,
    source: Any = None,
    **kwargs: Any,
) -> ImportCheck:
    """Return just the
    [`ImportCheck`][rdflib.bulk.ImportCheck] for an import source.

    See [`stage_import`][rdflib.bulk.stage_import] for the source
    arguments.
    """

    return stage_import(target, source, **kwargs).check


def bulk_import(
    target: Dataset,
    source: Any = None,
    *,
    replace: bool = True,
    bnode_policy: _BNodePolicy = "canonical",
    dry_run: bool = False,
    publicID: str | None = None,  # noqa: N803
    format: str | None = None,
    location: str | None = None,
    file: Any = None,
    data: str | bytes | None = None,
    **parser_args: Any,
) -> ImportReport:
    """Stage and commit an import in a single call.

    See [`StagedImport.commit`][rdflib.bulk.StagedImport.commit] for the
    semantics of ``replace``, ``bnode_policy`` and ``dry_run``, and
    [`stage_import`][rdflib.bulk.stage_import] for the source arguments.

    Returns:
        An [`ImportReport`][rdflib.bulk.ImportReport] listing the
        statements actually added and replaced per graph.

    Raises:
        BulkImportError: If parsing, checking or writing fails. The target
            is left unchanged on parse or check failure, and rolled back on
            write failure.
    """

    staged = stage_import(
        target,
        source,
        publicID=publicID,
        format=format,
        location=location,
        file=file,
        data=data,
        **parser_args,
    )
    return staged.commit(
        replace=replace,
        bnode_policy=bnode_policy,
        dry_run=dry_run,
    )
