"""
Preview SPARQL Update requests before they are written to a graph.

This module implements a *preview-then-commit* workflow for SPARQL 1.1
Update requests:

1. [`preview_update`][rdflib.plugins.sparql.preview.preview_update] executes
   the request against an in-memory *shadow* copy of the target graph, so the
   real graph is never touched while previewing. It returns an
   [`UpdatePlan`][rdflib.plugins.sparql.preview.UpdatePlan] describing the
   quads that would be added, removed and left unchanged, together with
   per-statement effects and conflicts.
2. After review, [`UpdatePlan.apply`][rdflib.plugins.sparql.preview.UpdatePlan.apply]
   writes the *materialized* preview diff (not a re-execution of the query
   text) to the target, so the outcome is identical to the preview, rolls
   back automatically if any write fails, and refuses to commit when the
   target data or the request text changed between preview and commit.

Notes and limitations:

- Quads are represented as ``(subject, predicate, object, graph_id)``
  tuples. For a plain [`Graph`][rdflib.graph.Graph] the graph id is the
  graph's own identifier.
- Previewing takes a full snapshot before and after every operation,
  i.e. ``O(n * k)`` where ``n`` is the number of quads and ``k`` the number
  of operations. This is intended for analyst-sized temporary datasets.
- ``LOAD`` performs its network/file access while previewing, like a normal
  update does while executing; the fetched quads are reused at commit time,
  so commit itself performs no second fetch.
- Atomic rollback is guaranteed for local stores (e.g. the default in-memory
  store). For remote stores such as ``SPARQLStore`` restoration is
  best-effort. The stale-state protection works against every store.
- Empty named graphs carry no quads, so ``CREATE``/``DROP`` of an empty
  graph is not visible in the quad-based diff.

!!! warning "Security Considerations"

    Previewing can access indirectly requested network endpoints, for example
    ``LOAD`` will attempt to fetch the referenced IRIs. When processing
    untrusted or potentially malicious requests, measures should be taken to
    restrict network and file access.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from rdflib.graph import ConjunctiveGraph, Dataset, Graph
from rdflib.plugins.sparql.algebra import translateUpdate
from rdflib.plugins.sparql.parser import parseUpdate
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import QueryContext, Update
from rdflib.plugins.sparql.update import UpdateObserver, evalUpdate
from rdflib.term import Identifier, Node

#: A quad as used throughout the preview: ``(s, p, o, graph_identifier)``.
Quad = tuple[Identifier, Identifier, Identifier, Identifier]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UpdatePreviewError(Exception):
    """Raised when an update request cannot be previewed.

    The failure happens on the shadow copy, so the target graph is always
    unchanged when this is raised.
    """

    def __init__(
        self,
        index: int,
        operation: str,
        original: BaseException,
    ) -> None:
        self.statement_index = index
        self.operation = operation
        self.original = original
        super().__init__(
            f"Statement {index + 1} ({operation}) failed during preview: {original}"
        )


class StaleUpdatePlanError(Exception):
    """Raised when a plan can no longer be applied as previewed.

    Either the target graph changed after the preview, or the request text
    passed to :meth:`UpdatePlan.apply` differs from the previewed text.
    Nothing has been written when this is raised.
    """

    def __init__(
        self,
        message: str,
        *,
        external_additions: frozenset[Quad] = frozenset(),
        external_deletions: frozenset[Quad] = frozenset(),
    ) -> None:
        self.external_additions = external_additions
        self.external_deletions = external_deletions
        super().__init__(message)


class UpdateApplicationError(Exception):
    """Raised when committing a plan failed.

    The target graph has been restored to its pre-commit contents (for local
    stores) before this is raised; ``__cause__`` holds the original write
    error.
    """


# ---------------------------------------------------------------------------
# Preview result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatementEffect:
    """The actual effect of one statement of the update request."""

    #: Zero-based position of the statement in the request.
    index: int
    #: Algebra operation name, e.g. ``"InsertData"``.
    operation: str
    #: Whether the statement carried a ``SILENT`` modifier.
    silent: bool
    #: Whether the statement succeeded on the shadow (``False`` for swallowed
    #: ``SILENT`` failures).
    succeeded: bool
    #: Quads actually added by this statement.
    added: frozenset[Quad]
    #: Quads actually removed by this statement.
    removed: frozenset[Quad]


@dataclass(frozen=True)
class UpdateConflict:
    """A conflict or warning involving one or more statements."""

    #: ``"delete_missing"``, ``"insert_existing"`` or ``"statement_order"``.
    kind: str
    #: ``"conflict"`` (the statement targets data that does not exist) or
    #: ``"warning"`` (legal but redundant/order-dependent behaviour).
    severity: str
    #: Zero-based indices of the statements involved.
    statements: tuple[int, ...]
    #: Affected quads.
    quads: frozenset[Quad]
    #: Human-readable explanation.
    message: str


@dataclass(frozen=True)
class UpdatePreview:
    """The projected outcome of an update request."""

    #: Quads present after the request but not before it.
    additions: frozenset[Quad]
    #: Quads present before the request but not after it.
    deletions: frozenset[Quad]
    #: Number of quads present both before and after the request.
    unchanged_count: int
    #: Quads explicitly referenced by statements that end up unchanged,
    #: e.g. an ``INSERT DATA`` of already present triples, or an add/remove
    #: pair that cancels out.
    unchanged_referenced: frozenset[Quad]
    #: Per-statement effects, in request order.
    statements: tuple[StatementEffect, ...]
    #: Conflicts and warnings, ordered by the first statement involved.
    conflicts: tuple[UpdateConflict, ...]
    #: SHA-256 of the request text the preview was built from, if available.
    request_hash: str | None

    @property
    def added_count(self) -> int:
        return len(self.additions)

    @property
    def deleted_count(self) -> int:
        return len(self.deletions)

    def summary(self, max_quads: int = 10) -> str:
        """Return a short human-readable summary of the projected changes."""

        lines = [
            f"Projected changes: {len(self.additions)} addition(s), "
            f"{len(self.deletions)} deletion(s), "
            f"{self.unchanged_count} unchanged quad(s).",
        ]
        for title, quads in (
            ("Will add", self.additions),
            ("Will delete", self.deletions),
        ):
            shown = sorted(quads, key=_quad_key)[:max_quads]
            for quad in shown:
                lines.append(f"  {title}: {_format_quad(quad)}")
            remaining = len(quads) - len(shown)
            if remaining > 0:
                lines.append(f"  {title}: ... and {remaining} more")
        for conflict in self.conflicts:
            involved = ", ".join(str(i + 1) for i in conflict.statements)
            lines.append(
                f"  [{conflict.severity}] {conflict.kind} "
                f"(statement(s) {involved}): {conflict.message}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Snapshotting and shadow copies
# ---------------------------------------------------------------------------


def _make_quad(s: Node, p: Node, o: Node, g: Identifier) -> Quad:
    return cast(Quad, (s, p, o, g))


def _snapshot(graph: Graph) -> frozenset[Quad]:
    """Return all quads of *graph* as a hashable set."""

    snapshot: set[Quad] = set()
    if isinstance(graph, ConjunctiveGraph):
        for s, p, o, c in graph.quads((None, None, None, None)):
            snapshot.add(_make_quad(s, p, o, _context_id(c)))
    else:
        for s, p, o in graph.triples((None, None, None)):
            snapshot.add(_make_quad(s, p, o, graph.identifier))
    return frozenset(snapshot)


def _context_id(context: object) -> Identifier:
    return cast(Identifier, getattr(context, "identifier", context))


def _quad_key(quad: Quad) -> tuple[str, str, str, str]:
    return cast(tuple[str, str, str, str], tuple(term.n3() for term in quad))


def _text_token(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _shadow_graph(graph: Graph) -> Graph:
    """Build an independent in-memory copy of *graph* with identical ids."""

    if isinstance(graph, Dataset):
        dataset_shadow = _ShadowDataset(default_union=graph.default_union)
        quads = [
            (s, p, o, dataset_shadow.get_context(_context_id(c)))
            for s, p, o, c in graph.quads((None, None, None, None))
        ]
        dataset_shadow.addN(quads)
        shadow: Graph = dataset_shadow
    elif isinstance(graph, ConjunctiveGraph):
        conjunctive_shadow = ConjunctiveGraph(
            identifier=graph.default_context.identifier
        )
        conjunctive_shadow.default_union = graph.default_union
        quads = [
            (s, p, o, conjunctive_shadow.get_context(_context_id(c)))
            for s, p, o, c in graph.quads((None, None, None, None))
        ]
        conjunctive_shadow.addN(quads)
        shadow = conjunctive_shadow
    else:
        shadow = Graph(identifier=graph.identifier)
        for triple in graph.triples((None, None, None)):
            shadow.add(triple)
    return shadow


def _default_graph_id(graph: ConjunctiveGraph) -> Identifier:
    default_graph = getattr(graph, "default_graph", None)
    if default_graph is not None:
        return default_graph.identifier
    return graph.default_context.identifier


class _ShadowDataset(Dataset):
    """Dataset whose ``+=`` / ``-=`` accept triples as well as quads.

    ``QueryContext`` sets ``ctx.graph`` to the dataset itself when
    ``SPARQL_DEFAULT_GRAPH_UNION`` is enabled, and the update evaluator then
    performs ``dataset += triples``. ``Dataset.__iadd__`` only accepts quads,
    so ground triples currently raise ``ValueError`` there. Triples target
    the default graph in SPARQL Update (this is also what ``evalModify``
    already does explicitly), so route them there on the shadow copy.
    Iterables of quads and graphs are handled just like the base classes.
    """

    def __iadd__(self, other):  # type: ignore[misc]
        def normalised():
            for item in other:
                if len(item) == 4:
                    s, p, o, context = item
                    if isinstance(context, Graph):
                        yield s, p, o, context
                    else:
                        yield s, p, o, self.get_context(context)
                else:
                    s, p, o = item
                    yield s, p, o, self.default_graph

        self.addN(normalised())
        return self

    def __isub__(self, other):  # type: ignore[misc]
        for item in other:
            if len(item) == 4:
                s, p, o, context = item
                if not isinstance(context, Graph):
                    context = self.get_context(context)
                self.remove((s, p, o, context))
            else:
                self.default_graph.remove(item)
        return self


def _format_quad(quad: Quad) -> str:
    s, p, o, g = quad
    return f"{s.n3()} {p.n3()} {o.n3()} {g.n3()}"


# ---------------------------------------------------------------------------
# Recording observer
# ---------------------------------------------------------------------------


@dataclass
class _OperationRecord:
    index: int
    operation: str
    silent: bool
    succeeded: bool
    before: frozenset[Quad]
    after: frozenset[Quad]
    declared: frozenset[Quad] = frozenset()

    @property
    def added(self) -> frozenset[Quad]:
        return self.after - self.before

    @property
    def removed(self) -> frozenset[Quad]:
        return self.before - self.after


class _RecordingObserver(UpdateObserver):
    def __init__(self, shadow: Graph) -> None:
        self.shadow = shadow
        self.records: list[_OperationRecord] = []
        self.last_index = -1
        self.last_operation = ""
        self._before: frozenset[Quad] = frozenset()

    def before(self, index: int, operation: CompValue, ctx: QueryContext) -> None:
        self.last_index = index
        self.last_operation = str(operation.name)
        self._before = _snapshot(self.shadow)

    def after(
        self,
        index: int,
        operation: CompValue,
        ctx: QueryContext,
        *,
        succeeded: bool,
    ) -> None:
        after = _snapshot(self.shadow)
        declared: frozenset[Quad] = frozenset()
        if operation.name in ("InsertData", "DeleteData"):
            declared = _declared_ground_quads(operation, ctx, self._before)
        self.records.append(
            _OperationRecord(
                index=index,
                operation=str(operation.name),
                silent=bool(getattr(operation, "silent", False)),
                succeeded=succeeded,
                before=self._before,
                after=after,
                declared=declared,
            )
        )


def _declared_ground_quads(
    operation: CompValue, ctx: QueryContext, before: frozenset[Quad]
) -> frozenset[Quad]:
    """Quads an ``INSERT DATA``/``DELETE DATA`` statement explicitly names.

    The graph ids are resolved exactly as the update evaluator resolves
    them, so membership in *before* matches the real execution target.
    """

    if isinstance(ctx.graph, ConjunctiveGraph):
        default_id = _default_graph_id(ctx.graph)
    else:
        assert ctx.graph is not None
        default_id = ctx.graph.identifier

    declared: set[Quad] = set()
    for triple in getattr(operation, "triples", None) or ():
        s, p, o = triple
        if all(isinstance(term, Node) for term in (s, p, o)):
            declared.add((s, p, o, default_id))
    for graph_term, triples in (getattr(operation, "quads", None) or {}).items():
        if graph_term is None:
            graph_id = default_id
        else:
            graph_id = ctx.dataset.get_context(graph_term).identifier
        for triple in triples:
            s, p, o = triple
            if all(isinstance(term, Node) for term in (s, p, o)):
                declared.add((s, p, o, graph_id))
    return frozenset(declared)


# ---------------------------------------------------------------------------
# Conflict analysis
# ---------------------------------------------------------------------------


def _analyse_conflicts(
    records: Sequence[_OperationRecord],
    final: frozenset[Quad],
) -> tuple[tuple[UpdateConflict, ...], frozenset[Quad]]:
    conflicts: list[UpdateConflict] = []
    unchanged_referenced: set[Quad] = set()

    for record in records:
        if record.operation == "DeleteData":
            missing = record.declared - record.before
            if missing:
                conflicts.append(
                    UpdateConflict(
                        kind="delete_missing",
                        severity="conflict",
                        statements=(record.index,),
                        quads=frozenset(missing),
                        message=(
                            f"{_quad_word(len(missing))} named in DELETE DATA "
                            "do(es) not exist in the target graph and would "
                            "not remove anything."
                        ),
                    )
                )
        elif record.operation == "InsertData":
            present = record.declared & record.before
            if present:
                conflicts.append(
                    UpdateConflict(
                        kind="insert_existing",
                        severity="warning",
                        statements=(record.index,),
                        quads=frozenset(present),
                        message=(
                            f"{_quad_word(len(present))} named in INSERT DATA "
                            "already exist(s) and would be left unchanged."
                        ),
                    )
                )
                unchanged_referenced.update(present & final)

    # Quads added by one statement and removed by another: lexical order
    # decides the outcome.
    add_ops: dict[Quad, set[int]] = defaultdict(set)
    remove_ops: dict[Quad, set[int]] = defaultdict(set)
    for record in records:
        for quad in record.added:
            add_ops[quad].add(record.index)
        for quad in record.removed:
            remove_ops[quad].add(record.index)

    for quad in sorted(add_ops.keys() & remove_ops.keys(), key=_quad_key):
        statements = tuple(sorted(add_ops[quad] | remove_ops[quad]))
        outcome = "remains present" if quad in final else "ends up absent"
        conflicts.append(
            UpdateConflict(
                kind="statement_order",
                severity="warning",
                statements=statements,
                quads=frozenset({quad}),
                message=(
                    f"quad is added and later removed (or vice versa); "
                    f"lexical order decides, it {outcome}."
                ),
            )
        )
        if quad in final:
            unchanged_referenced.add(quad)

    conflicts.sort(key=lambda conflict: conflict.statements)
    return tuple(conflicts), frozenset(unchanged_referenced)


def _quad_word(count: int) -> str:
    return "1 quad" if count == 1 else f"{count} quads"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class UpdatePlan:
    """A previewed update request that can be applied atomically.

    Obtain instances from
    [`preview_update`][rdflib.plugins.sparql.preview.preview_update].
    """

    def __init__(
        self,
        target: Graph,
        before: frozenset[Quad],
        after: frozenset[Quad],
        preview: UpdatePreview,
        request_hash: str | None,
    ) -> None:
        self._target = target
        self._before = before
        self._after = after
        self._preview = preview
        self._request_hash = request_hash
        self._applied = False

    @property
    def preview(self) -> UpdatePreview:
        return self._preview

    @property
    def applied(self) -> bool:
        return self._applied

    def is_current(self) -> bool:
        """Whether the target still matches the state that was previewed."""

        return _snapshot(self._target) == self._before

    def apply(self, request: str | None = None) -> UpdatePreview:
        """Apply the previewed changes to the target graph.

        Args:
            request: Optional SPARQL Update text. When given, it must be
                identical to the text used for the preview; otherwise a
                [`StaleUpdatePlanError`][rdflib.plugins.sparql.preview.StaleUpdatePlanError]
                is raised. Use this to ensure the statements being committed
                are the ones that were reviewed.

        Raises:
            StaleUpdatePlanError: the request text differs from the
                previewed text, or the target graph changed since the
                preview. No data is written in either case.
            UpdateApplicationError: a write failed; the graph has been
                restored to its pre-commit state.
        """

        if self._applied:
            raise StaleUpdatePlanError("This update plan has already been applied.")
        if request is not None:
            if self._request_hash is None:
                raise StaleUpdatePlanError(
                    "The plan was built from a parsed update object; pass no "
                    "request text to apply it, or preview again."
                )
            if _text_token(request) != self._request_hash:
                raise StaleUpdatePlanError(
                    "The update statements differ from those that were "
                    "previewed; preview again and re-confirm before "
                    "committing."
                )

        current = _snapshot(self._target)
        if current != self._before:
            raise StaleUpdatePlanError(
                "The target graph changed since the preview; preview again "
                "and re-confirm before committing.",
                external_additions=current - self._before,
                external_deletions=self._before - current,
            )

        try:
            self._sync_to(self._after)
            resulting = _snapshot(self._target)
            if resulting != self._after:
                raise RuntimeError(
                    "post-commit verification failed: resulting graph does "
                    "not match the previewed result"
                )
        except Exception as original:
            self._restore(original)
            # _restore either raises UpdateApplicationError itself or returns
            raise UpdateApplicationError(
                "Applying the update failed; the graph was rolled back."
            ) from original

        self._applied = True
        return self._preview

    def _sync_to(self, desired: frozenset[Quad]) -> None:
        current = _snapshot(self._target)
        _remove_quads(self._target, current - desired)
        _add_quads(self._target, desired - current)

    def _restore(self, original: BaseException) -> None:
        try:
            current = _snapshot(self._target)
            _remove_quads(self._target, current - self._before)
            _add_quads(self._target, self._before - current)
            if _snapshot(self._target) != self._before:
                raise RuntimeError("restored graph does not match pre-commit state")
        except Exception as restore_error:
            raise UpdateApplicationError(
                "Applying the update failed and automatic rollback also "
                f"failed ({restore_error}); manual intervention is required."
            ) from original


# ---------------------------------------------------------------------------
# Bulk writes
# ---------------------------------------------------------------------------


def _group(quads: frozenset[Quad]) -> dict[Identifier, list[tuple[Node, Node, Node]]]:
    grouped: dict[Identifier, list[tuple[Node, Node, Node]]] = defaultdict(list)
    for s, p, o, graph_id in quads:
        grouped[graph_id].append((s, p, o))
    return grouped


def _add_quads(target: Graph, quads: frozenset[Quad]) -> None:
    if isinstance(target, ConjunctiveGraph):
        for graph_id, triples in _group(quads).items():
            context = target.get_context(graph_id)
            target.addN((s, p, o, context) for s, p, o in triples)
    else:
        target.addN((s, p, o, target) for s, p, o, _ in quads)


def _remove_quads(target: Graph, quads: frozenset[Quad]) -> None:
    if isinstance(target, ConjunctiveGraph):
        for graph_id, triples in _group(quads).items():
            context = target.get_context(graph_id)
            for s, p, o in triples:
                target.remove((s, p, o, context))
    else:
        for s, p, o, _ in quads:
            target.remove((s, p, o))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def preview_update(
    graph: Graph,
    strOrUpdate: str | Update,
    initNs: Mapping[str, Any] | None = None,
    initBindings: Mapping[str, Identifier] | None = None,
    base: str | None = None,
) -> UpdatePlan:
    """Preview an update request without modifying *graph*.

    Args:
        graph: The graph or dataset the update would be applied to.
        strOrUpdate: A SPARQL Update request string or a prepared
            [`Update`][rdflib.plugins.sparql.sparql.Update] (as returned by
            [`prepareUpdate`][rdflib.plugins.sparql.prepareUpdate]).
        initNs: Initial namespaces for prefix resolution; defaults to the
            namespaces bound on *graph*.
        initBindings: Initial variable bindings.
        base: Base URI used to resolve relative IRIs.

    Returns:
        An [`UpdatePlan`][rdflib.plugins.sparql.preview.UpdatePlan] holding
        the projected changes. Call
        [`UpdatePlan.apply`][rdflib.plugins.sparql.preview.UpdatePlan.apply]
        after review to commit them.

    Raises:
        UpdatePreviewError: A non-SILENT statement fails. The target graph
            is never modified.
    """

    request_text: str | None = None
    if isinstance(strOrUpdate, Update):
        update = strOrUpdate
        original_args = getattr(update, "_original_args", None)
        if original_args is not None:
            request_text = original_args[0]
    else:
        request_text = strOrUpdate
        if initNs is None:
            initNs = dict(graph.namespaces())
        update = translateUpdate(parseUpdate(strOrUpdate), base, initNs)

    shadow = _shadow_graph(graph)
    before = _snapshot(shadow)
    observer = _RecordingObserver(shadow)

    try:
        evalUpdate(shadow, update, initBindings, observer=observer)
    except Exception as original:
        raise UpdatePreviewError(
            observer.last_index, observer.last_operation, original
        ) from original

    after = _snapshot(shadow)
    conflicts, unchanged_referenced = _analyse_conflicts(observer.records, after)

    effects = tuple(
        StatementEffect(
            index=record.index,
            operation=record.operation,
            silent=record.silent,
            succeeded=record.succeeded,
            added=record.added,
            removed=record.removed,
        )
        for record in observer.records
    )

    preview = UpdatePreview(
        additions=after - before,
        deletions=before - after,
        unchanged_count=len(before & after),
        unchanged_referenced=unchanged_referenced,
        statements=effects,
        conflicts=conflicts,
        request_hash=_text_token(request_text) if request_text is not None else None,
    )

    return UpdatePlan(
        target=graph,
        before=before,
        after=after,
        preview=preview,
        request_hash=_text_token(request_text) if request_text is not None else None,
    )
