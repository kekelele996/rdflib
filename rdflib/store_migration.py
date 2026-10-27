"""Safe migration of datasets between storage backends.

Moving a dataset to another store is only safe after the destination store
has proven that it can handle the data: named graphs, batch writes,
namespace bindings and full read-back.  Stores otherwise fail at
unpredictable points -- while initialising a named graph or half way
through an import -- leaving partial data behind.

:func:`migrate` therefore works in four stages:

1. **Pre-flight** -- :func:`rdflib.store_capabilities.check_store` probes
   the destination *before* any real data is written.  Missing
   capabilities abort the migration with a :class:`MigrationError` while
   the destination is still untouched (apart from the inert, documented
   probe prefix binding).
2. **Collision check** -- by default the destination must not already
   contain a graph that the source would write.
3. **Copy** -- triples are copied per named graph with batched
   ``addN`` calls; namespace bindings are copied afterwards.
4. **Verify and commit** -- every source graph is read back from the
   destination and compared graph-by-graph, including graph identifiers
   and namespace bindings.  Any failure triggers a rollback: every graph
   written by the migration is removed again, so a failed migration never
   leaves partial data behind.  Transaction aware stores are rolled back
   via ``store.rollback()`` as well.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

from rdflib.namespace import NamespaceManager
from rdflib.store_capabilities import (
    ALL_CAPABILITIES,
    PROBE_GRAPH_NAMESPACE,
    PROBE_PREFIX,
    CapabilityReport,
    check_store,
)
from rdflib.term import IdentifiedNode, URIRef

if TYPE_CHECKING:
    from rdflib.graph import Dataset, _QuadType, _TripleType
    from rdflib.store import Store

__all__ = [
    "MigrationError",
    "MigrationResult",
    "migrate",
]

_GraphIdentifier = IdentifiedNode


class MigrationError(RuntimeError):
    """Raised when a migration cannot be performed or completed safely.

    Attributes:
        report: Capability report when pre-flight checks failed, else None.
        leftover: Identifiers of destination graphs that still hold data
            after a failed rollback.  Empty when rollback succeeded.
    """

    def __init__(
        self,
        message: str,
        *,
        report: CapabilityReport | None = None,
        leftover: Iterable[_GraphIdentifier] = (),
    ):
        super().__init__(message)
        self.report = report
        self.leftover = tuple(leftover)


@dataclass
class MigrationResult:
    """Summary of a successful migration.

    Attributes:
        graphs: Number of graphs copied, including the default graph.
        triples: Total number of triples copied.
        triples_per_graph: Number of triples written per graph identifier.
        namespaces: Number of namespace bindings copied.
        report: The pre-flight capability report of the destination store.
    """

    graphs: int
    triples: int
    triples_per_graph: dict[_GraphIdentifier, int] = field(default_factory=dict)
    namespaces: int = 0
    report: CapabilityReport | None = None


def _as_store(store_or_dataset: Union[Store, Dataset, Any]) -> Store:
    from rdflib.store import Store as _Store

    if isinstance(store_or_dataset, _Store):
        return store_or_dataset
    graph_store = getattr(store_or_dataset, "store", None)
    if isinstance(graph_store, _Store):
        return graph_store
    raise TypeError(
        f"expected a Store or Dataset instance, got {type(store_or_dataset).__name__!r}"
    )


def _coerce_dataset(
    store_or_dataset: Union[Store, Dataset, Any], *, role: str
) -> Dataset:
    """Return a Dataset for a Dataset or a context/graph aware Store."""
    from rdflib.graph import Dataset

    if isinstance(store_or_dataset, Dataset):
        return store_or_dataset
    store = _as_store(store_or_dataset)
    try:
        dataset = Dataset(store=store)
        # Resolving graphs must not implicitly inject rdflib's default
        # namespace bindings: migration copies the source bindings explicitly.
        dataset.namespace_manager = NamespaceManager(dataset, "none")
        return dataset
    except Exception as exc:
        raise MigrationError(
            f"the {role} store ({type(store).__name__}) cannot back a Dataset: {exc}"
        ) from exc


def _snapshot(dataset: Dataset) -> dict[_GraphIdentifier, frozenset[_TripleType]]:
    """Read every graph of a dataset into memory keyed by graph identifier."""
    snapshot: dict[_GraphIdentifier, frozenset[_TripleType]] = {}
    for graph in dataset.graphs():
        snapshot[graph.identifier] = frozenset(graph)
    return snapshot


def _namespaces(store: Store) -> dict[str, URIRef]:
    return {
        prefix: namespace
        for prefix, namespace in store.namespaces()
        if not prefix.startswith(PROBE_PREFIX)
    }


def _non_empty_graphs(
    dataset: Dataset,
    graph_ids: Iterable[_GraphIdentifier],
    errors: list[str],
) -> list[_GraphIdentifier]:
    non_empty: list[_GraphIdentifier] = []
    for graph_id in graph_ids:
        try:
            if len(dataset.graph(graph_id)):
                non_empty.append(graph_id)
        except Exception as exc:
            errors.append(f"could not inspect graph {graph_id}: {exc}")
            non_empty.append(graph_id)
    return non_empty


def _rollback(
    dataset: Dataset,
    migrated_graph_ids: Iterable[_GraphIdentifier],
) -> tuple[list[_GraphIdentifier], list[str]]:
    """Remove everything the migration wrote.

    Returns:
        ``(leftover_graphs, errors)`` where ``leftover_graphs`` lists graphs
        that still contain triples and ``errors`` lists cleanup problems.
    """
    errors: list[str] = []
    migrated_ids = list(migrated_graph_ids)

    transaction_rolled_back = False
    if dataset.store.transaction_aware:
        try:
            dataset.store.rollback()
            transaction_rolled_back = True
        except Exception as exc:
            errors.append(f"store.rollback() failed: {exc}")

    # After a successful transaction rollback nothing should be left; only
    # touch graphs that still hold triples.  Without a transaction every
    # migrated graph must be removed explicitly.
    if transaction_rolled_back:
        to_remove = _non_empty_graphs(dataset, migrated_ids, errors)
    else:
        to_remove = migrated_ids
    for graph_id in to_remove:
        try:
            dataset.remove_graph(dataset.graph(graph_id))
        except Exception as exc:
            errors.append(f"could not remove graph {graph_id}: {exc}")

    leftover = _non_empty_graphs(dataset, migrated_ids, errors)
    return leftover, errors


def migrate(
    source: Union[Store, Dataset, Any],
    target: Union[Store, Dataset, Any],
    *,
    batch_size: int = 1000,
    required_capabilities: Iterable[str] = ALL_CAPABILITIES,
    verify: bool = True,
    allow_existing: bool = False,
) -> MigrationResult:
    """Safely copy a dataset from one store backend to another.

    Args:
        source: The Dataset to copy, or a Store/context-aware graph backing
            one.
        target: The destination Dataset or Store.  The destination store is
            probed before any data is written.
        batch_size: Number of quads buffered per ``addN`` call.
        required_capabilities: Capabilities the destination must provide;
            migration aborts up-front if any of them are missing.
        verify: Read back and compare every graph after copying.
        allow_existing: Allow the destination to already contain graphs that
            the source also writes.  When False (default) any such collision
            aborts the migration before writing.

    Returns:
        A :class:`MigrationResult` with copy statistics.

    Raises:
        MigrationError: If the pre-flight fails, graphs collide, the copy
            fails, verification fails or the rollback cannot clean up.  In
            all failure cases the destination contains no migrated triples
            unless the error reports ``leftover`` graphs requiring manual
            cleanup.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    target_store = _as_store(target)
    source_store = _as_store(source)
    if target_store is source_store:
        raise MigrationError("source and target must use different stores")

    # -- Stage 1: pre-flight, before any real write -------------------------
    report = check_store(target_store, capabilities=required_capabilities)
    required = frozenset(required_capabilities)
    missing = {
        name: status for name, status in report.unsupported.items() if name in required
    }
    if missing:
        details = "; ".join(
            f"{name}: {status.reason or 'not supported'}"
            for name, status in sorted(missing.items())
        )
        raise MigrationError(
            f"destination store {type(target_store).__name__} failed "
            f"pre-flight capability checks ({details}); no data was written",
            report=report,
        )
    if report.leftover_graphs:
        raise MigrationError(
            "destination store could not remove pre-flight probe data from "
            f"graphs {[str(g) for g in report.leftover_graphs]}; refusing to "
            "migrate because rollback could not be guaranteed",
            report=report,
        )

    source_dataset = _coerce_dataset(source, role="source")
    target_dataset = _coerce_dataset(target, role="target")

    # Materialise the source once: read access must not see a changing store
    # while copying, and this is what verification compares against.
    source_snapshot = _snapshot(source_dataset)
    source_namespaces = _namespaces(source_store)

    # -- Stage 2: collision check, still before any write -------------------
    if not allow_existing:
        existing = {
            graph.identifier for graph in target_dataset.graphs() if len(graph) > 0
        }
        colliding = sorted(
            str(graph_id)
            for graph_id in source_snapshot
            if graph_id in existing and source_snapshot[graph_id]
        )
        if colliding:
            raise MigrationError(
                "destination already contains graphs that would be "
                f"migrated: {colliding}; pass allow_existing=True to override"
            )

    # -- Stage 3: copy with rollback tracking -------------------------------
    migrated_graph_ids: list[_GraphIdentifier] = []
    triples_per_graph: dict[_GraphIdentifier, int] = {}
    namespaces_copied = 0

    try:
        for graph_id, triples in source_snapshot.items():
            target_graph = target_dataset.graph(graph_id)
            migrated_graph_ids.append(graph_id)
            batch: list[_QuadType] = []
            for subject, predicate, obj in triples:
                batch.append((subject, predicate, obj, target_graph))
                if len(batch) >= batch_size:
                    target_dataset.addN(batch)
                    batch = []
            if batch:
                target_dataset.addN(batch)
            triples_per_graph[graph_id] = len(triples)

        # -- Stage 4a: verify graphs before touching namespaces -------------
        if verify:
            _verify_graphs(
                source_snapshot,
                target_dataset,
            )

        # Namespace bindings are copied last: a failure here still allows a
        # complete data rollback.  The store interface has no unbind, so
        # bindings already copied before a failure are listed in the error.
        for prefix, namespace in source_namespaces.items():
            target_store.bind(prefix, namespace, override=True)
            namespaces_copied += 1

        if verify:
            _verify_namespaces(source_namespaces, target_store)

        if target_store.transaction_aware:
            target_store.commit()
    except MigrationError as original:
        leftover, errors = _rollback(target_dataset, migrated_graph_ids)
        message = f"migration failed ({original}) and was rolled back"
        if errors:
            message += f"; cleanup problems: {'; '.join(errors)}"
        if leftover:
            message += (
                "; graphs with leftover data requiring manual cleanup: "
                f"{[str(g) for g in leftover]}"
            )
        raise MigrationError(message, report=report, leftover=leftover) from original
    except Exception as exc:
        leftover, errors = _rollback(target_dataset, migrated_graph_ids)
        message = f"migration failed ({type(exc).__name__}: {exc}) and was rolled back"
        if errors:
            message += f"; cleanup problems: {'; '.join(errors)}"
        if leftover:
            message += (
                "; graphs with leftover data requiring manual cleanup: "
                f"{[str(g) for g in leftover]}"
            )
        raise MigrationError(message, report=report, leftover=leftover) from exc

    return MigrationResult(
        graphs=len(triples_per_graph),
        triples=sum(triples_per_graph.values()),
        triples_per_graph=triples_per_graph,
        namespaces=namespaces_copied,
        report=report,
    )


def _verify_graphs(
    source_snapshot: dict[_GraphIdentifier, frozenset[_TripleType]],
    target_dataset: Dataset,
) -> None:
    """Compare source and destination graph-by-graph."""
    mismatches: list[str] = []
    for graph_id, expected in source_snapshot.items():
        actual = frozenset(target_dataset.graph(graph_id))
        if actual != expected:
            missing = expected.difference(actual)
            unexpected = actual.difference(expected)
            detail = [f"graph {graph_id} read-back mismatch"]
            if missing:
                detail.append(f"missing {len(missing)} triple(s)")
            if unexpected:
                detail.append(f"unexpected {len(unexpected)} triple(s)")
            mismatches.append(": ".join(detail))

    target_graphs = list(target_dataset.graphs())
    target_all_ids = {graph.identifier for graph in target_graphs}
    missing_graphs = [
        graph_id for graph_id in source_snapshot if graph_id not in target_all_ids
    ]
    if missing_graphs:
        mismatches.append(
            "graphs missing from destination enumeration: "
            + ", ".join(str(graph_id) for graph_id in missing_graphs)
        )

    source_ids = {graph_id for graph_id, triples in source_snapshot.items() if triples}
    target_ids = {
        graph.identifier
        for graph in target_graphs
        if len(graph) > 0
        and not str(graph.identifier).startswith(PROBE_GRAPH_NAMESPACE)
    }
    extra = target_ids.difference(source_ids)
    if extra:
        mismatches.append(
            "destination contains non-empty graphs absent from source: "
            + ", ".join(str(graph_id) for graph_id in sorted(extra, key=str))
        )
    if mismatches:
        raise MigrationError("; ".join(mismatches[:10]))


def _verify_namespaces(
    source_namespaces: dict[str, URIRef], target_store: Store
) -> None:
    actual = _namespaces(target_store)
    wrong = {
        prefix: (namespace, actual.get(prefix))
        for prefix, namespace in source_namespaces.items()
        if actual.get(prefix) != namespace
    }
    if wrong:
        raise MigrationError(
            "namespace bindings did not survive migration: "
            + ", ".join(
                f"{prefix}: expected {expected!s}, got {got!s}"
                for prefix, (expected, got) in sorted(wrong.items())
            )
        )
