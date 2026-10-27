"""Pre-flight capability checks for storage backends.

Different :class:`~rdflib.store.Store` implementations accept ordinary
triples but fail at different points when a dataset uses named graphs,
batch writes, namespace bindings or full enumeration: some fail while the
store is initialised, others only fail half way through an import.

The :func:`check_store` helper performs a small, self-cleaning probe
*before* any real data is written to the store:

* :data:`NAMED_GRAPHS` -- triples can be written to and read back from
  named graphs (IRI and blank-node identified), graphs show up in
  enumeration and triples do not leak across graphs;
* :data:`BATCH_WRITE` -- ``store.addN`` accepts a batch of quads spanning
  several named graphs;
* :data:`NAMESPACES` -- prefix bindings survive ``bind`` and are visible
  through ``namespaces`` / ``namespace``;
* :data:`FULL_READ` -- every triple of a named graph can be enumerated,
  per-graph counts are correct and the quads interface returns probe
  quads with the right context.

Probe triples live in uniquely named graphs and are removed again when
the check finishes.  The store interface has no generic "unbind"
operation, so the namespace probe leaves a single inert prefix binding
(:data:`PROBE_PREFIX`) pointing at a unique URN; rebinding it on every
check means at most one such binding ever accumulates.  The binding is
recorded in :attr:`CapabilityReport.residual_namespaces` and is ignored
by :mod:`rdflib.store_migration`.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

from rdflib.namespace import RDF, RDFS, XSD, NamespaceManager
from rdflib.term import BNode, Literal, URIRef

if TYPE_CHECKING:
    from rdflib.graph import Dataset
    from rdflib.store import Store

#: Named graphs can be written, enumerated and read back.
NAMED_GRAPHS = "named_graphs"
#: ``addN`` batches of quads spanning several named graphs work.
BATCH_WRITE = "batch_write"
#: Prefix bindings persist and are enumerable.
NAMESPACES = "namespaces"
#: Full per-graph enumeration, counts and quad iteration work.
FULL_READ = "full_read"

#: Every capability probed by :func:`check_store`.
ALL_CAPABILITIES: frozenset[str] = frozenset(
    {NAMED_GRAPHS, BATCH_WRITE, NAMESPACES, FULL_READ}
)

#: Prefix used by the namespace probe.  It is reused (rebound) on every
#: check so persistent stores accumulate at most one probe binding.
PROBE_PREFIX = "rdflibprobe"

#: URN namespace used for all probe graphs and probe terms.  Everything the
#: check creates lives under this URN so probes can never collide with real
#: data and leftovers are easy to recognise.
PROBE_GRAPH_NAMESPACE = "urn:rdflib:capability-probe:"

__all__ = [
    "ALL_CAPABILITIES",
    "BATCH_WRITE",
    "CapabilityCheckError",
    "CapabilityReport",
    "CapabilityStatus",
    "FULL_READ",
    "NAMED_GRAPHS",
    "NAMESPACES",
    "PROBE_GRAPH_NAMESPACE",
    "PROBE_PREFIX",
    "check_store",
]


@dataclass(frozen=True)
class CapabilityStatus:
    """Outcome of probing a single capability.

    Attributes:
        capability: One of the module level capability constants.
        supported: Whether the capability was exercised successfully.
        reason: Human readable explanation when the capability is not
            supported, including static-flag failures and exception text.
        error_type: Qualified class name of the exception that was raised,
            if any.
    """

    capability: str
    supported: bool
    reason: str | None = None
    error_type: str | None = None

    def raise_if_unsupported(self) -> None:
        """Raise :class:`CapabilityCheckError` if this capability is missing."""
        if not self.supported:
            raise CapabilityCheckError({self.capability: self})


@dataclass
class CapabilityReport:
    """Result of :func:`check_store`.

    Attributes:
        store_name: Name of the store class that was checked.
        statuses: Capability constant to :class:`CapabilityStatus` mapping.
        leftover_graphs: Identifiers of probe graphs that could not be
            removed completely.  Migration must not proceed while this is
            non-empty because a migration could not be rolled back safely.
        residual_namespaces: Prefixes left behind by the namespace probe
            (the generic store interface offers no unbind operation).
    """

    store_name: str
    statuses: dict[str, CapabilityStatus] = field(default_factory=dict)
    leftover_graphs: list[URIRef | BNode] = field(default_factory=list)
    residual_namespaces: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """True when every probed capability is supported and cleanup succeeded."""
        return all(status.supported for status in self.statuses.values()) and not (
            self.leftover_graphs
        )

    @property
    def supported(self) -> frozenset[str]:
        """The set of supported capability constants."""
        return frozenset(
            name
            for name, status in self.statuses.items()
            if status.supported and name not in self._capabilities_with_leftovers()
        )

    @property
    def unsupported(self) -> dict[str, CapabilityStatus]:
        """Capability constants that are missing mapped to their status."""
        missing = {
            name: status
            for name, status in self.statuses.items()
            if not status.supported
        }
        if self.leftover_graphs and FULL_READ not in missing:
            missing[FULL_READ] = CapabilityStatus(
                capability=FULL_READ,
                supported=False,
                reason=(
                    "probe data could not be removed from graphs: "
                    + ", ".join(str(graph_id) for graph_id in self.leftover_graphs)
                ),
            )
        return missing

    def supports(self, capabilities: Iterable[str] | str) -> bool:
        """Return whether all of the given capabilities are supported."""
        wanted = (
            frozenset({capabilities})
            if isinstance(capabilities, str)
            else frozenset(capabilities)
        )
        return wanted.issubset(self.supported)

    def require(self, capabilities: Iterable[str] | str) -> None:
        """Raise :class:`CapabilityCheckError` unless all capabilities work.

        The error is raised before any real data needs to be written.
        """
        wanted = (
            frozenset({capabilities})
            if isinstance(capabilities, str)
            else frozenset(capabilities)
        )
        missing = {
            name: self.unsupported[name] for name in wanted if name in self.unsupported
        }
        if missing:
            raise CapabilityCheckError(missing)

    def _capabilities_with_leftovers(self) -> frozenset[str]:
        return frozenset({FULL_READ}) if self.leftover_graphs else frozenset()

    def describe(self) -> str:
        """Multi-line human readable summary of the report."""
        lines = [f"Store {self.store_name}:"]
        for name in (NAMED_GRAPHS, BATCH_WRITE, NAMESPACES, FULL_READ):
            status = self.statuses.get(name)
            if status is None:
                continue
            if status.supported:
                lines.append(f"  [x] {name}")
            else:
                lines.append(f"  [ ] {name}: {status.reason}")
        for graph_id in self.leftover_graphs:
            lines.append(f"  ! leftover probe graph: {graph_id}")
        for prefix in self.residual_namespaces:
            lines.append(f"  ~ residual probe prefix binding: {prefix}")
        return "\n".join(lines)


class CapabilityCheckError(Exception):
    """Raised when a store does not provide one or more capabilities.

    Attributes:
        missing: Capability constant to the status explaining the failure.
    """

    def __init__(self, missing: dict[str, CapabilityStatus]):
        self.missing = dict(missing)
        detail = "; ".join(
            f"{name}: {status.reason or 'not supported'}"
            for name, status in sorted(self.missing.items())
        )
        super().__init__(f"store is missing required capabilities: {detail}")


def _store_name(store: Store) -> str:
    return type(store).__name__


def _as_store(store_or_graph: Union[Store, Any]) -> Store:
    """Extract a Store from a Store or graph-like object."""
    from rdflib.store import Store as _Store

    if isinstance(store_or_graph, _Store):
        return store_or_graph
    graph_store = getattr(store_or_graph, "store", None)
    if isinstance(graph_store, _Store):
        return graph_store
    raise TypeError(
        "expected a Store instance or a graph/dataset backed by one, got "
        f"{type(store_or_graph).__name__!r}"
    )


def _fail(statuses: dict[str, CapabilityStatus], name: str, exc: Exception) -> None:
    statuses[name] = CapabilityStatus(
        capability=name,
        supported=False,
        reason=str(exc) or f"{type(exc).__name__} raised without a message",
        error_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
    )


def _ok(statuses: dict[str, CapabilityStatus], name: str) -> None:
    statuses[name] = CapabilityStatus(capability=name, supported=True)


def check_store(
    store_or_graph: Union[Store, Any],
    *,
    capabilities: Iterable[str] = ALL_CAPABILITIES,
    probe_bnode_graph: bool = True,
) -> CapabilityReport:
    """Probe a store for the capabilities needed by dataset migration.

    The check writes a handful of uniquely identified triples and removes
    them again.  It never raises for a missing capability -- failures are
    reported on the returned :class:`CapabilityReport` -- but it may raise
    if called with something that is not a store at all.

    Args:
        store_or_graph: The :class:`~rdflib.store.Store` to check, or a
            graph/dataset backed by one.
        capabilities: Restrict the check to a subset of
            :data:`ALL_CAPABILITIES`.
        probe_bnode_graph: Also check a blank-node identified named graph.
            Some remote stores reject blank-node graph names even though
            IRI named graphs work.

    Returns:
        A :class:`CapabilityReport` with one status per capability.
    """
    store = _as_store(store_or_graph)
    wanted = frozenset(capabilities)
    unknown = wanted.difference(ALL_CAPABILITIES)
    if unknown:
        raise ValueError(f"unknown capabilities: {sorted(unknown)!r}")

    statuses: dict[str, CapabilityStatus] = {}
    report = CapabilityReport(store_name=_store_name(store), statuses=statuses)

    if NAMESPACES in wanted:
        _probe_namespaces(store, statuses, report)

    context_capabilities = wanted.intersection({NAMED_GRAPHS, BATCH_WRITE, FULL_READ})
    if not context_capabilities:
        return report

    # Static checks first: these stores fail deterministically, there is no
    # point exercising them (and Dataset construction asserts on these flags).
    if not getattr(store, "context_aware", False):
        reason = (
            f"{_store_name(store)} is not context_aware: it cannot keep "
            "triples in separate named graphs"
        )
        for name in context_capabilities:
            statuses[name] = CapabilityStatus(
                capability=name, supported=False, reason=reason
            )
        return report
    if not getattr(store, "graph_aware", False):
        reason = (
            f"{_store_name(store)} is not graph_aware: named graphs cannot "
            "be tracked reliably (Dataset requires a graph-aware store)"
        )
        for name in context_capabilities:
            statuses[name] = CapabilityStatus(
                capability=name, supported=False, reason=reason
            )
        return report

    from rdflib.graph import Dataset

    token = uuid.uuid4().hex
    base = URIRef(f"{PROBE_GRAPH_NAMESPACE}{token}:")
    probe_graph_ids: list[URIRef | BNode] = []

    try:
        dataset: Dataset = Dataset(store=store)
        # Do not let resolving named graphs implicitly bind rdflib's default
        # namespaces: pre-flight must only write its own probe data.
        dataset.namespace_manager = NamespaceManager(dataset, "none")

        if NAMED_GRAPHS in wanted:
            graph_id = URIRef(base + "named")
            probe_graph_ids.append(graph_id)
            other_id = URIRef(base + "named-other")
            probe_graph_ids.append(other_id)
            extra_ids: list[BNode] = []
            if probe_bnode_graph:
                bnode_id = BNode()
                probe_graph_ids.append(bnode_id)
                extra_ids.append(bnode_id)
            _probe(statuses, NAMED_GRAPHS)(
                _probe_named_graphs,
                dataset,
                graph_id,
                other_id,
                extra_ids,
                base,
            )

        if BATCH_WRITE in wanted:
            first_id = URIRef(base + "batch-1")
            second_id = URIRef(base + "batch-2")
            probe_graph_ids.extend([first_id, second_id])
            _probe(statuses, BATCH_WRITE)(
                _probe_batch_write, dataset, first_id, second_id, base
            )

        if FULL_READ in wanted:
            full_id = URIRef(base + "full")
            probe_graph_ids.append(full_id)
            _probe(statuses, FULL_READ)(_probe_full_read, dataset, full_id, base)
    except Exception as exc:  # pragma: no cover - defensive guard
        # Dataset construction itself failed despite the static flags above.
        for name in context_capabilities:
            statuses.setdefault(
                name,
                CapabilityStatus(
                    capability=name,
                    supported=False,
                    reason=str(exc) or type(exc).__name__,
                    error_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                ),
            )
    finally:
        _cleanup_probe_graphs(store, probe_graph_ids, report)

    return report


def _probe(statuses: dict[str, CapabilityStatus], name: str):
    """Return a runner that records success/failure of one probe function."""

    def run(function, *args: Any) -> None:
        try:
            function(*args)
        except Exception as exc:
            _fail(statuses, name, exc)
        else:
            _ok(statuses, name)

    return run


def _probe_named_graphs(
    dataset: Dataset,
    graph_id: URIRef,
    other_id: URIRef,
    extra_ids: list[BNode],
    base: URIRef,
) -> None:
    graph = dataset.graph(graph_id)
    other = dataset.graph(other_id)
    triple = (URIRef(base + "s1"), RDF.type, URIRef(base + "Thing"))
    graph.add(triple)

    if triple not in graph:
        raise AssertionError(
            f"triple written to {graph_id} could not be read back from it"
        )
    if triple in other:
        raise AssertionError(f"triple written to {graph_id} leaked into {other_id}")

    known_ids = {context.identifier for context in dataset.graphs()}
    if graph_id not in known_ids:
        raise AssertionError(f"named graph {graph_id} is not enumerated")

    # A second triple in the same graph exercises repeated context use.
    second = (
        URIRef(base + "s1"),
        RDFS.label,
        Literal(f"named graph probe {graph_id}"),
    )
    graph.add(second)
    if set(graph) != {triple, second}:
        raise AssertionError("named graph did not retain a second triple")

    for bnode_id in extra_ids:
        bnode_graph = dataset.graph(bnode_id)
        bnode_triple = (URIRef(base + "bn"), RDF.type, URIRef(base + "BNodeGraph"))
        bnode_graph.add(bnode_triple)
        if bnode_triple not in bnode_graph:
            raise AssertionError(
                "triple written to a blank-node named graph could not be "
                "read back from it"
            )
        bnode_ids = {context.identifier for context in dataset.graphs()}
        if bnode_id not in bnode_ids:
            raise AssertionError("blank-node named graph is not enumerated")


def _probe_batch_write(
    dataset: Dataset,
    first_id: URIRef,
    second_id: URIRef,
    base: URIRef,
) -> None:
    first = dataset.graph(first_id)
    second = dataset.graph(second_id)
    quads = [
        (
            URIRef(base + "b1"),
            RDFS.label,
            Literal("batch-1"),
            first,
        ),
        (
            URIRef(base + "b2"),
            RDF.value,
            Literal(2, datatype=XSD.integer),
            second,
        ),
        (
            URIRef(base + "b3"),
            RDFS.comment,
            Literal("batch-2 second triple"),
            second,
        ),
    ]
    # Exercise the store directly so the probe does not depend on graph
    # level addN filtering.
    dataset.store.addN(iter(quads))

    first_triples = set(first)
    second_triples = set(second)
    if first_triples != {quads[0][:3]}:
        raise AssertionError(
            f"batch addN did not land in {first_id}: got {first_triples!r}"
        )
    if second_triples != {quads[1][:3], quads[2][:3]}:
        raise AssertionError(
            f"batch addN did not land both quads in {second_id}: got {second_triples!r}"
        )
    if quads[1][:3] in first or quads[0][:3] in second:
        raise AssertionError("batch addN mixed triples across named graphs")


def _probe_full_read(
    dataset: Dataset,
    graph_id: URIRef,
    base: URIRef,
) -> None:
    graph = dataset.graph(graph_id)
    triples = {
        (URIRef(base + "f1"), RDF.type, URIRef(base + "Full")),
        (URIRef(base + "f2"), RDFS.label, Literal("full read", lang="en")),
        (
            URIRef(base + "f3"),
            RDFS.comment,
            Literal("42", datatype=XSD.integer),
        ),
    }
    # Deliberately use single adds: full-read support must not depend on
    # addN support.
    for triple in triples:
        graph.add(triple)

    if len(graph) != len(triples):
        raise AssertionError(
            f"per-graph length reports {len(graph)}, expected {len(triples)}"
        )
    read_back = set(graph)
    if read_back != triples:
        raise AssertionError(
            f"full enumeration mismatch: missing {triples - read_back!r}, "
            f"unexpected {read_back - triples!r}"
        )

    expected_quads = {(s, p, o, graph_id) for s, p, o in triples}
    seen_quads = {
        (s, p, o, context)
        for s, p, o, context in dataset.quads((None, None, None, None))
        if context == graph_id
    }
    if seen_quads != expected_quads:
        raise AssertionError(
            "quad enumeration did not return every probe quad with the correct context"
        )

    context_ids = {context.identifier for context in dataset.graphs()}
    if graph_id not in context_ids:
        raise AssertionError(f"probed graph {graph_id} missing from enumeration")

    sample_triple = next(iter(triples))
    triple_contexts = set(dataset.store.contexts(sample_triple))
    if graph_id not in {context.identifier for context in triple_contexts}:
        raise AssertionError(
            "store.contexts() does not report the graph holding a probe triple"
        )


def _probe_namespaces(
    store: Store,
    statuses: dict[str, CapabilityStatus],
    report: CapabilityReport,
) -> None:
    token = uuid.uuid4().hex
    namespace = URIRef(f"{PROBE_GRAPH_NAMESPACE}{token}:ns/")
    try:
        store.bind(PROBE_PREFIX, namespace, override=True)
        bindings = dict(store.namespaces())
        if bindings.get(PROBE_PREFIX) != namespace:
            raise AssertionError(
                f"bound prefix {PROBE_PREFIX!r} is not returned by namespaces()"
            )
        if store.namespace(PROBE_PREFIX) != namespace:
            raise AssertionError(
                f"namespace({PROBE_PREFIX!r}) does not resolve to the bound URI"
            )
    except Exception as exc:
        _fail(statuses, NAMESPACES, exc)
    else:
        _ok(statuses, NAMESPACES)
    # The generic Store interface has no unbind: the probe prefix remains.
    # Reusing one fixed prefix keeps this to a single inert binding.
    report.residual_namespaces.append(PROBE_PREFIX)


def _cleanup_probe_graphs(
    store: Store,
    probe_graph_ids: list[URIRef | BNode],
    report: CapabilityReport,
) -> None:
    """Remove every probe graph and record graphs that still hold triples."""
    if not probe_graph_ids:
        return
    from rdflib.graph import Dataset

    try:
        dataset = Dataset(store=store)
        # Same as in check_store: no implicit default namespace bindings.
        dataset.namespace_manager = NamespaceManager(dataset, "none")
    except Exception:
        # A store that could not host a Dataset cannot hold probe graphs
        # created through one either.
        return

    for graph_id in probe_graph_ids:
        try:
            graph = dataset.graph(graph_id)
        except Exception:
            # Even resolving the probe graph failed; its state cannot be
            # inspected, assume its data may be left behind.
            report.leftover_graphs.append(graph_id)
            continue
        removal_error: Exception | None = None
        try:
            dataset.remove_graph(graph)
        except Exception as exc:
            removal_error = exc
            # Fall back to removing triples from the probe context only.
            try:
                store.remove((None, None, None), graph)
            except Exception:
                pass
        try:
            leftover = bool(set(graph))
        except Exception:
            # If we cannot even read it back and removal failed already,
            # assume probe data may be left behind.
            leftover = removal_error is not None
        if leftover:
            report.leftover_graphs.append(graph_id)
