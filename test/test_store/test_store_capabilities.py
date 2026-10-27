"""Tests for pre-flight store capability checks."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from rdflib import Dataset, Graph
from rdflib.graph import _QuadType
from rdflib.namespace import RDFS
from rdflib.plugins.stores.memory import Memory, SimpleMemory
from rdflib.store import Store
from rdflib.store_capabilities import (
    ALL_CAPABILITIES,
    BATCH_WRITE,
    FULL_READ,
    NAMED_GRAPHS,
    NAMESPACES,
    PROBE_GRAPH_NAMESPACE,
    PROBE_PREFIX,
    CapabilityCheckError,
    check_store,
)
from rdflib.term import BNode, Literal, URIRef


class NoBatchStore(Memory):
    """addN is not implemented (default loop-to-add is replaced on purpose)."""

    def addN(self, quads: Iterable[_QuadType]) -> None:  # noqa: N802
        raise NotImplementedError("batch writes are not supported")


class NoNamespaceStore(Memory):
    def bind(self, prefix: str, namespace: URIRef, override: bool = True) -> None:
        raise NotImplementedError("namespace persistence is not supported")


class BrokenLengthStore(Memory):
    def __len__(self, context: object | None = None) -> int:
        if context is not None:
            raise RuntimeError("per-graph count failure")
        return super().__len__(None)


class ReadOnlyStore(Memory):
    """Mimics SPARQLStore: every write/delete operation is rejected."""

    def add(self, triple: object, context: object = None, quoted: bool = False) -> None:
        raise TypeError("the store is read only")

    def remove(self, triple: object, context: object | None = None) -> None:
        raise TypeError("the store is read only")

    def remove_graph(self, graph: object) -> None:
        raise TypeError("the store is read only")


class UnremovableStore(Memory):
    """Accepts writes but can neither drop graphs nor delete triples."""

    def remove_graph(self, graph: object) -> None:
        pass

    def remove(self, triple: object, context: object | None = None) -> None:
        raise OSError("delete operations are not supported")


def _probe_triples_remaining(store: Store) -> int:
    dataset = Dataset(store=store)
    # Dataset.quads yields graph identifier nodes (or None for default graph).
    return sum(
        1
        for _s, _p, _o, context in dataset.quads((None, None, None, None))
        if context is not None and str(context).startswith(PROBE_GRAPH_NAMESPACE)
    )


def test_memory_store_passes_every_capability() -> None:
    store = Memory()
    report = check_store(store)

    assert report
    assert report.supports(ALL_CAPABILITIES)
    assert report.leftover_graphs == []
    assert report.residual_namespaces == [PROBE_PREFIX]
    assert set(report.statuses) == set(ALL_CAPABILITIES)


def test_probe_leaves_no_triples_or_named_graphs_behind() -> None:
    store = Memory()
    check_store(store)

    dataset = Dataset(store=store)
    enumerated = {str(graph.identifier) for graph in dataset.graphs()}
    assert not any(
        graph_id.startswith(PROBE_GRAPH_NAMESPACE) for graph_id in enumerated
    )
    assert _probe_triples_remaining(store) == 0

    # The fixed probe namespace binding was (re)bound at most once and the
    # probe did not inject rdflib's default namespace bindings.
    assert set(dict(store.namespaces())) == {PROBE_PREFIX}


def test_check_accepts_graph_and_dataset_wrappers() -> None:
    dataset = Dataset(store=Memory())
    assert check_store(dataset).supports(ALL_CAPABILITIES)
    graph = Graph(store=Memory())
    assert check_store(graph).supports(NAMESPACES)


def test_unknown_capability_rejected() -> None:
    with pytest.raises(ValueError, match="unknown capabilities"):
        check_store(Memory(), capabilities=["bogus"])


def test_non_context_store_fails_named_graph_capabilities() -> None:
    report = check_store(SimpleMemory())

    assert not report.supports(NAMED_GRAPHS)
    assert not report.supports(BATCH_WRITE)
    assert not report.supports(FULL_READ)
    assert report.supports(NAMESPACES)
    reason = report.unsupported[NAMED_GRAPHS].reason or ""
    assert "context_aware" in reason
    assert not report


def test_require_raises_before_writes() -> None:
    report = check_store(SimpleMemory())
    with pytest.raises(CapabilityCheckError) as exc_info:
        report.require(ALL_CAPABILITIES)
    assert "named_graphs" in str(exc_info.value)


def test_addn_failure_isolated_to_batch_capability() -> None:
    store = NoBatchStore()
    report = check_store(store)

    assert not report.supports(BATCH_WRITE)
    assert report.supports(NAMED_GRAPHS)
    assert report.supports(FULL_READ)
    assert report.supports(NAMESPACES)
    status = report.unsupported[BATCH_WRITE]
    assert status.error_type is not None
    assert "batch writes are not supported" in (status.reason or "")
    assert _probe_triples_remaining(store) == 0


def test_namespace_failure_isolated_to_namespace_capability() -> None:
    store = NoNamespaceStore()
    report = check_store(store)

    assert not report.supports(NAMESPACES)
    assert report.supports(NAMED_GRAPHS)
    assert report.supports(BATCH_WRITE)
    assert report.supports(FULL_READ)
    assert "namespace" in (report.unsupported[NAMESPACES].reason or "")
    assert _probe_triples_remaining(store) == 0


def test_broken_length_fails_full_read_only() -> None:
    store = BrokenLengthStore()
    report = check_store(store)

    assert report.supports(NAMED_GRAPHS)
    assert report.supports(BATCH_WRITE)
    assert not report.supports(FULL_READ)
    assert _probe_triples_remaining(store) == 0


def test_read_only_store_fails_write_capabilities() -> None:
    report = check_store(ReadOnlyStore())

    assert not report.supports(NAMED_GRAPHS)
    assert not report.supports(BATCH_WRITE)
    assert not report.supports(FULL_READ)
    assert report.supports(NAMESPACES)
    # Error text from the store must be surfaced.
    assert "read only" in (report.unsupported[NAMED_GRAPHS].reason or "")
    # Nothing was written, so nothing can be left over.
    assert report.leftover_graphs == []


def test_unremovable_probe_data_blocks_full_read() -> None:
    store = UnremovableStore()
    report = check_store(store)

    # Named graph writes and reads work, but cleanup does not.
    assert report.statuses[NAMED_GRAPHS].supported
    assert report.statuses[BATCH_WRITE].supported
    assert not report.supports(FULL_READ)
    assert report.leftover_graphs
    assert not report
    assert any(
        str(graph_id).startswith(PROBE_GRAPH_NAMESPACE)
        for graph_id in report.leftover_graphs
    )


def test_bnode_named_graph_can_be_skipped() -> None:
    store = Memory()
    report = check_store(store, probe_bnode_graph=False)
    assert report.supports(ALL_CAPABILITIES)
    assert _probe_triples_remaining(store) == 0


def test_report_describe_is_readable() -> None:
    report = check_store(SimpleMemory())
    text = report.describe()
    assert "[ ]" in text and "named_graphs" in text
    assert "[x] namespaces" in text


def test_bnode_named_graph_roundtrip_on_memory() -> None:
    store = Memory()
    report = check_store(store, capabilities=[NAMED_GRAPHS])
    assert report.supports(NAMED_GRAPHS)
    # Smoke test: a blank-node graph used directly also works.
    dataset = Dataset(store=store)
    graph = dataset.graph(BNode())
    triple = (URIRef("urn:example:s"), RDFS.label, Literal("x"))
    graph.add(triple)
    assert triple in graph
