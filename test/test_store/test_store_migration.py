"""Tests for safe dataset migration between stores."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from rdflib import Dataset
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, _QuadType
from rdflib.namespace import RDF, RDFS, XSD, Namespace
from rdflib.plugins.stores.memory import Memory, SimpleMemory
from rdflib.store import Store
from rdflib.store_capabilities import PROBE_PREFIX
from rdflib.store_migration import MigrationError, migrate
from rdflib.term import BNode, IdentifiedNode, Literal, URIRef

EX = Namespace("http://example.org/")
G1_ID = URIRef("http://example.org/g1")
EMPTY_GRAPH_ID = URIRef("http://example.org/empty")


class NoBatchStore(Memory):
    def addN(self, quads: Iterable[_QuadType]) -> None:  # noqa: N802
        raise NotImplementedError("batch writes are not supported")


class QuotaStore(Memory):
    """Passes pre-flight (small probe batches) but rejects larger imports."""

    quota = 6

    def __init__(self) -> None:
        super().__init__()
        self._written = 0

    def addN(self, quads: Iterable[_QuadType]) -> None:  # noqa: N802
        quads = list(quads)
        if self._written + len(quads) > self.quota:
            raise OSError("write quota exceeded")
        self._written += len(quads)
        super().addN(iter(quads))


class BatchRecordingStore(Memory):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def addN(self, quads: Iterable[_QuadType]) -> None:  # noqa: N802
        quads = list(quads)
        self.batch_sizes.append(len(quads))
        super().addN(iter(quads))


class BindOnceStore(Memory):
    """Namespace binding works for the pre-flight probe, then breaks."""

    def __init__(self) -> None:
        super().__init__()
        self._bind_calls = 0

    def bind(self, prefix: str, namespace: URIRef, override: bool = True) -> None:
        self._bind_calls += 1
        if self._bind_calls > 1:
            raise OSError("namespace endpoint closed")
        super().bind(prefix, namespace, override=override)


class TransactionalStore(Memory):
    """Mimics a transaction aware backend: rollback discards the transaction."""

    transaction_aware = True

    def __init__(self, *, fail_import: bool = False) -> None:
        super().__init__()
        self.fail_import = fail_import
        self._addn_calls = 0
        self.commits = 0
        self.rollbacks = 0

    def addN(self, quads: Iterable[_QuadType]) -> None:  # noqa: N802
        quads = list(quads)
        # The first addN call is the 3-quad pre-flight batch; fail once the
        # real migration starts writing.
        if self.fail_import and self._addn_calls > 0:
            raise OSError("transactional write failed")
        self._addn_calls += 1
        super().addN(iter(quads))

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1
        # A real backend discards everything since the failed transaction;
        # emulate that by removing all triples across contexts.
        self.remove((None, None, None), context=None)


def _build_source() -> Dataset:
    source = Dataset()
    source.default_graph.add((EX.a, RDF.type, EX.A))
    source.default_graph.add((EX.b, RDFS.label, Literal("default graph")))

    g1 = source.graph(G1_ID)
    g1.add((EX.c, RDF.type, EX.C))
    g1.add((EX.c, RDFS.label, Literal("named one", lang="en")))
    g1.add((EX.c, EX.size, Literal(42, datatype=XSD.integer)))

    g2 = source.graph(BNode())
    g2.add((EX.d, RDFS.comment, Literal("blank node graph")))
    g2.add((EX.e, RDF.type, EX.E))

    source.graph(EMPTY_GRAPH_ID)  # intentionally empty, graph-aware entry

    source.store.bind("ex", URIRef(EX), override=True)
    source.store.bind("demo", URIRef("http://demo.example/"), override=True)
    return source


def _snapshot(dataset: Dataset) -> dict[IdentifiedNode, frozenset]:
    return {graph.identifier: frozenset(graph) for graph in dataset.graphs()}


def _total_triples(store: Store) -> int:
    dataset = Dataset(store=store)
    return sum(len(graph) for graph in dataset.graphs())


def test_migrate_memory_to_memory_is_graph_by_graph_equal() -> None:
    source = _build_source()
    target_store = Memory()
    source_bindings = dict(source.store.namespaces())

    result = migrate(source, target_store)

    target = Dataset(store=target_store)
    expected = _snapshot(source)
    actual = _snapshot(target)
    assert set(actual) == set(expected)
    for graph_id, triples in expected.items():
        assert actual[graph_id] == triples

    assert result.triples == 7
    assert result.graphs == 4  # default + g1 + bnode graph + empty graph
    assert result.triples_per_graph[DATASET_DEFAULT_GRAPH_ID] == 2
    assert result.triples_per_graph[G1_ID] == 3
    assert result.namespaces == len(source_bindings)
    assert result.report is not None and result.report.supports(
        ["named_graphs", "batch_write", "namespaces", "full_read"]
    )

    # Every source binding is present with the same URI; the pre-flight
    # probe prefix is the only tolerated extra binding.
    target_bindings = dict(target_store.namespaces())
    for prefix, namespace in source_bindings.items():
        assert target_bindings[prefix] == namespace
    extra = set(target_bindings).difference(source_bindings)
    assert extra == {PROBE_PREFIX}

    # No probe data survives the migration.
    assert _total_triples(target_store) == 7
    assert all(
        not str(graph.identifier).startswith("urn:rdflib:capability-probe:")
        for graph in target.graphs()
    )


def test_migrate_accepts_datasets_directly() -> None:
    source = _build_source()
    target = Dataset(store=Memory())

    result = migrate(source, target)
    assert result.triples == 7
    assert _snapshot(target)[G1_ID] == _snapshot(source)[G1_ID]


def test_preflight_blocks_before_any_data_is_written() -> None:
    source = _build_source()
    target_store = SimpleMemory()

    with pytest.raises(MigrationError) as exc_info:
        migrate(source, target_store)
    assert "pre-flight" in str(exc_info.value)
    assert exc_info.value.report is not None

    # The destination contains no migrated triples.
    assert len(target_store) == 0
    # And the source was not modified.
    assert _total_triples(source.store) == 7


def test_preflight_blocks_store_without_batch_writes() -> None:
    source = _build_source()
    target_store = NoBatchStore()

    with pytest.raises(MigrationError) as exc_info:
        migrate(source, target_store)
    assert "batch_write" in str(exc_info.value)
    assert _total_triples(target_store) == 0


def test_failing_import_is_rolled_back_without_partial_data() -> None:
    source = _build_source()
    target_store = QuotaStore()

    with pytest.raises(MigrationError) as exc_info:
        migrate(source, target_store, batch_size=1)
    assert "rolled back" in str(exc_info.value)
    assert exc_info.value.leftover == ()
    assert _total_triples(target_store) == 0

    target = Dataset(store=target_store)
    migrated_ids = set(_snapshot(source))
    for graph_id in migrated_ids:
        assert len(target.graph(graph_id)) == 0


def test_namespace_failure_after_copy_is_rolled_back() -> None:
    source = _build_source()
    target_store = BindOnceStore()

    with pytest.raises(MigrationError) as exc_info:
        migrate(source, target_store)
    assert "rolled back" in str(exc_info.value)
    assert exc_info.value.leftover == ()
    assert _total_triples(target_store) == 0


def test_migration_uses_configured_batch_size() -> None:
    source = _build_source()
    target_store = BatchRecordingStore()

    migrate(source, target_store, batch_size=2)

    # First recorded batch is the 3-quad pre-flight batch probe;
    # migration batches for 7 triples at size 2 are 2 | 2,1 | 2.
    migration_batches = target_store.batch_sizes[1:]
    assert max(migration_batches) <= 2
    assert sum(migration_batches) == 7
    assert sorted(migration_batches) == [1, 2, 2, 2]


def test_existing_graph_collision_aborts_before_writes() -> None:
    source = _build_source()
    target_store = Memory()
    target = Dataset(store=target_store)
    pre_existing = (EX.preexisting, RDFS.label, Literal("was already there"))
    target.graph(G1_ID).add(pre_existing)

    with pytest.raises(MigrationError, match="already contains graphs"):
        migrate(source, target_store)

    # Pre-existing data is left untouched, no migration data added.
    target_after = Dataset(store=target_store)
    assert set(target_after.graph(G1_ID)) == {pre_existing}
    assert _total_triples(target_store) == 1


def test_migration_verification_detects_triple_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate a store that loses one triple of every batch after pre-flight:
    # the copy appears to succeed but read-back verification must fail and
    # trigger rollback.
    source = _build_source()
    target_store = Memory()

    original_add_n = Memory.addN
    call_count = {"n": 0}

    def flaky_add_n(self: Memory, quads: Iterable[_QuadType]) -> None:
        quads = list(quads)
        call_count["n"] += 1
        if call_count["n"] > 1:  # skip the pre-flight probe batch
            quads = quads[:-1] or quads[:1]  # drop the last quad, keep one
        original_add_n(self, iter(quads))

    monkeypatch.setattr(Memory, "addN", flaky_add_n)
    try:
        with pytest.raises(MigrationError, match="rolled back") as exc_info:
            migrate(source, target_store)
        assert exc_info.value.leftover == ()
        assert _total_triples(target_store) == 0
    finally:
        monkeypatch.undo()


def test_source_and_target_must_differ() -> None:
    source = _build_source()
    with pytest.raises(MigrationError, match="different stores"):
        migrate(source, source.store)


def test_batch_size_validated() -> None:
    source = _build_source()
    with pytest.raises(ValueError, match="batch_size"):
        migrate(source, Memory(), batch_size=0)


def test_transactional_store_commits_on_success() -> None:
    source = _build_source()
    target_store = TransactionalStore()

    result = migrate(source, target_store)

    assert result.triples == 7
    assert target_store.commits == 1
    assert target_store.rollbacks == 0
    assert _snapshot(Dataset(store=target_store))[G1_ID] == _snapshot(source)[G1_ID]


def test_transactional_store_rolls_back_transaction_on_failure() -> None:
    source = _build_source()
    target_store = TransactionalStore(fail_import=True)

    with pytest.raises(MigrationError) as exc_info:
        migrate(source, target_store)
    assert "rolled back" in str(exc_info.value)
    assert exc_info.value.leftover == ()
    assert target_store.rollbacks == 1
    assert target_store.commits == 0
    assert _total_triples(target_store) == 0
