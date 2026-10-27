"""
Tests for fixed read views (`rdflib.snapshot`).

A snapshot materializes the statements of a graph once and serves all
paging, cursors and queries from that fixed view, so concurrent writes can
neither duplicate/skip pages nor be mixed into earlier results. When the
source graph changes mid-session the next request fails explicitly and the
caller chooses between overwriting the view and reopening it.
"""

import pytest

from rdflib import (
    Dataset,
    Graph,
    Literal,
    Page,
    Snapshot,
    SnapshotClosedError,
    SnapshotCursor,
    SnapshotError,
    StaleSnapshotError,
    URIRef,
)

P = URIRef("urn:p")


def _populate(graph: Graph, n: int, prefix: str = "urn:s") -> Graph:
    for i in range(n):
        graph.add((URIRef(f"{prefix}{i}"), P, Literal(i)))
    return graph


@pytest.fixture
def graph() -> Graph:
    return _populate(Graph(), 25)


@pytest.fixture
def snapshot(graph: Graph) -> Snapshot:
    return graph.snapshot()


# ---------------------------------------------------------------------------
# revision tracking
# ---------------------------------------------------------------------------


def test_revision_starts_at_zero_and_counts_mutations() -> None:
    graph = Graph()
    assert graph.revision == 0
    graph.add((URIRef("urn:a"), P, Literal(1)))
    assert graph.revision == 1
    graph.add((URIRef("urn:a"), P, Literal(1)))  # duplicate still bumps
    assert graph.revision == 2
    graph.remove((URIRef("urn:a"), None, None))
    assert graph.revision == 3
    graph.addN(
        [
            (URIRef("urn:b"), P, Literal(2), graph),
            (URIRef("urn:c"), P, Literal(3), graph),
        ]
    )
    assert graph.revision == 4


def test_revision_counts_parsing() -> None:
    graph = Graph()
    before = graph.revision
    graph.parse(data="<urn:a> <urn:p> 1 .", format="turtle")
    assert graph.revision > before


def test_failed_mutation_does_not_change_revision() -> None:
    graph = Graph()
    before = graph.revision
    # A bare 4-tuple is not an iterable of quads, addN fails while
    # consuming the argument; the revision must not be bumped.
    with pytest.raises((TypeError, ValueError)):
        graph.addN((URIRef("urn:a"), P, Literal(1), graph))  # type: ignore[arg-type]
    assert graph.revision == before


# ---------------------------------------------------------------------------
# the fixed view does not move with the live graph
# ---------------------------------------------------------------------------


def test_snapshot_is_created_from_graph(graph: Graph, snapshot: Snapshot) -> None:
    assert isinstance(snapshot, Snapshot)
    assert snapshot.source is graph
    assert snapshot.revision == graph.revision
    assert not snapshot.is_stale
    assert not snapshot.closed
    assert isinstance(snapshot.id, str) and len(snapshot.id) == 32
    assert len(snapshot) == 25


def test_snapshot_content_is_fixed_when_opened(
    graph: Graph, snapshot: Snapshot
) -> None:
    expected = tuple(sorted(graph))
    assert tuple(snapshot) == expected
    assert len(snapshot) == 25

    # Writes and deletes on the live graph never reach the fixed view.
    graph.add((URIRef("urn:new"), P, Literal("new")))
    graph.remove((URIRef("urn:s0"), P, Literal(0)))
    assert len(graph) == 25  # one added, one removed
    assert len(snapshot) == 25
    assert tuple(snapshot) == expected
    assert (URIRef("urn:s0"), P, Literal(0)) in snapshot
    assert (URIRef("urn:new"), P, Literal("new")) not in snapshot
    assert snapshot.is_stale


def test_snapshot_triples_filters_materialized_statements(snapshot: Snapshot) -> None:
    subjects = set(snapshot.triples((None, P, None)))
    assert len(subjects) == 25
    assert list(snapshot.triples((URIRef("urn:s0"), None, None))) == [
        (URIRef("urn:s0"), P, Literal(0))
    ]
    assert list(snapshot.triples((None, None, Literal(24)))) == [
        (URIRef("urn:s24"), P, Literal(24))
    ]
    assert list(snapshot.triples((URIRef("urn:missing"), None, None))) == []


def test_snapshot_opened_with_pattern() -> None:
    graph = _populate(Graph(), 5)
    graph.add((URIRef("urn:other"), URIRef("urn:q"), Literal("n")))
    snapshot = graph.snapshot((None, P, None))
    assert len(snapshot) == 5
    assert all(p == P for _s, p, _o in snapshot)


def test_snapshot_rejects_bad_patterns(graph: Graph) -> None:
    with pytest.raises(TypeError):
        graph.snapshot(("not a tuple",))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        graph.snapshot(("s", None, None))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        graph.snapshot((None, P / P, None))
    snapshot = graph.snapshot()
    with pytest.raises(TypeError):
        snapshot.cursor(triple=("s", None, None))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# paging
# ---------------------------------------------------------------------------


def test_cursor_pages_partition_the_view_without_gaps_or_duplicates(
    snapshot: Snapshot,
) -> None:
    cursor = snapshot.cursor(page_size=10)
    assert isinstance(cursor, SnapshotCursor)
    assert cursor.pages == 3
    assert cursor.total == 25
    assert cursor.page_number == 0
    assert cursor.has_next

    pages = list(cursor)
    assert [page.number for page in pages] == [1, 2, 3]
    assert [len(page) for page in pages] == [10, 10, 5]
    assert [page.is_last for page in pages] == [False, False, True]
    assert all(isinstance(page, Page) for page in pages)

    # Every page belongs to exactly the same fixed view.
    assert {page.snapshot_id for page in pages} == {snapshot.id}

    paged = [statement for page in pages for statement in page]
    assert paged == list(snapshot)
    assert len(paged) == len(set(paged)) == 25
    assert not cursor.has_next


def test_cursor_stop_after_last_page(snapshot: Snapshot) -> None:
    cursor = snapshot.cursor(page_size=100)
    page = cursor.next_page()
    assert page.number == 1 and page.is_last
    with pytest.raises(StopIteration):
        cursor.next_page()


def test_empty_snapshot_has_no_pages() -> None:
    snapshot = Graph().snapshot()
    cursor = snapshot.cursor(page_size=10)
    assert cursor.pages == 0
    assert not cursor.has_next
    with pytest.raises(StopIteration):
        cursor.next_page()
    with pytest.raises(IndexError):
        cursor.page(1)


def test_explicit_page_requests_never_move_silently(snapshot: Snapshot) -> None:
    cursor = snapshot.cursor(page_size=10)
    page = cursor.page(3)
    assert page.number == 3
    assert len(page) == 5
    assert cursor.page_number == 3
    with pytest.raises(StopIteration):
        cursor.next_page()

    # A jump is allowed, but only explicitly and within bounds.
    another = snapshot.cursor(page_size=10)
    assert another.page(1).number == 1
    assert another.next_page().number == 2
    for invalid in (0, 4, -1):
        with pytest.raises(IndexError):
            another.page(invalid)  # type: ignore[arg-type]


def test_cursor_validates_page_size(snapshot: Snapshot) -> None:
    with pytest.raises(ValueError):
        snapshot.cursor(page_size=0)
    with pytest.raises(TypeError):
        snapshot.cursor(page_size=True)  # type: ignore[arg-type]


def test_pattern_cursor_pages_filtered_subset(graph: Graph) -> None:
    snapshot = graph.snapshot()
    cursor = snapshot.cursor(triple=(None, None, None), page_size=7)
    assert cursor.pages == 4
    pages = list(cursor)
    assert [len(page) for page in pages] == [7, 7, 7, 4]
    assert [statement for page in pages for statement in page] == list(snapshot)


# ---------------------------------------------------------------------------
# concurrent writes: explicit choice, no silent page shifts
# ---------------------------------------------------------------------------


def test_write_between_pages_raises_stale(graph: Graph, snapshot: Snapshot) -> None:
    cursor = snapshot.cursor(page_size=10)
    first_page = cursor.next_page()
    assert len(first_page) == 10

    graph.add((URIRef("urn:inserted"), P, Literal("late")))
    assert snapshot.is_stale

    with pytest.raises(StaleSnapshotError) as exc_info:
        cursor.next_page()
    error = exc_info.value
    assert error.snapshot is snapshot
    assert error.captured_revision == snapshot.revision
    assert error.current_revision == graph.revision
    assert "overwrite()" in str(error)
    assert "graph.snapshot()" in str(error)

    # The page already served is untouched by the later write.
    assert not any(s == URIRef("urn:inserted") for s, _p, _o in first_page)


def test_remove_between_pages_also_raises_stale(
    graph: Graph, snapshot: Snapshot
) -> None:
    cursor = snapshot.cursor(page_size=10)
    cursor.next_page()
    graph.remove((URIRef("urn:s20"), None, None))
    with pytest.raises(StaleSnapshotError):
        cursor.next_page()


def test_new_writes_are_not_mixed_into_earlier_pages(graph: Graph) -> None:
    snapshot = graph.snapshot()
    cursor = snapshot.cursor(page_size=5)
    seen: list = []
    for _ in range(3):
        seen.extend(cursor.next_page())
    graph.add((URIRef("urn:zzz"), P, Literal("late")))
    graph.remove((URIRef("urn:s1"), None, None))
    with pytest.raises(StaleSnapshotError):
        cursor.next_page()
    # The 15 statements already paged remain exactly the first fixed page.
    assert seen == list(snapshot)[:15]
    assert len(seen) == len(set(seen)) == 15


def test_overwrite_replaces_view_and_invalidates_cursors(
    graph: Graph, snapshot: Snapshot
) -> None:
    cursor = snapshot.cursor(page_size=10)
    cursor.next_page()
    graph.add((URIRef("urn:new1"), P, Literal("a")))
    graph.add((URIRef("urn:new2"), P, Literal("b")))

    old_id = snapshot.id
    with pytest.raises(StaleSnapshotError):
        cursor.next_page()

    # Explicit recovery: overwrite the fixed view in place.
    result = snapshot.overwrite()
    assert result is snapshot
    assert snapshot.id != old_id
    assert snapshot.revision == graph.revision
    assert not snapshot.is_stale
    assert len(snapshot) == 27

    # Old cursors cannot be continued against the replacement view.
    with pytest.raises(SnapshotError):
        cursor.next_page()

    # A new cursor pages the replaced view from its beginning.
    new_cursor = snapshot.cursor(page_size=10)
    pages = list(new_cursor)
    assert sum(len(page) for page in pages) == 27
    assert all(page.snapshot_id == snapshot.id for page in pages)


def test_close_and_reopen_starts_a_new_view(graph: Graph, snapshot: Snapshot) -> None:
    cursor = snapshot.cursor(page_size=10)
    cursor.next_page()
    graph.add((URIRef("urn:new"), P, Literal("a")))

    old_id = snapshot.id
    snapshot.close()
    assert snapshot.closed
    with pytest.raises(SnapshotClosedError):
        snapshot.cursor()
    with pytest.raises(SnapshotClosedError):
        snapshot.query("ASK { ?s ?p ?o }")
    with pytest.raises(SnapshotClosedError):
        cursor.next_page()

    # Explicit recovery: reopen a fresh view.
    reopened = graph.snapshot()
    assert reopened.id != old_id
    assert len(reopened) == 26
    assert list(reopened.cursor(page_size=100).next_page())[0] is not None


def test_overwrite_closed_snapshot_fails(snapshot: Snapshot) -> None:
    snapshot.close()
    with pytest.raises(SnapshotClosedError):
        snapshot.overwrite()


def test_context_manager_closes_snapshot(graph: Graph) -> None:
    with graph.snapshot() as view:
        assert len(view) == 25
    assert view.closed


# ---------------------------------------------------------------------------
# queries run against the same fixed view
# ---------------------------------------------------------------------------

SELECT_QUERY = "SELECT ?s WHERE { ?s ?p ?o } ORDER BY ?s"


def test_query_is_evaluated_against_the_fixed_view(
    graph: Graph, snapshot: Snapshot
) -> None:
    first = list(snapshot.query(SELECT_QUERY))
    assert len(first) == 25

    graph.add((URIRef("urn:late"), P, Literal("late")))
    with pytest.raises(StaleSnapshotError):
        snapshot.query(SELECT_QUERY)

    snapshot.overwrite()
    rows = list(snapshot.query(SELECT_QUERY))
    assert len(rows) == 26


def test_limit_offset_queries_are_consistent_on_a_snapshot(
    graph: Graph, snapshot: Snapshot
) -> None:
    # Separate LIMIT/OFFSET requests are the classic source of duplicated or
    # missing rows on a changing graph; on the fixed view they partition it.
    page_one = list(snapshot.query(SELECT_QUERY + " LIMIT 10"))
    page_two = list(snapshot.query(SELECT_QUERY + " LIMIT 10 OFFSET 10"))
    page_three = list(snapshot.query(SELECT_QUERY + " LIMIT 10 OFFSET 20"))
    rows = page_one + page_two + page_three
    assert [len(page_one), len(page_two), len(page_three)] == [10, 10, 5]
    assert len(rows) == len({tuple(row) for row in rows}) == 25
    assert {str(row.s) for row in rows} == {f"urn:s{i}" for i in range(25)}


def test_query_cursor_pages_rows(graph: Graph, snapshot: Snapshot) -> None:
    cursor = snapshot.query_cursor(SELECT_QUERY, page_size=10)
    pages = list(cursor)
    assert [page.number for page in pages] == [1, 2, 3]
    assert [len(page) for page in pages] == [10, 10, 5]
    assert all(page.snapshot_id == snapshot.id for page in pages)
    rows = [row for page in pages for row in page]
    assert len(rows) == 25
    assert len({tuple(row) for row in rows}) == 25


def test_query_cursor_guards_staleness(graph: Graph, snapshot: Snapshot) -> None:
    cursor = snapshot.query_cursor(SELECT_QUERY, page_size=10)
    cursor.next_page()
    graph.add((URIRef("urn:late"), P, Literal("late")))
    with pytest.raises(StaleSnapshotError):
        cursor.next_page()


def test_query_cursor_only_supports_select(snapshot: Snapshot) -> None:
    with pytest.raises(SnapshotError):
        snapshot.query_cursor("ASK { ?s ?p ?o }")
    with pytest.raises(SnapshotError):
        snapshot.query_cursor("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


def test_dataset_snapshot_covers_union_of_graphs() -> None:
    dataset = Dataset()
    one = URIRef("urn:g1")
    two = URIRef("urn:g2")
    dataset.addN(
        [(URIRef(f"urn:a{i}"), P, Literal(i), one) for i in range(5)]
        + [(URIRef(f"urn:b{i}"), P, Literal(i), two) for i in range(5)]
    )
    dataset.add((URIRef("urn:default"), P, Literal("d")))

    snapshot = dataset.snapshot()
    assert len(snapshot) == 11
    cursor = snapshot.cursor(page_size=4)
    pages = list(cursor)
    assert [len(page) for page in pages] == [4, 4, 3]

    dataset.add((URIRef("urn:written"), P, Literal("x"), one))
    with pytest.raises(StaleSnapshotError):
        cursor.next_page()

    snapshot.overwrite()
    assert len(snapshot) == 12
    assert not snapshot.is_stale
