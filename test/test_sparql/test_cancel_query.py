"""
Cooperative cancellation of long-running SPARQL queries.

A ``threading.Event`` passed to ``Graph.query`` can be set from another
thread (e.g. when a user closes a page). Evaluation then stops before
producing or reading the next result, keeps the rows it already produced,
releases the store iterator it was reading from, and reports how it ended
through ``Result.status``:

* ``ResultStatus.COMPLETED`` -- the query ran to the end,
* ``ResultStatus.CANCELLED`` -- stopped by a cancellation request,
* ``ResultStatus.ERROR`` -- failed with an exception (still re-raised).
"""

from __future__ import annotations

import threading
import time
import types
from typing import Any

import pytest

from rdflib import Graph, Literal, URIRef
from rdflib.plugins.sparql import evaluate as eval_module
from rdflib.plugins.sparql import prepareQuery
from rdflib.query import QueryCancelledError, ResultStatus

EX = "http://example.org/"

SELECT_ALL = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"


def _graph(size: int = 50) -> Graph:
    g = Graph()
    for i in range(size):
        g.add((URIRef(f"{EX}s{i}"), URIRef(f"{EX}p{i % 5}"), Literal(i)))
    return g


def _wrap_triples(
    g: Graph,
    *,
    on_pull: Any = None,
    on_close: Any = None,
    fail_after: int | None = None,
) -> None:
    """Replace ``g.triples`` with an instrumented generator.

    ``on_pull`` is called for every triple pulled from the store,
    ``on_close`` when the store generator is closed (GeneratorExit), and
    the generator raises ``RuntimeError`` after ``fail_after`` pulls.
    """
    original = Graph.triples

    def instrumented(self: Graph, triple: Any):
        inner = original(self, triple)
        count = 0
        try:
            for row in inner:
                if fail_after is not None and count >= fail_after:
                    raise RuntimeError("store failed mid-query")
                count += 1
                if on_pull is not None:
                    on_pull()
                yield row
        except GeneratorExit:
            if on_close is not None:
                on_close()
            raise

    g.triples = types.MethodType(instrumented, g)


def test_completed_status() -> None:
    g = _graph(10)
    result = g.query(SELECT_ALL)
    # A streamed SELECT is RUNNING until it is drained.
    assert result.status is ResultStatus.RUNNING
    assert len(result) == 10
    assert result.status is ResultStatus.COMPLETED
    assert result.error is None


def test_cancelled_before_start_reads_nothing() -> None:
    g = _graph(10)
    pulls: list[int] = []
    _wrap_triples(g, on_pull=lambda: pulls.append(1))

    event = threading.Event()
    event.set()
    result = g.query(SELECT_ALL, event=event)

    rows = list(result)
    assert rows == []
    assert result.status is ResultStatus.CANCELLED
    assert isinstance(result.error, QueryCancelledError)
    assert pulls == []  # the store is never read after a cancellation


def test_cancel_mid_iteration_keeps_partial_results() -> None:
    g = _graph(50)
    event = threading.Event()
    result = g.query(SELECT_ALL, event=event)

    rows = []
    for index, row in enumerate(result):
        rows.append(row)
        if index == 9:
            event.set()

    assert result.status is ResultStatus.CANCELLED
    assert isinstance(result.error, QueryCancelledError)
    assert len(rows) == 10
    # The rows produced before the stop are retained on the result.
    assert len(result.bindings) == 10


def test_cancel_stops_store_reads_and_closes_iterator() -> None:
    g = _graph(50)
    pulls = 0
    closes = 0

    def count_pull() -> None:
        nonlocal pulls
        pulls += 1

    def count_close() -> None:
        nonlocal closes
        closes += 1

    _wrap_triples(g, on_pull=count_pull, on_close=count_close)

    event = threading.Event()
    result = g.query(SELECT_ALL, event=event)
    rows = []
    for index, _row in enumerate(result):
        rows.append(_row)
        if index == 4:
            event.set()

    assert result.status is ResultStatus.CANCELLED
    assert len(rows) == 5
    # Exactly the rows that became results were read: no extra store pull
    # happens after the cancellation, and the store iterator is closed.
    assert pulls == 5
    assert closes >= 1

    # Iterating or draining the result afterwards performs no store reads.
    pulls_before = pulls
    assert list(result) == rows
    assert result.bindings == result.bindings  # idempotent drain
    assert pulls == pulls_before
    assert result.status is ResultStatus.CANCELLED


def test_join_stops_at_result_boundary() -> None:
    g = Graph()
    for i in range(20):
        g.add((URIRef(EX + "s"), URIRef(EX + "p"), Literal(i)))
        g.add((URIRef(EX + "s"), URIRef(EX + "q"), Literal(i)))

    event = threading.Event()
    result = g.query(
        "SELECT ?v1 ?v2 WHERE { ?s ?p1 ?v1 . ?s ?p2 ?v2 }",
        event=event,
    )
    rows = []
    for index, row in enumerate(result):
        rows.append(row)
        if index == 9:
            event.set()

    assert result.status is ResultStatus.CANCELLED
    assert len(rows) == 10


def test_error_status_is_reported_and_partial_rows_kept() -> None:
    g = _graph(50)
    _wrap_triples(g, fail_after=3)

    result = g.query(SELECT_ALL)
    rows = []
    with pytest.raises(RuntimeError, match="store failed mid-query"):
        for row in result:
            rows.append(row)

    assert result.status is ResultStatus.ERROR
    assert isinstance(result.error, RuntimeError)
    # Rows produced before the failure were delivered and are retained.
    assert len(rows) == 3
    assert len(result.bindings) == 3


def test_cancel_eager_order_by_raises() -> None:
    # ORDER BY sorts while the query is built, before a result exists, so a
    # cancellation surfaces as QueryCancelledError from query().
    g = _graph(50)
    event = threading.Event()
    event.set()

    with pytest.raises(QueryCancelledError):
        g.query("SELECT ?s WHERE { ?s ?p ?o } ORDER BY ?s", event=event)


def test_cancel_grouped_query_reports_status() -> None:
    # Aggregation streams through a generator, so it ends as CANCELLED.
    g = _graph(50)
    event = threading.Event()
    event.set()

    grouped = g.query(
        "SELECT ?p (COUNT(*) AS ?n) WHERE { ?s ?p ?o } GROUP BY ?p",
        event=event,
    )
    assert list(grouped) == []
    assert grouped.status is ResultStatus.CANCELLED


def test_cancel_ask_and_construct_raises() -> None:
    g = _graph(50)
    event = threading.Event()
    event.set()

    with pytest.raises(QueryCancelledError):
        g.query("ASK { ?s ?p ?o }", event=event)

    with pytest.raises(QueryCancelledError):
        g.query(
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }",
            event=event,
        )


def test_service_silent_does_not_swallow_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SERVICE SILENT catches Exception to insert an empty solution; a
    # cancellation must not be swallowed, and no endpoint is contacted.
    def forbidden_urlopen(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("cancelled query must not contact an endpoint")

    monkeypatch.setattr(eval_module, "urlopen", forbidden_urlopen)

    g = Graph()
    event = threading.Event()
    event.set()

    result = g.query(
        "SELECT * WHERE { SERVICE SILENT <http://127.0.0.1:1/sparql> { ?s ?p ?o } }",
        event=event,
    )
    assert list(result) == []
    assert result.status is ResultStatus.CANCELLED


def test_cancel_from_other_thread() -> None:
    # Simulates a user closing a page while the worker keeps evaluating:
    # the store is slow, the main thread requests cancellation, the worker
    # ends early with partial results and releases the store iterator.
    g = _graph(50)
    gate = threading.Event()

    def slow_pull() -> None:
        # Block the first pull until the worker is known to be evaluating,
        # then let it proceed slowly.
        gate.set()
        time.sleep(0.005)

    _wrap_triples(g, on_pull=slow_pull)

    event = threading.Event()
    outcome: dict[str, Any] = {}

    def worker() -> None:
        result = g.query(SELECT_ALL, event=event)
        outcome["result"] = result
        outcome["rows"] = list(result)

    thread = threading.Thread(target=worker)
    thread.start()
    assert gate.wait(5)
    event.set()
    thread.join(5)

    assert not thread.is_alive()
    result = outcome["result"]
    assert result.status is ResultStatus.CANCELLED
    assert 0 < len(outcome["rows"]) < 50


def test_prepared_query_event() -> None:
    g = _graph(10)
    query = prepareQuery(SELECT_ALL)
    event = threading.Event()
    event.set()
    result = g.query(query, event=event)
    assert list(result) == []
    assert result.status is ResultStatus.CANCELLED
