"""
Tests for the SPARQL Update preview-then-commit workflow
(`rdflib.plugins.sparql.preview`).
"""

import pytest

from rdflib import BNode, ConjunctiveGraph, Dataset, Graph, Literal, URIRef
from rdflib.plugins.sparql import (
    StaleUpdatePlanError,
    UpdateApplicationError,
    UpdatePreviewError,
    prepareUpdate,
    preview_update,
)
from rdflib.plugins.sparql.preview import Quad, _snapshot

EX = "http://example.org/"
GID = URIRef(EX + "graph")
G1 = URIRef(EX + "g1")

SEED_TRIPLES = [
    (URIRef(EX + "a"), URIRef(EX + "p"), URIRef(EX + "b")),
    (URIRef(EX + "b"), URIRef(EX + "p"), URIRef(EX + "c")),
]

GRAPH_FACTORIES = [Graph, Dataset, ConjunctiveGraph]


def _make_graph(factory: type[Graph]) -> Graph:
    """Build an empty graph with a stable identifier across twin graphs."""

    if factory is Graph:
        return Graph(identifier=GID)
    if factory is Dataset:
        return Dataset()
    return ConjunctiveGraph(identifier=GID)


def _seed(graph: Graph, triples=SEED_TRIPLES) -> Graph:
    for triple in triples:
        graph.add(triple)
    return graph


def _triples(quads: frozenset[Quad]):
    return {(s, p, o) for s, p, o, _ in quads}


@pytest.mark.parametrize("factory", GRAPH_FACTORIES)
def test_insert_data_preview_and_apply(factory: type[Graph]) -> None:
    graph = _seed(_make_graph(factory))
    request = """PREFIX ex: <http://example.org/>
    INSERT DATA { ex:a ex:label "A" . }
    """

    plan = preview_update(graph, request)
    preview = plan.preview

    assert preview.added_count == 1
    assert preview.deleted_count == 0
    assert preview.unchanged_count == len(SEED_TRIPLES)
    assert _triples(preview.additions) == {
        (URIRef(EX + "a"), URIRef(EX + "label"), Literal("A"))
    }
    # Previewing never writes to the target graph.
    assert _snapshot(graph) == _snapshot(_seed(_make_graph(factory)))

    assert plan.is_current()
    plan.apply(request)
    assert (URIRef(EX + "a"), URIRef(EX + "label"), Literal("A")) in graph
    resulting = _snapshot(graph)
    assert resulting == (_snapshot(_seed(_make_graph(factory))) | preview.additions)


@pytest.mark.parametrize("factory", GRAPH_FACTORIES)
def test_delete_data_preview(factory: type[Graph]) -> None:
    graph = _seed(_make_graph(factory))
    request = """PREFIX ex: <http://example.org/>
    DELETE DATA { ex:a ex:p ex:b . }
    """

    plan = preview_update(graph, request)
    assert plan.preview.added_count == 0
    assert plan.preview.deleted_count == 1
    assert plan.preview.unchanged_count == len(SEED_TRIPLES) - 1

    plan.apply(request)
    assert (URIRef(EX + "a"), URIRef(EX + "p"), URIRef(EX + "b")) not in graph
    assert (URIRef(EX + "b"), URIRef(EX + "p"), URIRef(EX + "c")) in graph


@pytest.mark.parametrize("factory", GRAPH_FACTORIES)
def test_modify_preview_matches_real_execution(factory: type[Graph]) -> None:
    request = """PREFIX ex: <http://example.org/>
    DELETE { ?s ex:p ?o }
    INSERT { ?s ex:seen true }
    WHERE { ?s ex:p ?o }
    """

    control = _seed(_make_graph(factory))
    target = _seed(_make_graph(factory))
    # A named-graph triple not matched by the WHERE clause must survive.
    if isinstance(control, ConjunctiveGraph):
        assert isinstance(target, ConjunctiveGraph)
        control.get_context(G1).add((URIRef(EX + "n"), URIRef(EX + "keep"), Literal(1)))
        target.get_context(G1).add((URIRef(EX + "n"), URIRef(EX + "keep"), Literal(1)))

    control.update(request)
    result = preview_update(target, request).apply(request)

    assert _snapshot(control) == _snapshot(target)
    # The post-commit graph is exactly the previewed transformation.
    expected = (
        _snapshot(_seed(_make_graph(factory))) - result.deletions
    ) | result.additions
    if isinstance(target, ConjunctiveGraph):
        expected |= frozenset({(URIRef(EX + "n"), URIRef(EX + "keep"), Literal(1), G1)})
    assert _snapshot(target) == expected


def test_modify_bnode_is_materialized_identically() -> None:
    graph = _seed(Graph(identifier=GID))
    request = """PREFIX ex: <http://example.org/>
    INSERT { [] ex:p ?o } WHERE { ?s ex:p ?o }
    """

    plan = preview_update(graph, request)
    added_subjects = {s for s, _, _, _ in plan.preview.additions}
    # One fresh BNode is materialized per WHERE solution.
    assert len(added_subjects) == len(SEED_TRIPLES)
    assert all(isinstance(subject, BNode) for subject in added_subjects)

    plan.apply(request)
    # The concrete BNode minted during preview is the one committed.
    assert _snapshot(graph) == (
        _snapshot(_seed(Graph(identifier=GID))) | plan.preview.additions
    )


def test_failed_statement_aborts_preview_and_keeps_target() -> None:
    graph = _seed(Graph(identifier=GID))
    before = _snapshot(graph)
    request = """PREFIX ex: <http://example.org/>
    INSERT DATA { ex:a ex:p ex:new . } ;
    CREATE GRAPH <http://example.org/g1> ;
    INSERT DATA { ex:c ex:p ex:d . }
    """

    with pytest.raises(UpdatePreviewError) as exc_info:
        preview_update(graph, request)
    error = exc_info.value
    assert error.statement_index == 1
    assert error.operation == "Create"
    # Nothing was written to the real graph, not even statement 1.
    assert _snapshot(graph) == before


def test_silent_failure_is_recorded_but_does_not_abort() -> None:
    graph = _seed(Graph(identifier=GID))
    # CREATE always fails in rdflib; SILENT must swallow that.
    request = """PREFIX ex: <http://example.org/>
    INSERT DATA { ex:a ex:p ex:new . } ;
    CREATE SILENT GRAPH <http://example.org/g1>
    """

    plan = preview_update(graph, request)
    effects = plan.preview.statements
    assert effects[0].succeeded is True
    assert effects[1].succeeded is False
    assert effects[1].silent is True
    assert plan.preview.added_count == 1

    plan.apply(request)
    assert (URIRef(EX + "a"), URIRef(EX + "p"), URIRef(EX + "new")) in graph


def test_delete_missing_conflict() -> None:
    graph = _seed(Graph(identifier=GID))
    request = """PREFIX ex: <http://example.org/>
    DELETE DATA { ex:zzz ex:missing ex:1 . }
    """

    plan = preview_update(graph, request)
    conflicts = plan.preview.conflicts
    assert len(conflicts) == 1
    assert conflicts[0].kind == "delete_missing"
    assert conflicts[0].severity == "conflict"
    assert conflicts[0].statements == (0,)
    assert plan.preview.added_count == 0
    assert plan.preview.deleted_count == 0


def test_insert_existing_conflict_is_unchanged() -> None:
    graph = _seed(Graph(identifier=GID))
    request = """PREFIX ex: <http://example.org/>
    INSERT DATA { ex:a ex:p ex:b . ex:new ex:p ex:x . }
    """

    plan = preview_update(graph, request)
    conflicts = {c.kind: c for c in plan.preview.conflicts}
    assert "insert_existing" in conflicts
    assert conflicts["insert_existing"].severity == "warning"
    assert plan.preview.added_count == 1
    existing = _triples(conflicts["insert_existing"].quads)
    assert existing == {(URIRef(EX + "a"), URIRef(EX + "p"), URIRef(EX + "b"))}
    referenced = _triples(plan.preview.unchanged_referenced)
    assert (URIRef(EX + "a"), URIRef(EX + "p"), URIRef(EX + "b")) in referenced


def test_statement_order_conflict() -> None:
    graph = _seed(Graph(identifier=GID))
    request = """PREFIX ex: <http://example.org/>
    INSERT DATA { ex:q ex:p ex:r . } ;
    DELETE DATA { ex:q ex:p ex:r . }
    """

    plan = preview_update(graph, request)
    order_conflicts = [c for c in plan.preview.conflicts if c.kind == "statement_order"]
    assert len(order_conflicts) == 1
    assert order_conflicts[0].statements == (0, 1)
    # The later DELETE wins: net effect is no additions, one deletion? No:
    # the quad did not exist before either, so the net diff is empty.
    assert plan.preview.added_count == 0
    assert plan.preview.deleted_count == 0
    assert not order_conflicts[0].quads & plan.preview.additions


def test_delete_missing_then_insert_is_not_order_conflict() -> None:
    graph = _seed(Graph(identifier=GID))
    # The delete is vacuous (quad absent); it is flagged as delete_missing,
    # but there is no actual add/remove overlap between the statements.
    request = """PREFIX ex: <http://example.org/>
    DELETE DATA { ex:q ex:p ex:r . } ;
    INSERT DATA { ex:q ex:p ex:r . }
    """

    plan = preview_update(graph, request)
    kinds = [c.kind for c in plan.preview.conflicts]
    assert kinds == ["delete_missing"]
    assert plan.preview.added_count == 1
    assert plan.preview.deleted_count == 0
    plan.apply(request)
    assert (URIRef(EX + "q"), URIRef(EX + "p"), URIRef(EX + "r")) in graph


def test_rollback_on_write_failure_restores_graph() -> None:
    graph = _seed(Graph(identifier=GID))
    before = _snapshot(graph)
    request = """PREFIX ex: <http://example.org/>
    INSERT DATA {
        ex:n1 ex:p "1" .
        ex:n2 ex:p "2" .
        ex:n3 ex:p "3" .
        ex:n4 ex:p "4" .
    }
    """
    plan = preview_update(graph, request)

    original_add = graph.store.add
    calls = 0

    def flaky_add(triple, context=None, quoted=False):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated store failure")
        return original_add(triple, context, quoted=quoted)

    graph.store.add = flaky_add  # type: ignore[method-assign]
    try:
        with pytest.raises(UpdateApplicationError):
            plan.apply(request)
    finally:
        graph.store.add = original_add  # type: ignore[method-assign]

    assert _snapshot(graph) == before
    assert plan.applied is False


def test_stale_plan_when_target_changed() -> None:
    graph = _seed(Graph(identifier=GID))
    before = _snapshot(graph)
    request = """PREFIX ex: <http://example.org/>
    INSERT DATA { ex:a ex:p ex:new . }
    """
    plan = preview_update(graph, request)

    graph.add((URIRef(EX + "intruder"), URIRef(EX + "p"), Literal("x")))
    assert not plan.is_current()
    with pytest.raises(StaleUpdatePlanError) as exc_info:
        plan.apply(request)
    assert len(exc_info.value.external_additions) == 1
    # Nothing was written by the rejected commit.
    assert (URIRef(EX + "a"), URIRef(EX + "p"), URIRef(EX + "new")) not in graph

    # After the external change is undone the old plan can commit.
    graph.remove((URIRef(EX + "intruder"), URIRef(EX + "p"), None))
    assert plan.is_current()
    plan.apply(request)
    assert _snapshot(graph) == before | plan.preview.additions


def test_stale_plan_when_request_text_changed() -> None:
    graph = _seed(Graph(identifier=GID))
    plan = preview_update(
        graph,
        """PREFIX ex: <http://example.org/>
        INSERT DATA { ex:a ex:p ex:new . }
        """,
    )
    with pytest.raises(StaleUpdatePlanError):
        plan.apply(
            """PREFIX ex: <http://example.org/>
            INSERT DATA { ex:a ex:p ex:DIFFERENT . }
            """
        )
    assert _snapshot(graph) == _snapshot(_seed(Graph(identifier=GID)))


def test_plan_is_one_shot() -> None:
    graph = _seed(Graph(identifier=GID))
    request = """PREFIX ex: <http://example.org/>
    INSERT DATA { ex:a ex:p ex:new . }
    """
    plan = preview_update(graph, request)
    plan.apply(request)
    assert plan.applied is True
    with pytest.raises(StaleUpdatePlanError):
        plan.apply(request)


def test_prepared_update_object() -> None:
    graph = _seed(Graph(identifier=GID))
    text = """PREFIX ex: <http://example.org/>
    INSERT DATA { ex:a ex:p ex:new . }
    """
    prepared = prepareUpdate(text)
    plan = preview_update(graph, prepared)
    plan.apply()  # no text: apply the reviewed parsed object directly
    assert (URIRef(EX + "a"), URIRef(EX + "p"), URIRef(EX + "new")) in graph

    # prepareUpdate keeps the original text, so applying with the same text
    # works; applying with altered text is rejected.
    plan2 = preview_update(_seed(Graph(identifier=GID)), prepared)
    plan2.apply(text)
    plan3 = preview_update(_seed(Graph(identifier=GID)), prepared)
    with pytest.raises(StaleUpdatePlanError):
        plan3.apply(text + " ")


def test_dataset_named_graph_quads() -> None:
    graph = Dataset()
    graph.get_context(G1).add((URIRef(EX + "y"), URIRef(EX + "p"), Literal("1")))
    request = """PREFIX ex: <http://example.org/>
    INSERT DATA {
        ex:z ex:p "default" .
        GRAPH <http://example.org/g1> { ex:z2 ex:p "named" . }
    } ;
    DELETE DATA {
        GRAPH <http://example.org/g1> { ex:y ex:p "1" . }
    }
    """

    plan = preview_update(graph, request)
    preview = plan.preview
    added_contexts = {g for _, _, _, g in preview.additions}
    removed_contexts = {g for _, _, _, g in preview.deletions}
    assert G1 in added_contexts
    assert G1 in removed_contexts

    plan.apply(request)
    g1 = graph.get_context(G1)
    assert (URIRef(EX + "z2"), URIRef(EX + "p"), Literal("named")) in g1
    assert (URIRef(EX + "y"), URIRef(EX + "p"), Literal("1")) not in g1
    assert (URIRef(EX + "z"), URIRef(EX + "p"), Literal("default")) in (
        graph.default_graph
    )
    assert _snapshot(graph) == (
        _snapshot(Dataset())
        | frozenset({(URIRef(EX + "y"), URIRef(EX + "p"), Literal("1"), G1)})
        - preview.deletions
        | preview.additions
    )


def test_per_statement_effects_sum_to_actual_writes() -> None:
    graph = _seed(Graph(identifier=GID))
    request = """PREFIX ex: <http://example.org/>
    INSERT DATA { ex:n1 ex:p "1" . } ;
    DELETE WHERE { ?s ex:p ex:b . } ;
    INSERT DATA { ex:n2 ex:p "2" . }
    """
    plan = preview_update(graph, request)
    effects = plan.preview.statements
    assert [e.operation for e in effects] == [
        "InsertData",
        "DeleteWhere",
        "InsertData",
    ]
    # Gross per-statement writes (before cancellation by order).
    assert len(set().union(*(e.added for e in effects))) == 2
    assert len(set().union(*(e.removed for e in effects))) == 1
    # Net preview is what the commit produces.
    plan.apply(request)
    assert _snapshot(graph) == (
        _snapshot(_seed(Graph(identifier=GID))) - plan.preview.deletions
        | plan.preview.additions
    )


def test_summary_mentions_counts_and_conflicts() -> None:
    graph = _seed(Graph(identifier=GID))
    request = """PREFIX ex: <http://example.org/>
    INSERT DATA { ex:a ex:p ex:b . ex:new ex:p ex:x . } ;
    DELETE DATA { ex:zzz ex:missing ex:1 . }
    """
    summary = preview_update(graph, request).preview.summary()
    assert "1 addition(s)" in summary
    assert "delete_missing" in summary
    assert "insert_existing" in summary


@pytest.mark.parametrize("factory", GRAPH_FACTORIES)
def test_preview_is_independent_of_target_store(factory: type[Graph]) -> None:
    # The shadow must be an independent store: executing the request while
    # previewing cannot leak into the target.
    graph = _seed(_make_graph(factory))
    request = """PREFIX ex: <http://example.org/>
    INSERT DATA { ex:a ex:p ex:new . ex:a ex:p ex:new2 . } ;
    DELETE WHERE { ?s ex:p ex:b . }
    """
    preview_update(graph, request)
    assert _snapshot(graph) == _snapshot(_seed(_make_graph(factory)))
