"""Tests for named-graph level permissions on a shared dataset."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, cast

import pytest

import rdflib.plugins.sparql as sparql_module
from rdflib import Literal, Node, URIRef
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, BatchAddGraph
from rdflib.security import (
    ANY_GRAPH,
    AccessDeniedError,
    GraphAccessPolicy,
    IdentityRequiredError,
    Permission,
    SecuredDataset,
    principal,
)

EX = "http://example.org/"
GA = URIRef(EX + "team-a")
GB = URIRef(EX + "team-b")
GS = URIRef(EX + "shared")
GN = URIRef(EX + "brand-new")

S = URIRef(EX + "s")
P = URIRef(EX + "p")


@pytest.fixture(autouse=True)
def _dataset_semantics(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Standard RDF dataset semantics for SPARQL (the W3C test harness uses
    # the same setting when evaluating datasets).
    monkeypatch.setattr(sparql_module, "SPARQL_DEFAULT_GRAPH_UNION", False)
    yield


@pytest.fixture
def dataset() -> SecuredDataset:
    policy = GraphAccessPolicy()
    # Deployment owner: full access to every graph, used to seed and verify.
    policy.grant(
        "owner",
        ANY_GRAPH,
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
    )
    # Team A owns the default graph and GA; has read-only access to GS.
    for graph in (DATASET_DEFAULT_GRAPH_ID, GA):
        policy.grant(
            "team-a",
            graph,
            Permission.READ,
            Permission.WRITE,
            Permission.DELETE,
        )
    policy.grant("team-a", GS, Permission.READ)
    # Team B owns GB; can read and write (but not delete) GS.
    policy.grant(
        "team-b",
        GB,
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
    )
    policy.grant("team-b", GS, Permission.READ, Permission.WRITE)
    # Auditor: read-only across every graph.
    policy.grant("auditor", ANY_GRAPH, Permission.READ)

    dataset = SecuredDataset(policy)
    with principal("owner"):
        add_quad(dataset, S, P, Literal("in default"))
        add_quad(dataset, S, P, Literal("in a"), GA)
        add_quad(dataset, S, P, Literal("in b"), GB)
        add_quad(dataset, S, P, Literal("in shared"), GS)
    return dataset


def add_quad(
    dataset: SecuredDataset,
    subject: Node,
    predicate: Node,
    object_: Node,
    graph: Node | None = None,
) -> None:
    """Add a quad, naming a graph by identifier as the public API allows."""
    if graph is None:
        dataset.add((subject, predicate, object_))
    else:
        dataset.add((subject, predicate, object_, graph))  # type: ignore[arg-type]


def graph_sizes(dataset: SecuredDataset) -> dict[str, int]:
    with principal("owner"):
        return {
            "default": len(dataset.graph(DATASET_DEFAULT_GRAPH_ID)),
            "a": len(dataset.graph(GA)),
            "b": len(dataset.graph(GB)),
            "shared": len(dataset.graph(GS)),
        }


def visible_graphs(dataset: SecuredDataset, identity: str) -> set[str]:
    with principal(identity):
        return {str(graph.identifier) for graph in dataset.graphs()}


def visible_quad_graphs(dataset: SecuredDataset, identity: str) -> set[str]:
    with principal(identity):
        # Dataset.quads() reports default graph quads with a None context.
        return {
            str(context) if context is not None else str(DATASET_DEFAULT_GRAPH_ID)
            for _s, _p, _o, context in dataset.quads()
        }


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identity_required(dataset: SecuredDataset) -> None:
    with pytest.raises(IdentityRequiredError):
        dataset.add((S, P, Literal("x")))
    with pytest.raises(IdentityRequiredError):
        list(dataset.quads())


def test_fixed_identity() -> None:
    policy = GraphAccessPolicy()
    policy.grant(
        "svc",
        DATASET_DEFAULT_GRAPH_ID,
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
    )
    dataset = SecuredDataset(policy, identity="svc")
    dataset.add((S, P, Literal("x")))
    assert len(dataset) == 1


def test_principal_contexts_are_isolated(dataset: SecuredDataset) -> None:
    with principal("team-a"):
        assert str(GB) not in visible_graphs(dataset, "team-a")
        with principal("owner"):
            assert str(GB) in visible_graphs(dataset, "owner")
        assert str(GB) not in visible_graphs(dataset, "team-a")


# ---------------------------------------------------------------------------
# Read access
# ---------------------------------------------------------------------------


def test_unscoped_reads_are_filtered(dataset: SecuredDataset) -> None:
    assert visible_quad_graphs(dataset, "team-a") == {
        str(DATASET_DEFAULT_GRAPH_ID),
        str(GA),
        str(GS),
    }
    assert visible_quad_graphs(dataset, "team-b") == {str(GB), str(GS)}
    # Auditor sees all graphs through the wildcard grant.
    assert visible_quad_graphs(dataset, "auditor") == {
        str(DATASET_DEFAULT_GRAPH_ID),
        str(GA),
        str(GB),
        str(GS),
    }


def test_graph_listing_is_filtered(dataset: SecuredDataset) -> None:
    assert visible_graphs(dataset, "team-a") == {
        str(DATASET_DEFAULT_GRAPH_ID),
        str(GA),
        str(GS),
    }
    assert visible_graphs(dataset, "team-b") == {str(GB), str(GS)}
    # Default graph is not force-disclosed to a principal without READ on it.
    assert str(DATASET_DEFAULT_GRAPH_ID) not in visible_graphs(dataset, "team-b")


def test_explicit_read_is_refused_with_details(dataset: SecuredDataset) -> None:
    with principal("team-a"):
        with pytest.raises(AccessDeniedError) as exc_info:
            list(dataset.graph(GB).triples((None, None, None)))
        error = exc_info.value
        assert error.principal == "team-a"
        assert error.graph == GB
        assert error.permission is Permission.READ
        assert "READ" in str(error)
        assert GB.n3() in str(error)
        # Other explicit read entry points refuse identically.
        with pytest.raises(AccessDeniedError):
            len(dataset.graph(GB))
        with pytest.raises(AccessDeniedError):
            (S, P, Literal("in b")) in dataset.graph(GB)
        with pytest.raises(AccessDeniedError):
            list(dataset.quads((None, None, None, GB)))  # type: ignore[arg-type]


def test_allowed_reads_succeed(dataset: SecuredDataset) -> None:
    with principal("team-a"):
        assert (S, P, Literal("in a")) in dataset.graph(GA)
        values = [str(object_) for _s, _p, object_ in dataset.graph(GA)]
        assert values == ["in a"]


def test_dataset_serialization_filters_forbidden_graphs(
    dataset: SecuredDataset,
) -> None:
    with principal("team-a"):
        serialised = dataset.serialize(format="trig")
    assert str(GB) not in serialised
    assert str(GA) in serialised


def test_forbidden_graph_serialization_is_refused(dataset: SecuredDataset) -> None:
    with principal("team-a"), pytest.raises(AccessDeniedError) as exc_info:
        dataset.graph(GB).serialize()
    assert exc_info.value.permission is Permission.READ


# ---------------------------------------------------------------------------
# Write access
# ---------------------------------------------------------------------------


def test_write_to_own_graph(dataset: SecuredDataset) -> None:
    with principal("team-a"):
        add_quad(dataset, S, P, Literal("new"), GA)
    assert graph_sizes(dataset)["a"] == 2


def test_write_to_foreign_graph_refused_without_changes(
    dataset: SecuredDataset,
) -> None:
    before = graph_sizes(dataset)
    with principal("team-a"):
        with pytest.raises(AccessDeniedError) as exc_info:
            add_quad(dataset, S, P, Literal("new"), GB)
        assert exc_info.value.permission is Permission.WRITE
        assert exc_info.value.graph == GB
    assert graph_sizes(dataset) == before


def test_write_permission_does_not_grant_delete(dataset: SecuredDataset) -> None:
    # Team B may write GS but not delete from it.
    with principal("team-b"):
        add_quad(dataset, S, P, Literal("added"), GS)
        with pytest.raises(AccessDeniedError) as exc_info:
            dataset.graph(GS).remove((None, None, None))
        assert exc_info.value.permission is Permission.DELETE
        assert exc_info.value.graph == GS
    assert graph_sizes(dataset)["shared"] == 2


def test_bulk_write_is_authorised_before_anything_is_added(
    dataset: SecuredDataset,
) -> None:
    before = graph_sizes(dataset)
    with principal("team-a"):
        with pytest.raises(AccessDeniedError):
            dataset.addN(
                [
                    (S, P, Literal("ok"), dataset.graph(GA)),
                    (S, P, Literal("not ok"), dataset.graph(GB)),
                ]
            )
    assert graph_sizes(dataset) == before


def test_batch_add_graph_cannot_smuggle_quads_in(dataset: SecuredDataset) -> None:
    before = graph_sizes(dataset)
    with principal("team-a"):
        with pytest.raises(AccessDeniedError), BatchAddGraph(dataset) as batch:
            batch.add((S, P, Literal("ok"), dataset.graph(GA)))
            batch.add((S, P, Literal("not ok"), dataset.graph(GB)))
    assert graph_sizes(dataset) == before


def test_creating_a_graph_requires_write(dataset: SecuredDataset) -> None:
    before = graph_sizes(dataset)
    with principal("team-a"):
        with pytest.raises(AccessDeniedError) as exc_info:
            dataset.graph(GN)
        assert exc_info.value.permission is Permission.WRITE
    assert graph_sizes(dataset) == before


def test_revoked_grant_no_longer_applies(dataset: SecuredDataset) -> None:
    dataset.policy.revoke("team-a", GA, Permission.WRITE)
    with principal("team-a"):
        with pytest.raises(AccessDeniedError):
            add_quad(dataset, S, P, Literal("x"), GA)


# ---------------------------------------------------------------------------
# Delete access
# ---------------------------------------------------------------------------


def test_remove_graph_requires_delete(dataset: SecuredDataset) -> None:
    before = graph_sizes(dataset)
    with principal("team-a"):
        with pytest.raises(AccessDeniedError) as exc_info:
            dataset.remove_graph(GB)
        assert exc_info.value.permission is Permission.DELETE
        assert exc_info.value.graph == GB
    assert graph_sizes(dataset) == before
    with principal("team-b"):
        dataset.remove_graph(GB)
    assert graph_sizes(dataset)["b"] == 0


def test_clear_graph_requires_delete(dataset: SecuredDataset) -> None:
    before = graph_sizes(dataset)
    with principal("team-b"):
        # GS is writable but not deletable.
        with pytest.raises(AccessDeniedError):
            dataset.graph(GS).remove((None, None, None))
    assert graph_sizes(dataset) == before
    with principal("team-a"):
        dataset.graph(GA).remove((None, None, None))
    assert graph_sizes(dataset)["a"] == 0
    assert graph_sizes(dataset)["b"] == before["b"]


def test_unscoped_remove_requires_delete_on_every_graph(
    dataset: SecuredDataset,
) -> None:
    before = graph_sizes(dataset)
    with principal("team-a"):
        with pytest.raises(AccessDeniedError) as exc_info:
            # A 3-tuple remove on a dataset hits every named graph.
            dataset.remove((None, None, None))  # type: ignore[arg-type]
        assert exc_info.value.permission is Permission.DELETE
        # Team A may not delete either of the other teams' graphs.
        assert exc_info.value.graph in (GB, GS)
    assert graph_sizes(dataset) == before


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_quad_parse_is_atomic_when_one_graph_is_forbidden(
    dataset: SecuredDataset,
) -> None:
    before = graph_sizes(dataset)
    trig = (
        f"GRAPH <{GA}> {{ <{S}> <{P}> 'parsed a' }}\n"
        f"GRAPH <{GB}> {{ <{S}> <{P}> 'parsed b' }}\n"
    )
    with principal("team-a"):
        with pytest.raises(AccessDeniedError) as exc_info:
            dataset.parse(data=trig, format="trig")
        assert exc_info.value.graph == GB
        assert exc_info.value.permission is Permission.WRITE
    assert graph_sizes(dataset) == before


def test_quad_parse_to_allowed_graphs_succeeds(dataset: SecuredDataset) -> None:
    nquads = f'<{S}> <{P}> "parsed a" <{GA}> .\n'
    with principal("team-a"):
        dataset.parse(data=nquads, format="nquads")
    assert graph_sizes(dataset)["a"] == 2


def test_triple_parse_to_default_graph(dataset: SecuredDataset) -> None:
    with principal("team-a"):
        dataset.parse(data=f"<{S}> <{P}> 'parsed default' .", format="turtle")
    assert graph_sizes(dataset)["default"] == 2


# ---------------------------------------------------------------------------
# SPARQL queries
# ---------------------------------------------------------------------------


def test_sparql_explicit_graph_read_refused(dataset: SecuredDataset) -> None:
    query = f"SELECT ?o WHERE {{ GRAPH <{GB}> {{ ?s ?p ?o }} }}"
    with principal("team-a"), pytest.raises(AccessDeniedError) as exc_info:
        list(dataset.query(query))
    assert exc_info.value.graph == GB
    assert exc_info.value.permission is Permission.READ


def test_sparql_from_named_refused(dataset: SecuredDataset) -> None:
    query = f"SELECT ?o FROM NAMED <{GB}> WHERE {{ GRAPH <{GB}> {{ ?s ?p ?o }} }}"
    with principal("team-a"), pytest.raises(AccessDeniedError):
        list(dataset.query(query))


def test_sparql_graph_variable_only_visits_readable_graphs(
    dataset: SecuredDataset,
) -> None:
    query = "SELECT ?g ?o WHERE { GRAPH ?g { ?s ?p ?o } }"
    with principal("team-a"):
        graphs = {
            str(cast(Mapping[str, Any], binding)["g"])
            for binding in dataset.query(query)
        }
    assert graphs == {str(GA), str(GS)}


def test_sparql_default_graph_query_without_read(dataset: SecuredDataset) -> None:
    dataset.policy.grant("limited", GA, Permission.READ)
    query = "SELECT ?o WHERE { ?s ?p ?o }"
    with principal("limited"), pytest.raises(AccessDeniedError) as exc_info:
        list(dataset.query(query))
    assert exc_info.value.graph == DATASET_DEFAULT_GRAPH_ID
    assert exc_info.value.permission is Permission.READ


def test_sparql_allowed_query(dataset: SecuredDataset) -> None:
    query = f"SELECT ?o WHERE {{ GRAPH <{GA}> {{ ?s ?p ?o }} }}"
    with principal("team-a"):
        objects = [
            str(cast(Mapping[str, Any], binding)["o"])
            for binding in dataset.query(query)
        ]
    assert objects == ["in a"]


# ---------------------------------------------------------------------------
# SPARQL updates
# ---------------------------------------------------------------------------


def test_sparql_insert_data_refused_atomically(dataset: SecuredDataset) -> None:
    before = graph_sizes(dataset)
    update = (
        f"INSERT DATA {{ "
        f"<{S}> <{P}> 'default leak' . "
        f"GRAPH <{GB}> {{ <{S}> <{P}> 'b leak' }} }}"
    )
    with principal("team-a"), pytest.raises(AccessDeniedError) as exc_info:
        dataset.update(update)
    assert exc_info.value.graph == GB
    assert exc_info.value.permission is Permission.WRITE
    assert graph_sizes(dataset) == before


def test_sparql_multi_request_refused_before_first_op_runs(
    dataset: SecuredDataset,
) -> None:
    before = graph_sizes(dataset)
    update = (
        f"INSERT DATA {{ <{S}> <{P}> 'first ok' }}; "
        f"INSERT DATA {{ GRAPH <{GB}> {{ <{S}> <{P}> 'second bad' }} }}"
    )
    with principal("team-a"), pytest.raises(AccessDeniedError):
        dataset.update(update)
    assert graph_sizes(dataset) == before


def test_sparql_delete_data_refused_atomically(dataset: SecuredDataset) -> None:
    before = graph_sizes(dataset)
    update = f"DELETE DATA {{ GRAPH <{GB}> {{ <{S}> <{P}> 'in b' }} }}"
    with principal("team-a"), pytest.raises(AccessDeniedError) as exc_info:
        dataset.update(update)
    assert exc_info.value.permission is Permission.DELETE
    assert graph_sizes(dataset) == before


def test_sparql_clear_graph_refused(dataset: SecuredDataset) -> None:
    before = graph_sizes(dataset)
    with principal("team-a"), pytest.raises(AccessDeniedError) as exc_info:
        dataset.update(f"CLEAR GRAPH <{GB}>")
    assert exc_info.value.permission is Permission.DELETE
    assert graph_sizes(dataset) == before


def test_sparql_clear_all_refused_and_leaves_everything(
    dataset: SecuredDataset,
) -> None:
    before = graph_sizes(dataset)
    with principal("team-a"), pytest.raises(AccessDeniedError) as exc_info:
        dataset.update("CLEAR ALL")
    assert exc_info.value.permission is Permission.DELETE
    assert graph_sizes(dataset) == before
    with principal("team-a"), pytest.raises(AccessDeniedError):
        dataset.update("DROP NAMED")
    assert graph_sizes(dataset) == before


def test_silent_clause_does_not_swallow_security_errors(
    dataset: SecuredDataset,
) -> None:
    before = graph_sizes(dataset)
    with principal("team-a"), pytest.raises(AccessDeniedError):
        dataset.update(f"DROP SILENT GRAPH <{GB}>")
    assert graph_sizes(dataset) == before


def test_sparql_dynamic_graph_target_refused_atomically(
    dataset: SecuredDataset,
) -> None:
    # GRAPH ?g visits GA (writable) and GS (read-only for team-a); the
    # dynamic target cannot be statically resolved, so the refusal is found
    # at run time and the whole request is rolled back.
    before = graph_sizes(dataset)
    update = "INSERT { GRAPH ?g { ?s ?p 'copied' } } WHERE { GRAPH ?g { ?s ?p ?o } }"
    with principal("team-a"), pytest.raises(AccessDeniedError) as exc_info:
        dataset.update(update)
    assert exc_info.value.graph == GS
    assert exc_info.value.permission is Permission.WRITE
    assert graph_sizes(dataset) == before


def test_sparql_with_clause_delete_refused(dataset: SecuredDataset) -> None:
    before = graph_sizes(dataset)
    update = f"WITH <{GB}> DELETE {{ ?s ?p ?o }} WHERE {{ ?s ?p ?o }}"
    with principal("team-a"), pytest.raises(AccessDeniedError) as exc_info:
        dataset.update(update)
    assert exc_info.value.graph == GB
    assert exc_info.value.permission is Permission.READ
    assert graph_sizes(dataset) == before


def test_sparql_allowed_insert_and_delete(dataset: SecuredDataset) -> None:
    with principal("team-a"):
        dataset.update(f"INSERT DATA {{ GRAPH <{GA}> {{ <{S}> <{P}> 'new a' }} }}")
        assert graph_sizes(dataset)["a"] == 2
        dataset.update(f"DELETE DATA {{ GRAPH <{GA}> {{ <{S}> <{P}> 'in a' }} }}")
    assert graph_sizes(dataset)["a"] == 1


def test_sparql_modify_where_with_named_graph(dataset: SecuredDataset) -> None:
    # Team B may read the default graph and GB, but not GA; a DELETE whose
    # WHERE clause reads GA must be refused.
    dataset.policy.grant("team-b", DATASET_DEFAULT_GRAPH_ID, Permission.READ)
    update = f"DELETE {{ ?s ?p ?o }} WHERE {{ GRAPH <{GA}> {{ ?s ?p ?o }} }}"
    with principal("team-b"), pytest.raises(AccessDeniedError) as exc_info:
        dataset.update(update)
    assert exc_info.value.graph == GA
    assert exc_info.value.permission is Permission.READ
    assert graph_sizes(dataset)["a"] == 1
