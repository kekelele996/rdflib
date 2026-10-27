"""Tests for atomic, idempotent named graph imports (`rdflib.bulk`)."""

from __future__ import annotations

from dataclasses import replace as replace_dataclass

import pytest

from rdflib import BNode, Dataset, Graph, Literal, URIRef
from rdflib.bulk import (
    BulkImportError,
    StagedImport,
    bulk_import,
    inspect_import,
    stage_import,
)

EX = "http://example.org/"
G1 = URIRef(EX + "g1")
G2 = URIRef(EX + "g2")
A = URIRef(EX + "a")
B = URIRef(EX + "b")
OLD = URIRef(EX + "old")
P = URIRef(EX + "p")
Q = URIRef(EX + "q")
R = URIRef(EX + "r")

TRIG = """\
@prefix ex: <http://example.org/> .
ex:g1 { ex:a ex:p [ ex:label "node1" ] . }
ex:g2 { ex:b ex:q "y" . }
{ ex:d ex:t "default triple" . }
"""

TRIG_V2 = """\
@prefix ex: <http://example.org/> .
ex:g1 { ex:a ex:p [ ex:label "node1" ] . ex:a ex:p2 "extra" . }
ex:g2 { ex:b ex:q "y" . }
{ ex:d ex:t "default triple" . }
"""


def _quad_set(dataset: Dataset) -> set:
    return set(dataset.quads((None, None, None, None)))


def _role_labels(staged: StagedImport) -> dict[str, str]:
    """Map a blank node's ``ex:label`` literal to its remapped label."""

    mapping = staged._bnode_mapping
    roles = {}
    for s, _p, o, _g in staged._raw_quads:
        if str(o) in ("node1", "node2", "node3"):
            roles[str(o)] = str(mapping[s])
    return roles


class TestInspection:
    def test_reports_graphs_statements_and_bnodes(self):
        target = Dataset()
        target.get_context(G2).add((OLD, R, Literal("gone")))

        check = inspect_import(target, data=TRIG, format="trig")

        assert check.ok
        assert check.raw_statement_count == 4
        assert check.statement_count == 4
        by_id = {graph.identifier: graph for graph in check.graphs}
        assert set(by_id) == {G1, G2, None}
        assert by_id[G1].exists is False
        assert by_id[G1].statement_count == 2
        assert by_id[G2].exists is True
        assert by_id[G2].statement_count == 1
        assert by_id[G2].existing_statement_count == 1
        assert by_id[None].is_default is True
        assert by_id[None].statement_count == 1
        assert check.bnodes.count == 1
        assert all(
            new_label.startswith("bulk") for new_label in check.bnodes.mapping.values()
        )
        assert check.bnodes.collisions == frozenset()
        # staging must not mutate the target
        assert len(target) == 1

    def test_collision_is_reported_when_label_exists_in_target(self):
        target = Dataset()
        target.get_context(G1).add((BNode("shared"), Q, Literal("old")))
        nquads = f"<{A}> <{P}> _:shared <{G1}> .\n"

        staged = stage_import(
            target,
            data=nquads,
            format="nquads",
            bnode_context={"shared": BNode("shared")},
        )
        assert staged.check.bnodes.collisions == frozenset({"shared"})

        with pytest.raises(BulkImportError, match="already exist"):
            staged.commit(bnode_policy="keep")
        assert _quad_set(target) == {(BNode("shared"), Q, Literal("old"), G1)}

    def test_target_must_be_dataset(self):
        with pytest.raises(TypeError, match="Dataset"):
            stage_import(Graph(), data=f'<{A}> <{P}> "x" .', format="turtle")

    def test_unknown_bnode_policy_rejected(self):
        staged = stage_import(Dataset(), data=TRIG, format="trig")
        with pytest.raises(BulkImportError, match="blank node policy"):
            staged.commit(bnode_policy="whatever")  # type: ignore[arg-type]

    def test_check_errors_block_commit(self):
        target = Dataset()
        staged = stage_import(target, data=TRIG, format="trig")
        object.__setattr__(
            staged,
            "check",
            replace_dataclass(staged.check, errors=("synthetic error",)),
        )
        with pytest.raises(BulkImportError, match="synthetic error"):
            staged.commit()
        assert len(target) == 0


class TestAtomicity:
    def test_parse_error_leaves_target_untouched(self):
        target = Dataset()
        target.get_context(G1).add((A, P, Literal("keep")))
        snapshot = _quad_set(target)
        broken = (
            "@prefix ex: <http://example.org/> .\n"
            'ex:g1 { ex:a ex:p "ok" . }\n'
            "ex:g2 { ex:b ex:q ."  # truncated statement
        )
        with pytest.raises(BulkImportError, match="parse"):
            bulk_import(target, data=broken, format="trig")
        assert _quad_set(target) == snapshot
        assert G2 not in {c.identifier for c in target.store.contexts()}

    def test_write_failure_rolls_everything_back(self):
        target = Dataset()
        target.get_context(G2).add((OLD, R, Literal("gone")))
        snapshot = _quad_set(target)

        real_add_n = target.store.addN

        def exploding_add_n(quads):
            for quad in quads:
                if quad[0] == B:
                    raise RuntimeError("simulated store failure")
                real_add_n(iter([quad]))

        target.store.addN = exploding_add_n
        try:
            with pytest.raises(BulkImportError, match="rolled back"):
                bulk_import(target, data=TRIG, format="trig")
        finally:
            target.store.addN = real_add_n

        assert _quad_set(target) == snapshot
        # the removed statement was restored
        assert (OLD, R, Literal("gone"), G2) in _quad_set(target)
        # the half-created graph was deregistered
        assert G1 not in {c.identifier for c in target.store.contexts()}

    def test_transaction_aware_store_uses_native_rollback(self):
        target = Dataset()
        target.get_context(G1).add((A, P, Literal("keep")))

        target.store.transaction_aware = True
        real_add_n = target.store.addN

        def exploding_add_n(quads):
            for quad in quads:
                if quad[0] == B:
                    raise RuntimeError("simulated transaction failure")
                real_add_n(iter([quad]))

        rollback_calls = []
        target.store.addN = exploding_add_n
        target.store.rollback = lambda: rollback_calls.append(True)  # type: ignore[method-assign]
        target.store.commit = lambda: None  # type: ignore[method-assign]
        try:
            with pytest.raises(BulkImportError):
                bulk_import(target, data=TRIG, format="trig")
        finally:
            target.store.transaction_aware = False
        assert rollback_calls == [True]


class TestCommit:
    def test_replaces_graph_and_lists_added_and_replaced(self):
        target = Dataset()
        target.get_context(G2).add((OLD, R, Literal("gone")))

        report = bulk_import(target, data=TRIG, format="trig")

        assert isinstance(report.changes, tuple)
        g2_change = next(change for change in report.changes if change.identifier == G2)
        assert g2_change.created is False
        assert g2_change.replaced == frozenset({(OLD, R, Literal("gone"))})
        assert g2_change.added == frozenset({(B, Q, Literal("y"))})
        assert g2_change.unchanged == 0
        g1_change = next(change for change in report.changes if change.identifier == G1)
        assert g1_change.created is True
        assert len(g1_change.added) == 2
        assert report.total_added == 4
        assert report.total_replaced == 1

        # the target content matches the report exactly
        assert (OLD, R, Literal("gone")) not in target.get_context(G2)
        assert (B, Q, Literal("y")) in target.get_context(G2)

    def test_merge_mode_never_removes(self):
        target = Dataset()
        target.get_context(G2).add((OLD, R, Literal("gone")))

        report = bulk_import(target, data=TRIG, format="trig", replace=False)

        assert report.total_replaced == 0
        assert (OLD, R, Literal("gone")) in target.get_context(G2)
        assert (B, Q, Literal("y")) in target.get_context(G2)

        follow_up = bulk_import(target, data=TRIG_V2, format="trig", replace=False)
        assert follow_up.total_replaced == 0
        assert (A, URIRef(EX + "p2"), Literal("extra")) in target.get_context(G1)
        assert (OLD, R, Literal("gone")) in target.get_context(G2)

    def test_rerun_is_idempotent(self):
        target = Dataset()
        first = bulk_import(target, data=TRIG, format="trig")
        assert first.is_noop is False
        state_after_first = _quad_set(target)

        second = bulk_import(target, data=TRIG, format="trig")
        assert second.is_noop is True
        assert second.total_added == 0
        assert second.total_replaced == 0
        assert _quad_set(target) == state_after_first

        third = bulk_import(target, data=TRIG, format="trig")
        assert third.is_noop is True

    def test_rerun_is_idempotent_after_content_change(self):
        target = Dataset()
        bulk_import(target, data=TRIG, format="trig")
        bulk_import(target, data=TRIG_V2, format="trig")
        state = _quad_set(target)

        again = bulk_import(target, data=TRIG_V2, format="trig")
        assert again.is_noop
        assert _quad_set(target) == state

    def test_dry_run_does_not_modify_target(self):
        target = Dataset()
        target.get_context(G2).add((OLD, R, Literal("gone")))
        snapshot = _quad_set(target)

        staged = stage_import(target, data=TRIG, format="trig")
        dry = staged.commit(dry_run=True)
        assert dry.dry_run is True
        assert dry.total_added == 4
        assert dry.total_replaced == 1
        assert _quad_set(target) == snapshot

        real = staged.commit()
        assert real.dry_run is False
        assert real.total_added == dry.total_added
        assert real.total_replaced == dry.total_replaced

    def test_non_quad_format_goes_to_default_graph(self):
        target = Dataset()
        report = bulk_import(target, data=f'<{A}> <{P}> "x" .', format="turtle")
        (change,) = report.changes
        assert change.identifier is None
        assert (A, P, Literal("x")) in target.default_graph


class TestBlankNodes:
    def test_labels_are_deterministic_across_parses(self):
        data = """\
@prefix ex: <http://example.org/> .
ex:g1 { ex:a ex:p [ ex:label "node1" ] , [ ex:label "node2" ] . }
ex:g2 { ex:b ex:q [ ex:label "node3" ] . }
"""
        first = _role_labels(stage_import(Dataset(), data=data, format="trig"))
        for _ in range(5):
            assert (
                _role_labels(stage_import(Dataset(), data=data, format="trig")) == first
            )

    def test_labels_distinguish_graph_and_structure(self):
        data = """\
@prefix ex: <http://example.org/> .
ex:g1 { ex:a ex:p [ ex:label "same" ] , [ ex:label "same" ] . }
ex:g2 { ex:b ex:q [ ex:label "same" ] . }
"""
        staged = stage_import(Dataset(), data=data, format="trig")
        labels = list(staged.check.bnodes.mapping.values())
        # symmetric twins stay distinct, graphs stay distinct
        assert len(set(labels)) == 3
        assert all(label.startswith("bulk") and len(label) > 40 for label in labels)

    def test_canonical_policy_avoids_existing_labels(self):
        target = Dataset()
        target.get_context(G1).add((BNode("shared"), Q, Literal("old")))
        nquads = f"<{A}> <{P}> _:shared <{G1}> .\n"

        report = bulk_import(
            target,
            data=nquads,
            format="nquads",
            replace=False,
            bnode_context={"shared": BNode("shared")},
        )
        assert report.total_added == 1
        labels = {
            str(term)
            for s, _p, o, _g in target.quads()
            for term in (s, o)
            if isinstance(term, BNode)
        }
        assert "shared" in labels  # pre-existing node untouched
        assert any(label.startswith("bulk") for label in labels)  # import remapped
        # both statements coexist, unrelated nodes were not merged
        assert len(target.get_context(G1)) == 2

    def test_bnode_named_graph_is_idempotent(self):
        data = """\
@prefix ex: <http://example.org/> .
graph _:gname { ex:a ex:p "x" . }
"""
        target = Dataset()
        first = bulk_import(target, data=data, format="trig")
        assert first.total_added == 1
        (graph_id,) = (
            change.identifier
            for change in first.changes
            if change.identifier is not None
        )
        assert str(graph_id).startswith("bulkgn")

        second = bulk_import(target, data=data, format="trig")
        assert second.is_noop
        named_graphs = [
            c.identifier
            for c in target.store.contexts()
            if str(c.identifier).startswith("bulkgn")
        ]
        assert len(named_graphs) == 1
