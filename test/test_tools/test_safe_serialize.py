from __future__ import annotations

import errno
import io
import logging
import os
import stat
from pathlib import Path

import pytest

from rdflib import BNode, Dataset, Graph, Literal, Namespace
from rdflib.exceptions import SerializationVerificationError
from rdflib.plugins.serializers.nquads import NQuadsSerializer, _nq_row
from rdflib.plugins.serializers.nt import NTSerializer, _nt_row
from rdflib.tools.safe_serialize import PublishResult, safe_serialize

EX = Namespace("http://example.org/")

OLD_CONTENT = b"# the previous good daily export\n"


def _make_graph() -> Graph:
    g = Graph()
    g.add((EX.a, EX.label, Literal("a")))
    g.add((EX.b, EX.label, Literal("b")))
    g.add((EX.b, EX.seeAlso, BNode()))
    return g


def _make_dataset() -> Dataset:
    ds = Dataset()
    ds.add((EX.root, EX.label, Literal("default graph")))
    g1 = ds.graph(EX.g1)
    g1.add((EX.x, EX.label, Literal("one")))
    g1.add((EX.y, EX.seeAlso, BNode()))
    g2 = ds.graph(EX.g2)
    g2.add((EX.z, EX.label, Literal("two")))
    return ds


@pytest.mark.parametrize("format", ["turtle", "nt", "xml"])
def test_safe_serialize_graph_replaces_target(
    tmp_path: Path, format: str, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / f"out.{format if format != 'xml' else 'rdf'}"
    target.write_bytes(OLD_CONTENT)
    graph = _make_graph()

    caplog.set_level(logging.INFO, logger="rdflib.tools.safe_serialize")
    result = safe_serialize(graph, target, format=format)

    assert isinstance(result, PublishResult)
    assert result.destination == target.resolve()
    assert result.format == format
    assert result.statements == 3
    assert result.named_graphs == ()
    assert result.bytes_written == target.stat().st_size
    assert result.bytes_written > 0

    published = target.read_bytes()
    assert published != OLD_CONTENT
    recovered = Graph()
    recovered.parse(target, format=format)
    assert graph.isomorphic(recovered)

    # the side file is gone, only the published file remains
    assert os.listdir(tmp_path) == [target.name]

    summary = result.summary()
    assert format in summary
    assert "3 statement(s)" in summary
    assert str(target) in summary
    assert "published" in caplog.text
    assert summary in caplog.text


@pytest.mark.parametrize("format", ["trig", "nquads"])
def test_safe_serialize_dataset_verifies_named_graphs(
    tmp_path: Path, format: str
) -> None:
    target = tmp_path / f"out.{'trig' if format == 'trig' else 'nq'}"
    target.write_bytes(OLD_CONTENT)
    dataset = _make_dataset()

    result = safe_serialize(dataset, target, format=format)

    assert result.format == format
    # one default graph statement + two in g1 + one in g2
    assert result.statements == 4
    assert set(result.named_graphs) == {EX.g1, EX.g2}
    assert "2 named graph(s)" in result.summary()

    recovered = Dataset()
    recovered.parse(target, format=format)
    assert len(recovered) == 4
    assert {graph.identifier for graph in recovered.graphs() if len(graph) > 0} == {
        EX.g1,
        EX.g2,
        recovered.default_graph.identifier,
    }
    assert os.listdir(tmp_path) == [target.name]


def test_safe_serialize_bnode_named_graph_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "out.nq"
    dataset = Dataset()
    graph_name = BNode()
    named = dataset.graph(graph_name)
    named.add((EX.q, EX.label, Literal("in a blank-node named graph")))

    result = safe_serialize(dataset, target, format="nquads")

    assert result.statements == 1
    assert len(result.named_graphs) == 1
    assert isinstance(result.named_graphs[0], BNode)
    recovered = Dataset()
    recovered.parse(target, format="nquads")
    assert len(recovered) == 1


def test_format_is_guessed_from_suffix(tmp_path: Path) -> None:
    graph = _make_graph()
    assert safe_serialize(graph, tmp_path / "out.nt").format == "nt"
    assert safe_serialize(graph, tmp_path / "out.ttl").format == "turtle"
    assert safe_serialize(graph, tmp_path / "out.rdf").format == "xml"

    dataset = _make_dataset()
    assert safe_serialize(dataset, tmp_path / "ds.nq").format == "nquads"
    assert safe_serialize(dataset, tmp_path / "ds.trig").format == "trig"
    # no recognizable suffix: per-kind defaults
    assert safe_serialize(graph, tmp_path / "plain").format == "turtle"
    assert safe_serialize(dataset, tmp_path / "quads").format == "trig"


def test_empty_graph_is_published_and_verified(tmp_path: Path) -> None:
    target = tmp_path / "empty.nt"
    result = safe_serialize(Graph(), target, format="nt")
    assert result.statements == 0
    assert Graph().parse(target, format="nt") is not None
    assert os.listdir(tmp_path) == [target.name]


def test_write_error_preserves_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.nt"
    target.write_bytes(OLD_CONTENT)

    def raise_no_space(self, stream, base=None, encoding=None, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(NTSerializer, "serialize", raise_no_space)

    with pytest.raises(SerializationVerificationError) as exc_info:
        safe_serialize(_make_graph(), target, format="nt")
    error = exc_info.value
    assert error.stage == "write"
    assert isinstance(error.__cause__, OSError)
    assert error.__cause__.errno == errno.ENOSPC

    # the old file is byte-for-byte intact and no side file is left behind
    assert target.read_bytes() == OLD_CONTENT
    assert os.listdir(tmp_path) == [target.name]


def test_write_error_without_existing_target_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.nt"

    def raise_no_space(self, stream, base=None, encoding=None, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(NTSerializer, "serialize", raise_no_space)

    with pytest.raises(SerializationVerificationError) as exc_info:
        safe_serialize(_make_graph(), target, format="nt")
    assert exc_info.value.stage == "write"
    assert not target.exists()
    assert os.listdir(tmp_path) == []


def test_truncated_output_fails_verification_and_keeps_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.nt"
    target.write_bytes(OLD_CONTENT)

    def serialize_only_first_triple(self, stream, base=None, encoding=None, **kwargs):
        first_triple = next(iter(self.store))
        stream.write(_nt_row(first_triple).encode("utf-8"))

    monkeypatch.setattr(NTSerializer, "serialize", serialize_only_first_triple)

    with pytest.raises(SerializationVerificationError) as exc_info:
        safe_serialize(_make_graph(), target, format="nt")
    error = exc_info.value
    assert error.stage == "verify"
    assert "3 statement(s)" in str(error)
    assert "1" in str(error)

    assert target.read_bytes() == OLD_CONTENT
    assert os.listdir(tmp_path) == [target.name]


def test_unparseable_output_fails_readback_and_keeps_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.nt"
    target.write_bytes(OLD_CONTENT)

    def serialize_garbage(self, stream, base=None, encoding=None, **kwargs):
        stream.write(b"this is definitely not n-triples <<<\x00\n")

    monkeypatch.setattr(NTSerializer, "serialize", serialize_garbage)

    with pytest.raises(SerializationVerificationError) as exc_info:
        safe_serialize(_make_graph(), target, format="nt")
    assert exc_info.value.stage == "readback"
    assert target.read_bytes() == OLD_CONTENT
    assert os.listdir(tmp_path) == [target.name]


def test_dropped_named_graph_fails_verification_and_keeps_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.nq"
    target.write_bytes(OLD_CONTENT)

    def serialize_without_g2(self, stream, base=None, encoding=None, **kwargs):
        for context in self.store.contexts():
            if context.identifier == EX.g2:
                continue
            for triple in context:
                stream.write(_nq_row(triple, context.identifier).encode("utf-8"))
        stream.write(b"\n")

    monkeypatch.setattr(NQuadsSerializer, "serialize", serialize_without_g2)

    with pytest.raises(SerializationVerificationError) as exc_info:
        safe_serialize(_make_dataset(), target, format="nquads")
    error = exc_info.value
    assert error.stage == "verify"
    assert str(EX.g2) in str(error)
    assert "missing named graph" in str(error)

    assert target.read_bytes() == OLD_CONTENT
    assert os.listdir(tmp_path) == [target.name]


def test_verify_false_replaces_target_without_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.nt"
    target.write_bytes(OLD_CONTENT)

    def serialize_garbage(self, stream, base=None, encoding=None, **kwargs):
        stream.write(b"unverified bytes\n")

    monkeypatch.setattr(NTSerializer, "serialize", serialize_garbage)

    result = safe_serialize(_make_graph(), target, format="nt", verify=False)
    assert target.read_bytes() == b"unverified bytes\n"
    assert result.bytes_written == len(b"unverified bytes\n")
    assert os.listdir(tmp_path) == [target.name]


def test_existing_target_permissions_are_preserved(tmp_path: Path) -> None:
    target = tmp_path / "out.nt"
    target.write_bytes(OLD_CONTENT)
    os.chmod(target, 0o640)

    safe_serialize(_make_graph(), target, format="nt")

    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o640


def test_rejects_open_file_object(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="filesystem path"):
        safe_serialize(_make_graph(), io.BytesIO())  # type: ignore[arg-type]


def test_missing_target_directory_raises(tmp_path: Path) -> None:
    target = tmp_path / "missing-dir" / "out.ttl"
    with pytest.raises(FileNotFoundError, match="directory"):
        safe_serialize(_make_graph(), target)
    assert os.listdir(tmp_path) == []
