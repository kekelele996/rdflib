"""
Crash-safe publishing of a serialized graph.

``Graph.serialize(destination=path)`` truncates the destination as soon as it
opens it, so a serializer error or a full disk halfway through leaves a
truncated file where the last good export used to be. This module publishes a
graph the other way around:

1. Serialize to a hidden temporary file in the same directory as the target
   ("next to" it), then flush and ``fsync`` it, so a full disk is reported
   before the target is touched.
2. Read the temporary file back into a fresh graph and check that it contains
   the same statements (and, for datasets, the same named graphs) as the
   source graph. Blank node labels are allowed to differ, only the statements
   they stand in have to match.
3. Atomically replace the target with ``os.replace``.

If anything goes wrong the temporary file is removed and the existing
destination file is left byte-for-byte untouched.

Example:
    ```python
    from rdflib import Dataset
    from rdflib.tools.safe_serialize import safe_serialize

    ds = Dataset()
    ds.parse("daily_input.trig", format="trig")
    result = safe_serialize(ds, "/srv/data/daily.trig")
    print(result.summary())
    # published 42 statement(s) (2 named graph(s)) as trig to
    # /srv/data/daily.trig (5678 bytes)
    ```
"""

from __future__ import annotations

import logging
import os
import stat
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from rdflib.compare import isomorphic, to_canonical_graph
from rdflib.exceptions import SerializationVerificationError
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, ConjunctiveGraph, Dataset, Graph
from rdflib.term import IdentifiedNode, Node, URIRef
from rdflib.util import guess_format

__all__ = ["PublishResult", "safe_serialize"]

logger = logging.getLogger(__name__)

#: A named graph entry as used internally: the graph IRI for IRI-named graphs,
#: or ``None`` for blank-node-named graphs whose labels are allowed to change
#: on a parse round-trip. The second element is the graph's canonicalized
#: statement set.
_NamedGraphKey = NamedTuple(
    "_NamedGraphKey",
    [
        ("identifier", URIRef | None),
        ("statements", frozenset[tuple[Node, Node, Node]]),
    ],
)


@dataclass(frozen=True)
class PublishResult:
    """Information about a successfully published graph file.

    Attributes:
        destination: Absolute path the graph was published to.
        format: The serializer format that was written, e.g. ``"turtle"``.
        statements: Number of statements published. For a plain
            [`Graph`][rdflib.graph.Graph] this is the number of triples; for a
            [`Dataset`][rdflib.graph.Dataset] or
            [`ConjunctiveGraph`][rdflib.graph.ConjunctiveGraph] this is the
            total number of quads, including default graph statements.
        named_graphs: Identifiers of the non-empty named graphs that were
            published. Empty for plain graphs.
        bytes_written: Size of the published file in bytes.
    """

    destination: Path
    format: str
    statements: int
    named_graphs: tuple[IdentifiedNode, ...] = ()
    bytes_written: int = 0

    def summary(self) -> str:
        """One-line, human-readable description of the publication."""
        if self.named_graphs:
            scope = f" ({len(self.named_graphs)} named graph(s))"
        else:
            scope = ""
        return (
            f"published {self.statements} statement(s){scope} as {self.format} "
            f"to {self.destination} ({self.bytes_written} bytes)"
        )

    def __str__(self) -> str:
        return self.summary()


def safe_serialize(
    graph: Graph,
    destination: str | os.PathLike[str],
    format: str | None = None,
    *,
    base: str | None = None,
    encoding: str | None = None,
    verify: bool = True,
    **kwargs: Any,
) -> PublishResult:
    """Serialize ``graph`` to ``destination`` without risking the old file.

    The graph is first written to a hidden temporary file next to the
    destination. When ``verify`` is true (the default) the temporary file is
    parsed back and checked against ``graph`` before the destination is
    atomically replaced. On any failure the temporary file is removed and the
    destination, if it existed, is left completely untouched.

    Args:
        graph: The graph (or dataset) to publish.
        destination: Filesystem path of the file to publish. Unlike
            [`Graph.serialize`][rdflib.graph.Graph.serialize] this must be a
            path, not an open file object, so the replacement can be atomic.
        format: Serializer format to write. If not given it is guessed from
            the destination suffix, falling back to ``"trig"`` for datasets
            and ``"turtle"`` for plain graphs.
        base: Base IRI forwarded to the serializer.
        encoding: Encoding forwarded to the serializer.
        verify: Read the side file back and verify statement count and named
            graphs before replacing the destination.
        **kwargs: Additional keyword arguments forwarded to the serializer.

    Returns:
        A [`PublishResult`][rdflib.tools.safe_serialize.PublishResult]
        describing the format, statement count, named graphs and publish
        location. The same line is logged at ``INFO`` level.

    Raises:
        TypeError: If ``destination`` is an open file object instead of a path.
        FileNotFoundError: If the destination directory does not exist.
        SerializationVerificationError: If writing, flushing, reading back,
            verifying or replacing fails. The existing destination file is
            preserved in every such case.
    """
    if hasattr(destination, "write"):
        raise TypeError(
            "safe_serialize needs a filesystem path to replace the destination "
            "atomically, not an open file object"
        )

    target = Path(destination)
    target_dir = target.parent
    if not target_dir.is_dir():
        raise FileNotFoundError(
            f"cannot publish to {target}: directory {target_dir} does not exist"
        )

    if format is None:
        format = guess_format(target.name) or (
            "trig" if isinstance(graph, ConjunctiveGraph) else "turtle"
        )

    statements = len(graph)
    named_graph_ids = _named_graph_identifiers(graph)

    # A unique, hidden side file in the same directory guarantees that
    # os.replace is atomic (same filesystem) and that concurrent runs do not
    # collide with each other.
    fd, tmp_name = tempfile.mkstemp(
        dir=target_dir, prefix=f".{target.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    published = False
    try:
        # Keep the permission bits downstream already relies on.
        try:
            current_mode = stat.S_IMODE(target.stat().st_mode)
        except FileNotFoundError:
            pass
        else:
            os.chmod(tmp_path, current_mode)

        try:
            with os.fdopen(fd, "wb") as stream:
                graph.serialize(
                    destination=stream,
                    format=format,
                    base=base,
                    encoding=encoding,
                    **kwargs,
                )
                stream.flush()
                # A full disk is often only reported while flushing the
                # buffers or syncing, so do both before touching the target.
                os.fsync(stream.fileno())
        except Exception as exc:
            raise SerializationVerificationError(
                f"failed to write {format} graph to side file {tmp_path.name}: {exc}",
                stage="write",
            ) from exc

        bytes_written = tmp_path.stat().st_size

        if verify:
            roundtrip = _new_container(graph)
            try:
                roundtrip.parse(tmp_path, format=format)
            except Exception as exc:
                raise SerializationVerificationError(
                    f"side file {tmp_path.name} could not be parsed back as "
                    f"{format!r}, it is not a valid {format} serialization: {exc}",
                    stage="readback",
                ) from exc
            _verify_equivalence(graph, roundtrip)

        try:
            os.replace(tmp_path, target)
        except OSError as exc:
            raise SerializationVerificationError(
                f"failed to replace {target} with the verified side file "
                f"{tmp_path.name}: {exc}",
                stage="publish",
            ) from exc
        _fsync_directory(target_dir)
        published = True

        result = PublishResult(
            destination=target.resolve(),
            format=format,
            statements=statements,
            named_graphs=named_graph_ids,
            bytes_written=bytes_written,
        )
        logger.info(result.summary())
        return result
    finally:
        if not published:
            tmp_path.unlink(missing_ok=True)


def _new_container(graph: Graph) -> Graph:
    """A fresh graph of the same kind (triples vs quads) for read-back."""
    if isinstance(graph, Dataset):
        return Dataset()
    if isinstance(graph, ConjunctiveGraph):
        return ConjunctiveGraph()
    return Graph()


def _default_identifier(graph: ConjunctiveGraph) -> IdentifiedNode:
    if isinstance(graph, Dataset):
        return DATASET_DEFAULT_GRAPH_ID
    return graph.default_context.identifier


def _contexts(graph: ConjunctiveGraph) -> Iterator[Graph]:
    # Dataset.contexts is deprecated, Dataset.graphs is its replacement;
    # plain ConjunctiveGraph only provides contexts().
    if isinstance(graph, Dataset):
        yield from graph.graphs()
    else:
        yield from graph.contexts()


def _canonical_statements(graph: Graph) -> frozenset[tuple[Node, Node, Node]]:
    """Statements of a graph with blank nodes deterministically relabelled."""
    return frozenset(to_canonical_graph(graph))


def _named_graph_identifiers(graph: Graph) -> tuple[IdentifiedNode, ...]:
    if not isinstance(graph, ConjunctiveGraph):
        return ()
    default_id = _default_identifier(graph)
    identifiers = {
        context.identifier
        for context in _contexts(graph)
        if context.identifier != default_id and len(context) > 0
    }
    return tuple(sorted(identifiers, key=lambda node: str(node)))


def _dataset_signature(
    graph: ConjunctiveGraph,
) -> tuple[frozenset[tuple[Node, Node, Node]], Counter[_NamedGraphKey]]:
    """Comparable structure: canonical default graph + multiset of named graphs.

    Named graphs are keyed by their IRI when they have one, otherwise only by
    their canonicalized content, because blank-node graph names get fresh
    labels when parsed back.
    """
    default_id = _default_identifier(graph)
    default_statements: frozenset[tuple[Node, Node, Node]] = frozenset()
    named_graphs: Counter[_NamedGraphKey] = Counter()
    for context in _contexts(graph):
        if context.identifier == default_id:
            default_statements = _canonical_statements(context)
        elif len(context) > 0:
            statements = _canonical_statements(context)
            identifier = context.identifier
            if isinstance(identifier, URIRef):
                named_graphs[_NamedGraphKey(identifier, statements)] += 1
            else:
                named_graphs[_NamedGraphKey(None, statements)] += 1
    return default_statements, named_graphs


def _verify_equivalence(source: Graph, readback: Graph) -> None:
    """Assert the read-back graph matches the source graph.

    Every mismatch that is cheap to determine is collected, so the error can
    name the statement count and the missing or differing named graphs.
    Blank-node labels are allowed to differ.
    """
    problems: list[str] = []
    if len(source) != len(readback):
        problems.append(
            "statement count mismatch: source graph has "
            f"{len(source)} statement(s) but the side file read back has "
            f"{len(readback)}"
        )

    if not isinstance(source, ConjunctiveGraph):
        if not isomorphic(source, readback):
            problems.append(
                "the statements read back from the side file do not match the "
                "source graph"
            )
    else:
        source_default, source_named = _dataset_signature(source)
        readback_default, readback_named = _dataset_signature(
            readback  # type: ignore[arg-type]
        )
        if source_default != readback_default:
            problems.append("the default graph statements differ")
        if source_named != readback_named:
            problems.extend(_describe_named_graph_diff(source_named, readback_named))

    if problems:
        raise SerializationVerificationError(
            "side file verification failed: " + "; ".join(problems),
            stage="verify",
        )


def _describe_named_graph_diff(
    source: Counter[_NamedGraphKey],
    readback: Counter[_NamedGraphKey],
) -> list[str]:
    source_by_id = {
        key.identifier: key.statements for key in source if key.identifier is not None
    }
    readback_by_id = {
        key.identifier: key.statements for key in readback if key.identifier is not None
    }
    source_ids = set(source_by_id)
    readback_ids = set(readback_by_id)

    problems: list[str] = []
    missing = source_ids - readback_ids
    extra = readback_ids - source_ids
    changed = {
        identifier
        for identifier in source_ids & readback_ids
        if source_by_id[identifier] != readback_by_id[identifier]
    }
    if missing:
        problems.append(
            f"missing named graph(s): {', '.join(map(str, sorted(missing)))}"
        )
    if extra:
        problems.append(
            f"unexpected named graph(s): {', '.join(map(str, sorted(extra)))}"
        )
    if changed:
        problems.append(
            f"named graph(s) with different statements: "
            f"{', '.join(map(str, sorted(changed)))}"
        )
    if not problems:
        # Different number of, or different statements in, blank-node named
        # graphs whose labels cannot be compared across the round trip.
        problems.append(
            f"source has {sum(source.values())} named graph(s) but the side "
            f"file read back has {sum(readback.values())} with matching content"
        )
    return problems


def _fsync_directory(directory: Path) -> None:
    """Best-effort flush of the directory entry after a rename."""
    flags = os.O_RDONLY
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is not None:
        flags |= directory_flag
    try:
        dir_fd = os.open(str(directory), flags)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        # Some filesystems do not support fsync on directories.
        pass
    finally:
        os.close(dir_fd)
