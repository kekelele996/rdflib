"""
Fixed, immutable read views (snapshots) for consistent paging and querying.

When a graph or dataset keeps receiving statements while it is being read,
every page is an independent request against the current state: statements
that were inserted between two page requests shift the offsets, so rows are
returned twice or skipped, and separate SPARQL queries may return different
result sets. There is no way to tell whether two pages even belong to the
same view of the data.

A [`Snapshot`][rdflib.snapshot.Snapshot] solves this by materializing the
matching statements once, at the moment it is opened. All paging, cursors
and SPARQL queries are then served from that fixed materialized view, and
each page carries the snapshot's identifier:

```python
snapshot = graph.snapshot()
cursor = snapshot.cursor(page_size=100)
pages = [page.items for page in cursor]
```

The snapshot records the source graph's
[`revision`][rdflib.graph.Graph.revision]. If the graph is written to while
the snapshot is being paged through, the next cursor or query request raises
[`StaleSnapshotError`][rdflib.snapshot.StaleSnapshotError] instead of
silently shifting pages or mixing newly written statements into an earlier
result. The caller then makes an explicit choice:

- [`Snapshot.overwrite()`][rdflib.snapshot.Snapshot.overwrite] replaces the
  fixed view with the current content (the snapshot object is reused, with a
  new view identifier, and existing cursors are invalidated); or
- [`Snapshot.close()`][rdflib.snapshot.Snapshot.close] followed by a new
  [`Graph.snapshot()`][rdflib.graph.Graph.snapshot] call reopens a fresh
  view.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from rdflib.paths import Path
from rdflib.query import ResultRow
from rdflib.term import Node

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rdflib.graph import Graph, _TriplePatternType
    from rdflib.query import Result

__all__ = [
    "Snapshot",
    "SnapshotCursor",
    "Page",
    "SnapshotError",
    "SnapshotClosedError",
    "StaleSnapshotError",
]

_T = TypeVar("_T")


class SnapshotError(Exception):
    """Base class for errors raised while working with snapshots."""


class SnapshotClosedError(SnapshotError):
    """Raised when a closed snapshot, or one of its cursors, is used."""


class StaleSnapshotError(SnapshotError):
    """Raised when the source graph changed after the snapshot was opened.

    Already served pages are not affected (they came from the fixed view);
    no further page or query is served until an explicit decision is made.

    Attributes:
        snapshot: The snapshot that went stale.
        captured_revision: Revision of the source graph when the fixed view
            was captured.
        current_revision: Revision of the source graph at the time of the
            failed request.
    """

    def __init__(
        self,
        snapshot: Snapshot,
        captured_revision: int,
        current_revision: int,
    ) -> None:
        self.snapshot = snapshot
        self.captured_revision = captured_revision
        self.current_revision = current_revision
        super().__init__(
            f"snapshot {snapshot.id!r} of {snapshot.source!r} is stale: the "
            f"source graph changed since revision {captured_revision} (now "
            f"{current_revision}). Statements written since the view was "
            "opened were not mixed into any earlier page. Choose explicitly "
            "how to continue: call snapshot.overwrite() to replace the fixed "
            "view with the current content (existing cursors are "
            "invalidated), or call snapshot.close() and graph.snapshot() to "
            "reopen a new view."
        )


def _validate_pattern(pattern: tuple[object, object, object]) -> None:
    if not isinstance(pattern, tuple) or len(pattern) != 3:
        raise TypeError(
            "triple pattern must be a tuple of three elements "
            "(subject, predicate, object)"
        )
    for position, term in zip(("subject", "predicate", "object"), pattern):
        if term is None:
            continue
        if isinstance(term, Path):
            raise TypeError(
                f"snapshots do not support property paths in the {position} "
                "position, query the snapshot with SPARQL instead"
            )
        if not isinstance(term, Node):
            raise TypeError(
                f"the {position} of a triple pattern must be an rdflib term "
                f"or None, got {term!r}"
            )


def _matches(
    triple: tuple[Node, Node, Node], pattern: tuple[object, object, object]
) -> bool:
    subject, predicate, object_ = triple
    pattern_subject, pattern_predicate, pattern_object = pattern
    return (
        (pattern_subject is None or pattern_subject == subject)
        and (pattern_predicate is None or pattern_predicate == predicate)
        and (pattern_object is None or pattern_object == object_)
    )


class Page(Generic[_T]):
    """One consecutive page of items read from a snapshot.

    Attributes:
        number: 1-based position of the page within the cursor.
        size: Requested page size (the last page may contain fewer items).
        items: Materialized items of this page.
        total: Total number of items the cursor pages over.
        snapshot_id: Identifier of the fixed view this page belongs to. Two
            pages with the same identifier are guaranteed to come from the
            same materialized view.
        is_last: `True` if this is the last page of the cursor.
    """

    def __init__(
        self,
        number: int,
        size: int,
        items: tuple[_T, ...],
        total: int,
        snapshot_id: str,
        is_last: bool,
    ):
        self.number = number
        self.size = size
        self.items = items
        self.total = total
        self.snapshot_id = snapshot_id
        self.is_last = is_last

    def __iter__(self) -> Iterator[_T]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> _T:
        return self.items[item]

    def __repr__(self) -> str:
        return (
            f"<Page number={self.number} size={self.size} "
            f"items={len(self.items)} total={self.total} "
            f"snapshot_id={self.id!r}>"
        )

    @property
    def id(self) -> str:
        """Alias of [`Page.snapshot_id`][rdflib.snapshot.Page.snapshot_id]."""
        return self.snapshot_id


class SnapshotCursor(Generic[_T]):
    """A strictly sequential cursor over a fixed snapshot view.

    Pages are produced by consecutive calls to
    [`next_page()`][rdflib.snapshot.SnapshotCursor.next_page] (or iterating
    the cursor); offsets always refer to the same materialized items, so a
    cursor never silently skips or repeats a page. A specific page can be
    requested explicitly with
    [`page()`][rdflib.snapshot.SnapshotCursor.page].
    """

    def __init__(
        self,
        snapshot: Snapshot,
        items: tuple[_T, ...],
        page_size: int,
    ):
        if not isinstance(page_size, int) or isinstance(page_size, bool):
            raise TypeError("page_size must be an integer")
        if page_size < 1:
            raise ValueError("page_size must be greater than 0")
        self._snapshot = snapshot
        self._items = items
        self.page_size = page_size
        self.page_number = 0
        #: `None` while usable, otherwise why the cursor was invalidated.
        self._invalid: str | None = None

    @property
    def snapshot(self) -> Snapshot:
        """The fixed view this cursor pages over."""
        return self._snapshot

    @property
    def total(self) -> int:
        """Total number of items the cursor pages over."""
        return len(self._items)

    @property
    def pages(self) -> int:
        """Total number of pages (``0`` if there are no items)."""
        return (len(self._items) + self.page_size - 1) // self.page_size

    @property
    def has_next(self) -> bool:
        """Whether another consecutive page is available."""
        return self.page_number < self.pages

    def _invalidate(self, reason: str) -> None:
        self._invalid = reason

    def _guard(self) -> None:
        if self._invalid is not None:
            if self._invalid == "closed":
                raise SnapshotClosedError(
                    f"cursor on snapshot {self._snapshot.id!r} cannot be used "
                    "after the snapshot was closed"
                )
            raise SnapshotError(
                f"cursor on snapshot {self._snapshot.id!r} was invalidated "
                f"because the view was {self._invalid}; request a new cursor "
                "from the current view"
            )
        # Raises SnapshotClosedError / StaleSnapshotError when applicable.
        self._snapshot._guard()

    def next_page(self) -> Page[_T]:
        """Return the next consecutive page.

        Raises:
            StopIteration: When every page has already been returned.
            StaleSnapshotError: When the source graph changed since the view
                was opened.
            SnapshotClosedError: When the snapshot was closed.
        """
        self._guard()
        if self.page_number >= self.pages:
            raise StopIteration
        return self._serve(self.page_number + 1)

    def page(self, number: int) -> Page[_T]:
        """Explicitly request page `number` (1-based).

        Unlike consecutive iteration this allows jumping to an arbitrary page,
        but only on an explicit caller request; it never clamps an invalid
        number or moves silently. The cursor position is set to `number`, so
        the following
        [`next_page()`][rdflib.snapshot.SnapshotCursor.next_page] call
        returns page `number + 1`.

        Raises:
            IndexError: When `number` is not a valid 1-based page number.
        """
        self._guard()
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or number < 1
            or number > self.pages
        ):
            raise IndexError(
                f"page number must be an integer between 1 and {self.pages}, "
                f"got {number!r}"
            )
        return self._serve(number)

    def _serve(self, number: int) -> Page[_T]:
        start = (number - 1) * self.page_size
        items = self._items[start : start + self.page_size]
        self.page_number = number
        return Page(
            number=number,
            size=self.page_size,
            items=tuple(items),
            total=len(self._items),
            snapshot_id=self._snapshot.id,
            is_last=number == self.pages,
        )

    def close(self) -> None:
        """Detach the cursor from its snapshot."""
        if self._invalid is None:
            self._snapshot._detach_cursor(self)
            self._invalid = "closed"

    def __iter__(self) -> SnapshotCursor[_T]:
        return self

    def __next__(self) -> Page[_T]:
        return self.next_page()

    def __repr__(self) -> str:
        return (
            f"<SnapshotCursor snapshot_id={self._snapshot.id!r} "
            f"page={self.page_number}/{self.pages} "
            f"page_size={self.page_size}>"
        )


class Snapshot:
    """An immutable, materialized read view of a graph at one point in time.

    Use [`Graph.snapshot()`][rdflib.graph.Graph.snapshot] to open a snapshot
    rather than instantiating this class directly.
    """

    def __init__(
        self,
        graph: Graph,
        triple_pattern: _TriplePatternType = (None, None, None),
    ):
        _validate_pattern(triple_pattern)
        self._source = graph
        self._pattern = triple_pattern
        self._cursors: set[SnapshotCursor[Any]] = set()
        self._closed = False
        self._frozen: Graph | None = None
        self._id = uuid.uuid4().hex
        self._revision = graph.revision
        self._items: tuple[tuple[Node, Node, Node], ...] = self._materialize()

    def _materialize(self) -> tuple[tuple[Node, Node, Node], ...]:
        # Sorted into a deterministic order: pages are derived by fixed
        # offsets into this tuple, so a stable order prevents overlap and
        # gaps between pages regardless of store internals.
        from rdflib.graph import ConjunctiveGraph

        if isinstance(self._source, ConjunctiveGraph):
            # A dataset is browsed as the union of all its (default and
            # named) graphs; collapse identical statements from several
            # graphs the way store-level triple iteration does.
            statements = {
                (s, p, o)
                for s, p, o, _context in self._source.quads((None, None, None, None))
            }
        else:
            statements = set(self._source.triples((None, None, None)))
        pattern = self._pattern
        if pattern == (None, None, None):
            return tuple(sorted(statements))
        return tuple(
            sorted(triple for triple in statements if _matches(triple, pattern))
        )

    @property
    def source(self) -> Graph:
        """The graph this view was opened from."""
        return self._source

    @property
    def id(self) -> str:
        """Identifier of the fixed view; changes when the view is overwritten."""
        return self._id

    @property
    def revision(self) -> int:
        """Revision of the source graph captured with this view."""
        return self._revision

    @property
    def current_revision(self) -> int:
        """Current revision of the source graph."""
        return self._source.revision

    @property
    def is_stale(self) -> bool:
        """`True` if the source graph changed after this view was opened."""
        return self._source.revision != self._revision

    @property
    def closed(self) -> bool:
        """`True` if this view has been closed."""
        return self._closed

    def _guard(self) -> None:
        if self._closed:
            raise SnapshotClosedError(
                f"snapshot {self._id!r} is closed; reopen one with graph.snapshot()"
            )
        current_revision = self._source.revision
        if current_revision != self._revision:
            raise StaleSnapshotError(self, self._revision, current_revision)

    def _attach_cursor(self, cursor: SnapshotCursor[Any]) -> None:
        self._cursors.add(cursor)

    def _detach_cursor(self, cursor: SnapshotCursor[Any]) -> None:
        self._cursors.discard(cursor)

    def triples(
        self,
        triple: _TriplePatternType = (None, None, None),
    ) -> Iterator[tuple[Node, Node, Node]]:
        """Iterate over the materialized statements matching `triple`.

        This reads the fixed view directly; it never observes statements
        written to the source graph after the view was opened, and therefore
        never raises a staleness error. Paging sessions and new queries,
        which must decide whether to account for newer writes, do.
        """
        _validate_pattern(triple)
        for statement in self._items:
            if _matches(statement, triple):
                yield statement

    def cursor(
        self,
        triple: _TriplePatternType | None = None,
        page_size: int = 100,
    ) -> SnapshotCursor[tuple[Node, Node, Node]]:
        """Open a paging cursor over the fixed view.

        Args:
            triple: Optional triple pattern restricting the cursor, matched
                against the materialized statements.
            page_size: Number of statements per page.

        Raises:
            StaleSnapshotError: When the source graph already changed since
                the view was opened.
        """
        self._guard()
        pattern = self._pattern if triple is None else triple
        _validate_pattern(pattern)
        items = tuple(
            statement for statement in self._items if _matches(statement, pattern)
        )
        cursor: SnapshotCursor[tuple[Node, Node, Node]] = SnapshotCursor(
            self, items, page_size
        )
        self._attach_cursor(cursor)
        return cursor

    def _frozen_graph(self) -> Graph:
        # Built lazily so opening a snapshot purely for triple paging does
        # not pay for a second copy. Rebuilt whenever the view is overwritten.
        if self._frozen is None:
            from rdflib.graph import Graph

            frozen = Graph()
            frozen.addN((s, p, o, frozen) for s, p, o in self._items)
            for prefix, namespace in self._source.namespaces():
                frozen.bind(prefix, namespace, override=False)
            self._frozen = frozen
        return self._frozen

    def query(self, *args: object, **kwargs: object) -> Result:
        """Run a SPARQL query against the fixed view.

        Has the same signature as [`Graph.query()`][rdflib.graph.Graph.query]
        but always evaluates against the materialized statements, so
        repeated queries (including separate `LIMIT`/`OFFSET` requests) see
        the same data. Raises
        [`StaleSnapshotError`][rdflib.snapshot.StaleSnapshotError] if the
        source graph changed since the view was opened.
        """
        self._guard()
        return self._frozen_graph().query(*args, **kwargs)  # type: ignore[arg-type]

    def query_cursor(
        self,
        query_object: object,
        page_size: int = 100,
        *args: object,
        **kwargs: object,
    ) -> SnapshotCursor[ResultRow]:
        """Open a paging cursor over a `SELECT` query on the fixed view.

        The query is evaluated once against the materialized statements and
        its rows are paged like
        [`cursor()`][rdflib.snapshot.Snapshot.cursor], with the same
        staleness rules.

        Raises:
            SnapshotError: When the query does not return a `SELECT` result.
        """
        self._guard()
        result = self._frozen_graph().query(query_object, *args, **kwargs)  # type: ignore[arg-type]
        if result.type != "SELECT":
            raise SnapshotError(
                f"query_cursor() only supports SELECT queries, got {result.type}"
            )
        rows: tuple[ResultRow, ...] = tuple(cast(ResultRow, row) for row in result)
        cursor: SnapshotCursor[ResultRow] = SnapshotCursor(self, rows, page_size)
        self._attach_cursor(cursor)
        return cursor

    def overwrite(self) -> Snapshot:
        """Overwrite the fixed view with the current source content.

        The same snapshot object is reused, but it gets a new view
        identifier and captures the current source revision; existing
        cursors of the previous view are invalidated.
        """
        if self._closed:
            raise SnapshotClosedError(
                f"snapshot {self._id!r} is closed, open a new one with graph.snapshot()"
            )
        for cursor in tuple(self._cursors):
            cursor._invalidate("overwritten")
        self._cursors.clear()
        self._frozen = None
        self._id = uuid.uuid4().hex
        self._revision = self._source.revision
        self._items = self._materialize()
        return self

    def refresh(self) -> Snapshot:
        """Alias of [`overwrite()`][rdflib.snapshot.Snapshot.overwrite]."""
        return self.overwrite()

    def close(self) -> None:
        """Close the view, releasing the materialized statements.

        Existing cursors are invalidated. Open a new view with
        [`Graph.snapshot()`][rdflib.graph.Graph.snapshot] afterwards.
        """
        if self._closed:
            return
        for cursor in tuple(self._cursors):
            cursor._invalidate("closed")
        self._cursors.clear()
        self._frozen = None
        self._items = ()
        self._closed = True

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[tuple[Node, Node, Node]]:
        return iter(self._items)

    def __contains__(self, triple: object) -> bool:
        if not isinstance(triple, tuple) or len(triple) != 3:
            raise TypeError(
                "a triple (subject, predicate, object) is required for "
                "membership testing"
            )
        return triple in self._items

    def __enter__(self) -> Snapshot:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else ("stale" if self.is_stale else "current")
        return (
            f"<Snapshot id={self.id!r} revision={self._revision} "
            f"statements={len(self)} state={state}>"
        )
