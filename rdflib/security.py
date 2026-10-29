"""
Named-graph level access control for shared datasets.

This module provides per-named-graph ``READ`` / ``WRITE`` / ``DELETE``
permissions for a :class:`~rdflib.graph.Dataset` that is shared by multiple
teams in one process.

The pieces are:

* :class:`Permission` -- the three access modes that can be granted.
* :class:`GraphAccessPolicy` -- maps ``(principal, named graph)`` pairs to the
  permissions that principal has on that graph.  Anything not granted is
  denied, and ``WRITE`` never implies ``DELETE``.
* :class:`SecuredDataset` -- a :class:`~rdflib.graph.Dataset` whose underlying
  store is wrapped in a :class:`GuardedStore`.  Every read and modification
  entry point (direct API calls, bulk ``addN``, parsing, SPARQL queries and
  SPARQL updates) is checked against the *same* current caller identity.
* :func:`principal` -- a context manager that sets the caller identity used by
  all checks, backed by a :mod:`contextvars` variable so each request/thread
  in a shared process can carry its own principal.

When an operation is refused, :class:`AccessDeniedError` names both the graph
and the missing permission, e.g.::

    access denied: principal 'team-a' is missing WRITE permission on graph
    <http://example.org/team-b>

Refused operations never leave partial modifications behind: direct bulk
operations are authorised before anything is written, and multi-graph
operations (parsing quad documents and SPARQL update requests, including
dynamically resolved graph names) run against a snapshot and are rolled back
if any of their steps is refused.

.. note::

    Enforcement is local to the process, at the store API level.  It does not
    secure remote stores (e.g. :class:`~rdflib.plugins.stores.sparqlstore`)
    that are spoken to over the network.
"""

from __future__ import annotations

import contextvars
import warnings
from collections.abc import (
    Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
)
from contextlib import contextmanager
from enum import Flag, auto
from typing import Any, TypeGuard, cast

from rdflib import query as rdflib_query
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Dataset, Graph
from rdflib.plugins.sparql.algebra import translateQuery, translateUpdate
from rdflib.plugins.sparql.parser import parseQuery, parseUpdate
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import Query, Update
from rdflib.query import Result
from rdflib.store import Store
from rdflib.term import BNode, IdentifiedNode, Identifier, Node, URIRef

__all__ = [
    "ANY_GRAPH",
    "AccessDeniedError",
    "AnyGraph",
    "GraphAccessPolicy",
    "GuardedStore",
    "IdentityRequiredError",
    "Permission",
    "SecuredDataset",
    "SecurityError",
    "current_principal",
    "principal",
]


# ---------------------------------------------------------------------------
# Permissions, identity and policy
# ---------------------------------------------------------------------------


class Permission(Flag):
    """Access modes that can be granted on a named graph.

    Permissions are independent: holding :attr:`WRITE` does not grant
    :attr:`DELETE` and holding either does not grant :attr:`READ`.  Flags can
    be combined, e.g. ``Permission.READ | Permission.WRITE``.
    """

    READ = auto()
    WRITE = auto()
    DELETE = auto()


class AnyGraph:
    """Sentinel used in a policy to refer to every named graph.

    A grant for :data:`ANY_GRAPH` applies to every graph, including graphs
    that do not exist yet.  Exact graph grants take precedence alongside it
    (their permissions are unioned).
    """

    _instance: AnyGraph | None = None

    def __new__(cls) -> AnyGraph:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ANY_GRAPH"

    def __str__(self) -> str:
        return "*"


ANY_GRAPH = AnyGraph()
"""Sentinel policy key matching any named graph."""

GraphRef = IdentifiedNode | AnyGraph
"""Something a policy can grant permissions on."""

Principal = Hashable
"""A caller identity. Any hashable value, typically a team or user string."""


class SecurityError(Exception):
    """Base class for security related errors."""


class IdentityRequiredError(SecurityError):
    """Raised when an operation needs a caller identity but none is set."""

    def __init__(self) -> None:
        super().__init__(
            "no caller principal is set; use rdflib.security.principal() or "
            "pass a fixed identity to SecuredDataset"
        )


class AccessDeniedError(SecurityError):
    """Raised when a principal is missing a permission on a named graph.

    Attributes:
        principal: The caller identity the operation was attempted as.
        graph: The graph identifier (or :data:`ANY_GRAPH`) that was accessed.
        permission: The permission that was missing.
    """

    def __init__(
        self,
        principal: Principal,
        graph: GraphRef,
        permission: Permission,
    ) -> None:
        self.principal = principal
        self.graph = graph
        self.permission = permission
        super().__init__(
            "access denied: principal %r is missing %s permission on graph %s"
            % (principal, _permission_name(permission), _graph_label(graph))
        )


def _permission_name(permission: Permission) -> str:
    name = permission.name
    if name is not None:
        return name
    return "|".join(part.name or str(part.value) for part in permission)


def _graph_label(graph: GraphRef) -> str:
    if isinstance(graph, AnyGraph):
        return "ANY_GRAPH"
    return graph.n3()


class GraphAccessPolicy:
    """Per-principal, per-named-graph permissions.

    By default nothing is allowed; every access must be granted explicitly,
    either for the exact graph or via :data:`ANY_GRAPH`.
    """

    def __init__(self, *, default: Permission = Permission(0)) -> None:
        """Args:
        default: Permissions granted to every principal on every graph.
            Defaults to no permissions, i.e. deny by default.
        """
        self._rules: dict[Any, dict[Principal, Permission]] = {}
        self._default = default

    def grant(
        self,
        principal: Principal,
        graph: GraphRef,
        *permissions: Permission,
    ) -> GraphAccessPolicy:
        """Grant permissions (unioning them with existing grants).

        Args:
            principal: The caller identity to grant to.
            graph: A graph identifier, or :data:`ANY_GRAPH`.
            *permissions: One or more :class:`Permission` flags.
        """
        if not permissions:
            raise ValueError("grant() requires at least one permission")
        table = self._rules.setdefault(graph, {})
        current = table.get(principal, Permission(0))
        for permission in permissions:
            current |= permission
        table[principal] = current
        return self

    def revoke(
        self,
        principal: Principal,
        graph: GraphRef,
        *permissions: Permission,
    ) -> GraphAccessPolicy:
        """Revoke permissions from a principal on a graph."""
        table = self._rules.get(graph)
        if table is None or principal not in table:
            return self
        current = table[principal]
        for permission in permissions:
            current &= ~permission
        if current is Permission(0):
            del table[principal]
            if not table:
                del self._rules[graph]
        else:
            table[principal] = current
        return self

    def permits(
        self,
        principal: Principal,
        graph: GraphRef,
        permission: Permission,
    ) -> bool:
        """Return whether `principal` holds `permission` on `graph`."""
        if permission in self._default:
            return True
        for key in (graph, ANY_GRAPH):
            table = self._rules.get(key)
            if table is not None and permission in table.get(principal, Permission(0)):
                return True
        return False


_current_principal: contextvars.ContextVar[Principal] = contextvars.ContextVar(
    "rdflib_security_current_principal"
)


def current_principal() -> Principal:
    """Return the caller identity for the current execution context.

    Raises:
        IdentityRequiredError: If no identity has been set with
            :func:`principal` (or a fixed identity on the dataset).
    """
    try:
        return _current_principal.get()
    except LookupError:
        raise IdentityRequiredError() from None


@contextmanager
def principal(identity: Principal) -> Iterator[None]:
    """Set the caller identity used by all security checks in this context.

    Backed by :mod:`contextvars`, so separate requests running in separate
    contexts (threads started with :func:`contextvars.copy_context`, async
    tasks, ...) each keep their own identity.

    Example:
        ```python
        with principal("team-a"):
            secured_ds.add((s, p, o, team_a_graph))
        ```
    """
    token = _current_principal.set(identity)
    try:
        yield
    finally:
        _current_principal.reset(token)


# ---------------------------------------------------------------------------
# Recorded violations (used to make multi-step operations atomic)
# ---------------------------------------------------------------------------

_violation_log: contextvars.ContextVar[list[AccessDeniedError] | None] = (
    contextvars.ContextVar("rdflib_security_violation_log", default=None)
)


# ---------------------------------------------------------------------------
# Guarded store
# ---------------------------------------------------------------------------

_Triple = tuple[Node, Node, Node]
_TriplePattern = tuple[Node | None, Node | None, Node | None]
_Snapshot = tuple[
    list[IdentifiedNode],
    list[tuple[Node, Node, Node, IdentifiedNode]],
]


class GuardedStore(Store):
    """A :class:`~rdflib.store.Store` proxy that enforces a graph policy.

    All graph data operations are funnelled through the four store mutation
    methods (``add``, ``addN``, ``remove``, ``remove_graph``) and the read
    methods (``triples``, ``triples_choices``, ``contexts``, ``__len__``), so
    wrapping a store guards every :class:`~rdflib.graph.Graph` /
    :class:`~rdflib.graph.Dataset` view and every SPARQL engine access,
    including views the engine creates itself.

    Explicitly targeting a graph that the caller cannot read raises
    :class:`AccessDeniedError`; unscoped scans (``context=None`` and graph
    enumeration) silently hide triples and graphs the caller may not read, so
    that union queries never leak other teams' data.
    """

    def __init__(
        self,
        store: Store,
        policy: GraphAccessPolicy,
        identity_provider: Callable[[], Principal] = current_principal,
    ) -> None:
        self._store = store
        self._policy = policy
        self._identity_provider = identity_provider
        # Mirror capability flags so Dataset/Graph initialisation behaves as
        # for the wrapped store.
        self.context_aware = store.context_aware
        self.formula_aware = store.formula_aware
        self.graph_aware = store.graph_aware
        self.transaction_aware = getattr(store, "transaction_aware", False)

    def __getattr__(self, name: str) -> Any:
        # Delegate everything we do not guard (namespace bindings, open/close,
        # ...) to the wrapped store.
        return getattr(self._store, name)

    # -- identity / authorisation helpers ----------------------------------

    def _principal(self) -> Principal:
        return self._identity_provider()

    def _identifier(self, context: Graph | IdentifiedNode | None) -> IdentifiedNode:
        if isinstance(context, Graph):
            return context.identifier
        assert context is not None
        return context

    def _readable(self, graph: IdentifiedNode) -> bool:
        return self._policy.permits(self._principal(), graph, Permission.READ)

    def _denied(self, graph: GraphRef, permission: Permission) -> bool:
        """Record or raise a violation.

        Returns ``True`` when the operation must be suppressed (a multi-step
        operation is recording violations), or raises
        :class:`AccessDeniedError` for ordinary calls.
        """
        error = AccessDeniedError(self._principal(), graph, permission)
        log = _violation_log.get()
        if log is not None:
            log.append(error)
            return True
        raise error

    def _check(self, graph: GraphRef, permission: Permission) -> bool:
        """Return True (suppress) if denied, False if allowed."""
        if self._policy.permits(self._principal(), graph, permission):
            return False
        return self._denied(graph, permission)

    def _existing_graphs(self) -> list[IdentifiedNode]:
        return sorted(
            (self._identifier(context) for context in self._store.contexts()),
            key=str,
        )

    # -- mutations ----------------------------------------------------------

    def add(
        self,
        triple: _Triple,
        context: Graph,
        quoted: bool = False,
    ) -> None:
        # Quoted triples belong to N3 formulae, not to dataset named graphs.
        if not quoted and self._check(context.identifier, Permission.WRITE):
            return
        self._store.add(triple, context, quoted=quoted)

    def addN(  # noqa: N802
        self, quads: Iterable[tuple[Node, Node, Node, Graph | None]]
    ) -> None:
        # Materialise first and authorise *every* target graph before any
        # quad reaches the store, so a mixed-graph bulk write is atomic.
        materialised = [
            (subject, predicate, object_, context)
            for subject, predicate, object_, context in quads
            if context is not None
        ]
        target_graphs: list[IdentifiedNode] = []
        for _s, _p, _o, context in materialised:
            identifier = context.identifier
            if identifier not in target_graphs:
                target_graphs.append(identifier)
        for identifier in target_graphs:
            if self._check(identifier, Permission.WRITE):
                return
        self._store.addN(materialised)

    def remove(
        self,
        triple_pattern: _TriplePattern,
        context: Graph | None = None,
    ) -> None:
        if context is None:
            # An unscoped remove touches every graph the triple is in: every
            # currently existing graph must be deletable, checked before
            # anything is removed.
            for identifier in self._existing_graphs():
                if self._check(identifier, Permission.DELETE):
                    return
        elif self._check(context.identifier, Permission.DELETE):
            return
        self._store.remove(triple_pattern, context)

    def add_graph(self, graph: Graph) -> None:
        identifier = graph.identifier
        # The default graph always exists; (re-)registering it is internal
        # bookkeeping (e.g. after it is cleared), not creating a new graph.
        if identifier != DATASET_DEFAULT_GRAPH_ID:
            already_known = any(
                existing == identifier for existing in self._existing_graphs()
            )
            if not already_known and self._check(identifier, Permission.WRITE):
                return
        self._store.add_graph(graph)

    def remove_graph(self, graph: Graph) -> None:
        # Dropping a graph that still holds triples destroys data -> DELETE.
        # Removing an empty (or unknown) registration is pure metadata with
        # nothing to protect (parsers create and discard scratch views while
        # loading, and DROP on an empty graph is a no-op in the store).
        if self._store.__len__(graph) > 0 and self._check(
            graph.identifier, Permission.DELETE
        ):
            return
        self._store.remove_graph(graph)

    def destroy(self, configuration: str) -> None:
        # Destroying the whole store is as destructive as deleting every
        # graph, so require DELETE on all of them.
        for identifier in self._existing_graphs():
            if self._check(identifier, Permission.DELETE):
                return
        self._store.destroy(configuration)

    # -- reads --------------------------------------------------------------

    def triples(
        self,
        triple_pattern: _TriplePattern,
        context: Graph | None = None,
    ) -> Iterator[tuple[_Triple, Iterator[Graph | None]]]:
        if context is None:
            for triple, contexts in self._store.triples(triple_pattern, None):
                readable = [
                    graph_context
                    for graph_context in contexts
                    if isinstance(graph_context, Graph)
                    and self._readable(graph_context.identifier)
                ]
                if readable:
                    yield triple, iter(cast(list[Graph | None], readable))
        else:
            if self._check(context.identifier, Permission.READ):
                return
            yield from self._store.triples(triple_pattern, context)

    def triples_choices(
        self,
        triple: tuple[Any, Any, Any],
        context: Graph | None = None,
    ) -> Generator[tuple[_Triple, Iterator[Graph | None]], None, None]:
        if context is None:
            for triple_, contexts in self._store.triples_choices(triple, None):
                readable = [
                    graph_context
                    for graph_context in contexts
                    if isinstance(graph_context, Graph)
                    and self._readable(graph_context.identifier)
                ]
                if readable:
                    yield triple_, iter(cast(list[Graph | None], readable))
        else:
            if self._check(context.identifier, Permission.READ):
                return
            yield from self._store.triples_choices(triple, context)

    def contexts(self, triple: _Triple | None = None) -> Generator[Graph, None, None]:
        for graph_context in self._store.contexts(triple):
            if isinstance(graph_context, Graph) and self._readable(
                graph_context.identifier
            ):
                yield graph_context

    def __len__(self, context: Graph | None = None) -> int:
        if context is None:
            return sum(
                1 for _triple, _contexts in self.triples((None, None, None), None)
            )
        if self._check(context.identifier, Permission.READ):
            return 0
        return self._store.__len__(context)

    # -- never forward native SPARQL; the in-process engine is guarded -----

    def query(
        self,
        query: Query | str,
        initNs: Mapping[str, Any],  # noqa: N803
        initBindings: Mapping[str, Identifier],  # noqa: N803
        queryGraph: str,  # noqa: N803
        **kwargs: Any,
    ) -> Result:
        raise NotImplementedError(
            "GuardedStore does not forward native store queries; "
            "use SecuredDataset.query()"
        )

    def update(
        self,
        update: Update | str,
        initNs: Mapping[str, Any],  # noqa: N803
        initBindings: Mapping[str, Identifier],  # noqa: N803
        queryGraph: str,  # noqa: N803
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "GuardedStore does not forward native store updates; "
            "use SecuredDataset.update()"
        )

    # -- snapshot/restore for atomic multi-graph operations ----------------

    def snapshot(self) -> _Snapshot:
        """Capture all graph registrations and quads in the wrapped store."""
        identifiers = self._existing_graphs()
        quads: list[tuple[Node, Node, Node, IdentifiedNode]] = []
        for identifier in identifiers:
            view = Graph(self._store, identifier=identifier)
            for (subject, predicate, object_), _contexts in self._store.triples(
                (None, None, None), view
            ):
                quads.append((subject, predicate, object_, identifier))
        return identifiers, quads

    def restore(self, snapshot: _Snapshot) -> None:
        """Reset the wrapped store to a state captured by :meth:`snapshot`."""
        identifiers, quads = snapshot
        for graph_context in list(self._store.contexts()):
            self._store.remove_graph(graph_context)
        views = {
            identifier: Graph(self._store, identifier=identifier)
            for identifier in identifiers
        }
        for view in views.values():
            self._store.add_graph(view)
        for subject, predicate, object_, identifier in quads:
            self._store.add(
                (subject, predicate, object_), views[identifier], quoted=False
            )


# ---------------------------------------------------------------------------
# SPARQL algebra static analysis
# ---------------------------------------------------------------------------


def _algebra_nodes(node: Any) -> Iterator[CompValue]:
    """Yield every :class:`CompValue` in an algebra tree."""
    if isinstance(node, CompValue):
        yield node
        for value in node.values():
            yield from _algebra_nodes(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _algebra_nodes(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _algebra_nodes(value)


_GRAPH_PATTERN_NAMES = frozenset({"Graph", "GraphGraphPattern"})


def _is_concrete_graph(term: Any) -> TypeGuard[IdentifiedNode]:
    """Whether an algebra term statically names a graph (IRI or BNode)."""
    return isinstance(term, (URIRef, BNode))


def _is_graph_pattern(node: CompValue) -> bool:
    """Whether an algebra node is a GRAPH pattern."""
    return node.name in _GRAPH_PATTERN_NAMES


# ---------------------------------------------------------------------------
# Secured dataset
# ---------------------------------------------------------------------------


class SecuredDataset(Dataset):
    """A :class:`~rdflib.graph.Dataset` enforcing a :class:`GraphAccessPolicy`.

    Args:
        policy: The graph permission policy to enforce.
        store: An optional context- and graph-aware store to wrap.  Defaults
            to a new in-memory ``Memory`` store.
        identity: A fixed caller identity for this dataset.  If omitted, the
            identity set with :func:`principal` for the current context is
            used.
        default_union: See :class:`~rdflib.graph.Dataset`.
        default_graph_base: See :class:`~rdflib.graph.Dataset`.

    Every entry point -- direct ``add``/``addN``/``remove``,
    ``remove_graph``, ``parse``, ``query`` and ``update`` -- is authorised for
    the same identity.  ``parse`` and SPARQL ``update`` requests are checked
    before they run and rolled back if a later step is refused, so a refused
    request never leaves partial modifications behind.
    """

    def __init__(
        self,
        policy: GraphAccessPolicy,
        *,
        store: Store | None = None,
        identity: Principal | None = None,
        default_union: bool = False,
        default_graph_base: str | None = None,
    ) -> None:
        if store is None:
            from rdflib import plugin

            store = cast(Store, plugin.get("Memory", Store)())
        if isinstance(store, GuardedStore):
            raise ValueError("store is already wrapped in a GuardedStore")
        self._policy = policy
        identity_provider = (
            current_principal
            if identity is None
            else (lambda: cast(Principal, identity))
        )
        self._guarded_store = GuardedStore(store, policy, identity_provider)
        super().__init__(
            store=self._guarded_store,
            default_union=default_union,
            default_graph_base=default_graph_base,
        )

    @property
    def policy(self) -> GraphAccessPolicy:
        return self._policy

    # -- authorisation helpers ---------------------------------------------

    def _require(self, graph: GraphRef, permission: Permission) -> None:
        principal = self._guarded_store._principal()
        if not self._policy.permits(principal, graph, permission):
            raise AccessDeniedError(principal, graph, permission)

    def _graph_refs(self, reference: Any) -> list[IdentifiedNode]:
        """Resolve a SPARQL graph reference ("DEFAULT"/"NAMED"/"ALL"/IRI)."""
        default = self.default_graph.identifier
        if reference == "DEFAULT":
            return [default]
        if reference in ("NAMED", "ALL"):
            identifiers = self._guarded_store._existing_graphs()
            if reference == "NAMED":
                identifiers = [g for g in identifiers if g != default]
            return identifiers
        return [cast(IdentifiedNode, reference)]

    def _authorise_query(self, query: Query) -> None:
        algebra = query.algebra
        dataset_clause = getattr(algebra, "datasetClause", None)
        if dataset_clause:
            for clause in dataset_clause:
                if clause.default is not None:
                    self._require(clause.default, Permission.READ)
                if clause.named is not None:
                    self._require(clause.named, Permission.READ)
        for node in _algebra_nodes(algebra):
            if _is_graph_pattern(node) and _is_concrete_graph(node.term):
                self._require(node.term, Permission.READ)

    def _authorise_update_operation(self, operation: CompValue) -> None:
        name = operation.name
        default = self.default_graph.identifier

        def concrete_keys(quads: Any) -> list[IdentifiedNode]:
            return [key for key in quads if _is_concrete_graph(key)]

        if name == "InsertData":
            if operation.triples:
                self._require(default, Permission.WRITE)
            for key in concrete_keys(operation.quads):
                self._require(key, Permission.WRITE)
        elif name == "DeleteData":
            if operation.triples:
                self._require(default, Permission.DELETE)
            for key in concrete_keys(operation.quads):
                self._require(key, Permission.DELETE)
        elif name == "DeleteWhere":
            self._require(default, Permission.READ)
            if operation.triples:
                self._require(default, Permission.DELETE)
            for key in concrete_keys(operation.quads):
                self._require(key, Permission.READ)
                self._require(key, Permission.DELETE)
        elif name == "Modify":
            # WHERE clause read scope.
            if operation.using:
                for using in operation.using:
                    if using.default is not None:
                        self._require(using.default, Permission.READ)
                    if using.named is not None:
                        self._require(using.named, Permission.READ)
            elif operation.withClause is not None:
                self._require(operation.withClause, Permission.READ)
            else:
                self._require(default, Permission.READ)
            for node in _algebra_nodes(operation.where):
                if _is_graph_pattern(node) and _is_concrete_graph(node.term):
                    self._require(node.term, Permission.READ)
            # Mutation targets.
            target = (
                operation.withClause if operation.withClause is not None else default
            )
            if operation.insert is not None:
                if operation.insert.triples:
                    self._require(target, Permission.WRITE)
                for key in concrete_keys(operation.insert.quads):
                    self._require(key, Permission.WRITE)
            if operation.delete is not None:
                if operation.delete.triples:
                    self._require(target, Permission.DELETE)
                for key in concrete_keys(operation.delete.quads):
                    self._require(key, Permission.DELETE)
        elif name in ("Clear", "Drop"):
            for identifier in self._graph_refs(operation.graphiri):
                # Bulk forms (ALL/NAMED) can only work over graphs the
                # caller can both enumerate and delete.
                if operation.graphiri in ("ALL", "NAMED"):
                    self._require(identifier, Permission.READ)
                self._require(identifier, Permission.DELETE)
        elif name == "Create":
            self._require(operation.graphiri, Permission.WRITE)
        elif name == "Load":
            target = operation.graphiri if operation.graphiri is not None else default
            self._require(target, Permission.WRITE)
        elif name in ("Add", "Move", "Copy"):
            source_ref, destination_ref = operation.graph
            source = default if source_ref == "DEFAULT" else source_ref
            destination = default if destination_ref == "DEFAULT" else destination_ref
            self._require(source, Permission.READ)
            self._require(destination, Permission.WRITE)
            if name in ("Move", "Copy"):
                self._require(destination, Permission.DELETE)
            if name == "Move":
                self._require(source, Permission.DELETE)
        else:
            raise ValueError("unknown SPARQL update operation: %s" % name)

    def _authorise_update(self, update: Update) -> None:
        for operation in update.algebra:
            self._authorise_update_operation(operation)

    def _readable_graphs(
        self, triple: tuple[Node, Node, Node] | None = None
    ) -> Generator[Graph, None, None]:
        default = self.default_graph.identifier
        seen: set[IdentifiedNode] = set()
        for graph_context in super(Dataset, self).contexts(triple):
            seen.add(graph_context.identifier)
            if self._guarded_store._readable(graph_context.identifier):
                yield graph_context
        if default not in seen and self._guarded_store._readable(default):
            yield self.get_context(default)

    def graphs(
        self, triple: tuple[Node, Node, Node] | None = None
    ) -> Generator[Graph, None, None]:
        """Iterate graphs the caller may read.

        Overrides :meth:`Dataset.graphs`, which always synthesises a default
        graph entry even when the caller may not read it.  Graph discovery is
        filtered just like quad discovery.
        """
        yield from self._readable_graphs(triple)

    def contexts(
        self, triple: tuple[Node, Node, Node] | None = None
    ) -> Generator[Graph, None, None]:
        warnings.warn(
            "Dataset.contexts is deprecated, use Dataset.graphs instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        yield from self._readable_graphs(triple)

    # -- guarded entry points ----------------------------------------------

    def query(
        self,
        query_object: str | Query,
        processor: str | rdflib_query.Processor = "sparql",
        result: str | type[rdflib_query.Result] = "sparql",
        initNs: Mapping[str, Any] | None = None,  # noqa: N803
        initBindings: Mapping[str, Identifier] | None = None,  # noqa: N803
        use_store_provided: bool = True,
        **kwargs: Any,
    ) -> rdflib_query.Result:
        if isinstance(query_object, str):
            query_object = translateQuery(parseQuery(query_object))
        self._authorise_query(query_object)
        return super().query(
            query_object,
            processor=processor,
            result=result,
            initNs=initNs,
            initBindings=initBindings,
            use_store_provided=use_store_provided,
            **kwargs,
        )

    def update(
        self,
        update_object: str | Update,
        processor: str | rdflib_query.UpdateProcessor = "sparql",
        initNs: Mapping[str, Any] | None = None,  # noqa: N803
        initBindings: Mapping[str, Identifier] | None = None,  # noqa: N803
        use_store_provided: bool = True,
        **kwargs: Any,
    ) -> None:
        if isinstance(update_object, str):
            update_object = translateUpdate(parseUpdate(update_object), initNs=initNs)
        # Static pre-flight: refuse the whole request before any effect.
        self._authorise_update(update_object)
        snapshot = self._guarded_store.snapshot()
        violations: list[AccessDeniedError] = []
        token = _violation_log.set(violations)
        try:
            super().update(
                update_object,
                processor=processor,
                initNs=initNs,
                initBindings=initBindings,
                use_store_provided=use_store_provided,
                **kwargs,
            )
        except BaseException:
            _violation_log.reset(token)
            if violations:
                self._guarded_store.restore(snapshot)
                raise violations[0] from None
            raise
        _violation_log.reset(token)
        if violations:
            self._guarded_store.restore(snapshot)
            raise violations[0]

    def parse(self, *args: Any, **kwargs: Any) -> SecuredDataset:
        # A quad document (TriG/N-Quads/...) may target several graphs,
        # including ones discovered while parsing, so record violations and
        # roll the whole parse back when one is refused.
        snapshot = self._guarded_store.snapshot()
        violations: list[AccessDeniedError] = []
        token = _violation_log.set(violations)
        try:
            super().parse(*args, **kwargs)
        except BaseException:
            _violation_log.reset(token)
            if violations:
                self._guarded_store.restore(snapshot)
                raise violations[0] from None
            raise
        _violation_log.reset(token)
        if violations:
            self._guarded_store.restore(snapshot)
            raise violations[0]
        return self
