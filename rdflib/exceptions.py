"""
TODO:
"""

from __future__ import annotations

__all__ = [
    "Error",
    "ParserError",
    "SerializationVerificationError",
    "UniquenessError",
]


from typing import Any


class Error(Exception):
    """Base class for rdflib exceptions."""

    def __init__(self, msg: str | None = None):
        Exception.__init__(self, msg)
        self.msg = msg


class ParserError(Error):
    """RDF Parser error."""

    def __init__(self, msg: str):
        Error.__init__(self, msg)
        self.msg: str = msg

    def __str__(self) -> str:
        return self.msg


class UniquenessError(Error):
    """A uniqueness assumption was made in the context, and that is not true"""

    def __init__(self, values: Any):
        Error.__init__(
            self,
            "\
Uniqueness assumption is not fulfilled. Multiple values are: %s"
            % values,
        )


class SerializationVerificationError(Error):
    """Serializing a graph to a file could not be completed safely.

    This is raised by [`rdflib.tools.safe_serialize`][rdflib.tools.safe_serialize]
    when writing, reading back or verifying the side file fails, or when the
    read-back graph does not contain the same statements (and named graphs)
    as the source graph. The destination file is never replaced in this case,
    so a previously published file is left untouched.

    Attributes:
        stage: Which step failed. One of ``"write"``, ``"readback"``,
            ``"verify"`` or ``"publish"``.
    """

    def __init__(self, msg: str, stage: str | None = None):
        Error.__init__(self, msg)
        self.stage = stage
