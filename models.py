"""Typed, format-oriented models with lossless access to source records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

JSONMapping = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    names: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    curies: tuple[str, ...] = ()
    descriptions: tuple[str, ...] = ()
    species: Any = None
    provenance: tuple[Any, ...] = ()
    synonyms: tuple[str, ...] = ()
    aras: tuple[str, ...] = ()
    raw: JSONMapping = field(default_factory=dict, repr=False, compare=False)

    @property
    def name(self) -> str | None:
        return self.names[0] if self.names else None


@dataclass(frozen=True, slots=True)
class Edge:
    id: str
    subject: str
    object: str
    predicate: str | None = None
    predicate_url: str | None = None
    knowledge_level: str | None = None
    provenance: tuple[Any, ...] = ()
    publications: JSONMapping = field(default_factory=dict)
    trials: tuple[str, ...] = ()
    aras: tuple[str, ...] = ()
    description: str | None = None
    raw: JSONMapping = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Path:
    id: str
    subgraph: tuple[str, ...]
    aras: tuple[str, ...] = ()
    raw: JSONMapping = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ScoreComponent:
    confidence: float | None = None
    clinical_evidence: float | None = None
    novelty: float | None = None
    normalized_score: float | None = None
    raw: JSONMapping = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class QueryResult:
    id: str
    subject: str
    object: str
    path_ids: tuple[str, ...]
    score: float | None = None
    name: str | None = None
    score_components: tuple[ScoreComponent, ...] = ()
    raw: JSONMapping = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Publication:
    id: str
    url: str | None = None
    source: JSONMapping = field(default_factory=dict)
    raw: JSONMapping = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Trial:
    id: str
    title: str | None = None
    url: str | None = None
    phase: Any = None
    start_date: str | None = None
    status: str | None = None
    type: str | None = None
    size: Any = None
    raw: JSONMapping = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ResultPath:
    """One traversed edge and the two adjacent nodes in a resolved path."""

    index: int
    source: Node
    edge: Edge
    target: Node
    direction: str  # "forward", "reverse", or "inconsistent"


@dataclass(frozen=True, slots=True)
class ResolvedPath:
    path: Path
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    steps: tuple[ResultPath, ...]


@dataclass(frozen=True, slots=True)
class PublicationEvidence:
    edge_id: str
    evidence_class: str
    publication_id: str
    infores: str | None
    support_text: str | None
    support_subject: Any = None
    support_object: Any = None
    publication: Publication | None = None
    raw: JSONMapping = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: str  # "error" or "warning"
    code: str
    location: str
    message: str


@dataclass(slots=True)
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(x for x in self.issues if x.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(x for x in self.issues if x.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, severity: str, code: str, location: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity, code, location, message))

