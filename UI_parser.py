"""Loading, reference resolution, traversal, evidence normalization and validation."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path as FilePath
from typing import Any, TextIO

from models import (
    Edge,
    Node,
    Path,
    Publication,
    PublicationEvidence,
    QueryResult,
    ResolvedPath,
    ResultPath,
    ScoreComponent,
    Trial,
    ValidationReport,
)


class TranslatorParseError(ValueError):
    """Raised when input is invalid JSON or violates the format in strict mode."""


def _tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _records(value: Any, section: str) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Accept both ID-keyed objects and arrays of records for forward compatibility."""
    if isinstance(value, Mapping):
        for key, record in value.items():
            if isinstance(record, Mapping):
                yield str(key), record
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, record in enumerate(value):
            if isinstance(record, Mapping):
                yield str(record.get("id", index)), record
        return
    if value is not None:
        raise TranslatorParseError(f"Section {section!r} must be an object or an array")


def _node(key: str, raw: Mapping[str, Any]) -> Node:
    return Node(
        id=str(raw.get("id", key)), names=_tuple(raw.get("names")),
        types=_tuple(raw.get("types")), curies=_tuple(raw.get("curies")),
        descriptions=_tuple(raw.get("descriptions")), species=raw.get("species"),
        provenance=_tuple(raw.get("provenance")), synonyms=_tuple(raw.get("synonyms")),
        aras=_tuple(raw.get("aras")), raw=raw,
    )


def _edge(key: str, raw: Mapping[str, Any]) -> Edge:
    return Edge(
        id=str(raw.get("id", key)), subject=str(raw.get("subject", "")),
        object=str(raw.get("object", "")), predicate=raw.get("predicate"),
        predicate_url=raw.get("predicate_url"), knowledge_level=raw.get("knowledge_level"),
        provenance=_tuple(raw.get("provenance")), publications=_mapping(raw.get("publications")),
        trials=tuple(str(x) for x in _tuple(raw.get("trials"))),
        aras=_tuple(raw.get("aras")), description=raw.get("description"), raw=raw,
    )


def _path(key: str, raw: Mapping[str, Any]) -> Path:
    return Path(id=str(raw.get("id", key)), subgraph=tuple(str(x) for x in _tuple(raw.get("subgraph"))),
                aras=_tuple(raw.get("aras")), raw=raw)


def _result(key: str, raw: Mapping[str, Any]) -> QueryResult:
    components = []
    for item in _tuple(raw.get("scoreComponents")):
        if isinstance(item, Mapping):
            components.append(ScoreComponent(
                confidence=_float(item.get("confidence")),
                clinical_evidence=_float(item.get("clinical_evidence")),
                novelty=_float(item.get("novelty")),
                normalized_score=_float(item.get("normalized_score")), raw=item,
            ))
    return QueryResult(
        id=str(raw.get("id", key)), subject=str(raw.get("subject", "")),
        object=str(raw.get("object", "")),
        path_ids=tuple(str(x) for x in _tuple(raw.get("paths"))),
        score=_float(raw.get("score")), name=raw.get("drug_name") or raw.get("name"),
        score_components=tuple(components), raw=raw,
    )


def _publication(key: str, raw: Mapping[str, Any]) -> Publication:
    return Publication(id=str(raw.get("id", key)), url=raw.get("url"),
                       source=_mapping(raw.get("source")), raw=raw)


def _trial(key: str, raw: Mapping[str, Any]) -> Trial:
    return Trial(id=str(raw.get("id", key)), title=raw.get("title"), url=raw.get("url"),
                 phase=raw.get("phase"), start_date=raw.get("start_date"),
                 status=raw.get("status"), type=raw.get("type"), size=raw.get("size"), raw=raw)


@dataclass(slots=True)
class TranslatorExport:
    meta: Mapping[str, Any]
    results: tuple[QueryResult, ...]
    nodes: dict[str, Node]
    edges: dict[str, Edge]
    paths: dict[str, Path]
    publications: dict[str, Publication]
    trials: dict[str, Trial]
    raw: Mapping[str, Any]

    def resolve_path(self, path_or_id: Path | str) -> ResolvedPath:
        path = self.paths[path_or_id] if isinstance(path_or_id, str) else path_or_id
        if not path.subgraph or len(path.subgraph) % 2 == 0:
            raise TranslatorParseError(
                f"Path {path.id!r} must contain an odd node/edge/.../node sequence"
            )
        try:
            nodes = tuple(self.nodes[x] for x in path.subgraph[0::2])
            edges = tuple(self.edges[x] for x in path.subgraph[1::2])
        except KeyError as exc:
            raise TranslatorParseError(f"Path {path.id!r} references missing ID {exc.args[0]!r}") from exc
        steps = []
        for index, (source, edge, target) in enumerate(zip(nodes, edges, nodes[1:])):
            if edge.subject == source.id and edge.object == target.id:
                direction = "forward"
            elif edge.subject == target.id and edge.object == source.id:
                direction = "reverse"
            else:
                direction = "inconsistent"
            steps.append(ResultPath(index, source, edge, target, direction))
        return ResolvedPath(path, nodes, edges, tuple(steps))

    def paths_for_result(self, result_or_id: QueryResult | str) -> Iterator[ResolvedPath]:
        result = result_or_id if isinstance(result_or_id, QueryResult) else self.result_by_id(result_or_id)
        for path_id in result.path_ids:
            yield self.resolve_path(path_id)

    def result_by_id(self, result_id: str) -> QueryResult:
        for result in self.results:
            if result.id == result_id:
                return result
        raise KeyError(result_id)

    def iter_publication_evidence(self, edge_or_id: Edge | str | None = None) -> Iterator[PublicationEvidence]:
        edges = (self.edges[edge_or_id],) if isinstance(edge_or_id, str) else ((edge_or_id,) if edge_or_id else self.edges.values())
        for edge in edges:
            for evidence_class, entries in edge.publications.items():
                for item in _tuple(entries):
                    if isinstance(item, str):
                        publication_id, infores, support, raw = item, None, {}, {"id": item}
                    elif isinstance(item, Mapping):
                        publication_id = str(item.get("id", ""))
                        infores, support, raw = item.get("infores"), _mapping(item.get("support")), item
                    else:
                        continue
                    yield PublicationEvidence(
                        edge_id=edge.id, evidence_class=str(evidence_class),
                        publication_id=publication_id, infores=infores,
                        support_text=support.get("text"), support_subject=support.get("subject"),
                        support_object=support.get("object"),
                        publication=self.publications.get(publication_id), raw=raw,
                    )

    def validate(self) -> ValidationReport:
        report = ValidationReport()
        for key, collection in (("nodes", self.nodes), ("edges", self.edges), ("paths", self.paths),
                                ("publications", self.publications), ("trials", self.trials)):
            for map_id, record in collection.items():
                if map_id != record.id:
                    report.add("warning", "id_key_mismatch", f"{key}.{map_id}",
                               f"Mapping key {map_id!r} differs from record ID {record.id!r}")
        for edge in self.edges.values():
            for role, node_id in (("subject", edge.subject), ("object", edge.object)):
                if node_id not in self.nodes:
                    report.add("error", "missing_node", f"edges.{edge.id}.{role}", f"Unknown node {node_id!r}")
            for trial_id in edge.trials:
                if trial_id not in self.trials:
                    report.add("warning", "missing_trial", f"edges.{edge.id}.trials", f"Unknown trial {trial_id!r}")
        for path in self.paths.values():
            if not path.subgraph or len(path.subgraph) % 2 == 0:
                report.add("error", "invalid_path_shape", f"paths.{path.id}.subgraph",
                           "Expected an odd node/edge/.../node sequence")
                continue
            for index, identifier in enumerate(path.subgraph):
                expected = self.nodes if index % 2 == 0 else self.edges
                if identifier not in expected:
                    kind = "node" if index % 2 == 0 else "edge"
                    report.add("error", f"missing_{kind}", f"paths.{path.id}.subgraph[{index}]",
                               f"Unknown {kind} {identifier!r}")
            try:
                resolved = self.resolve_path(path)
            except TranslatorParseError:
                continue
            for step in resolved.steps:
                if step.direction == "inconsistent":
                    report.add("error", "edge_path_mismatch", f"paths.{path.id}.subgraph",
                               f"Edge {step.edge.id!r} does not connect adjacent nodes "
                               f"{step.source.id!r} and {step.target.id!r}")
        result_ids: set[str] = set()
        for index, result in enumerate(self.results):
            location = f"results[{index}]"
            if result.id in result_ids:
                report.add("warning", "duplicate_result_id", location, f"Duplicate result ID {result.id!r}")
            result_ids.add(result.id)
            if result.subject and result.subject not in self.nodes:
                report.add("warning", "missing_result_subject", location, f"Unknown subject {result.subject!r}")
            if result.object and result.object not in self.nodes:
                report.add("warning", "missing_result_object", location, f"Unknown object {result.object!r}")
            for path_id in result.path_ids:
                if path_id not in self.paths:
                    report.add("error", "missing_path", f"{location}.paths", f"Unknown path {path_id!r}")
        for evidence in self.iter_publication_evidence():
            if evidence.publication_id and evidence.publication is None:
                report.add("warning", "missing_publication", f"edges.{evidence.edge_id}.publications",
                           f"No global record for publication {evidence.publication_id!r}")
        return report

    def summary(self) -> dict[str, Any]:
        report = self.validate()
        return {
            "meta": dict(self.meta), "results": len(self.results), "nodes": len(self.nodes),
            "edges": len(self.edges), "paths": len(self.paths),
            "publications": len(self.publications), "trials": len(self.trials),
            "publication_evidence": sum(1 for _ in self.iter_publication_evidence()),
            "validation_errors": len(report.errors), "validation_warnings": len(report.warnings),
        }


def load_export(source: str | FilePath | TextIO | Mapping[str, Any], *, strict: bool = False) -> TranslatorExport:
    """Load a path, open text stream, or already-decoded mapping.

    Unknown fields are preserved in each model's ``raw`` mapping. In strict mode,
    cross-reference or path-consistency errors raise ``TranslatorParseError``.
    """
    try:
        if isinstance(source, Mapping):
            raw = source
        elif hasattr(source, "read"):
            raw = json.load(source)  # type: ignore[arg-type]
        else:
            with FilePath(source).open("r", encoding="utf-8") as stream:
                raw = json.load(stream)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise TranslatorParseError(str(exc)) from exc
    if not isinstance(raw, Mapping):
        raise TranslatorParseError("Top-level JSON value must be an object")

    export = TranslatorExport(
        meta=_mapping(raw.get("meta")),
        results=tuple(_result(key, item) for key, item in _records(raw.get("results", ()), "results")),
        nodes={item.id: item for key, value in _records(raw.get("nodes", {}), "nodes") if (item := _node(key, value))},
        edges={item.id: item for key, value in _records(raw.get("edges", {}), "edges") if (item := _edge(key, value))},
        paths={item.id: item for key, value in _records(raw.get("paths", {}), "paths") if (item := _path(key, value))},
        publications={item.id: item for key, value in _records(raw.get("publications", {}), "publications") if (item := _publication(key, value))},
        trials={item.id: item for key, value in _records(raw.get("trials", {}), "trials") if (item := _trial(key, value))},
        raw=raw,
    )
    report = export.validate()
    if strict and not report.ok:
        preview = "; ".join(f"{x.location}: {x.message}" for x in report.errors[:5])
        raise TranslatorParseError(f"Validation failed with {len(report.errors)} error(s): {preview}")
    return export
