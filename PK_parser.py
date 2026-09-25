"""Fetch and parse all Translator ARS results associated with a message PK.

This module is the ARS/TRAPI companion to ``UI_parser.py``.  It returns the
same ``TranslatorExport`` object and uses the same model/field names, while
preserving every native ARS/TRAPI record in the models' ``raw`` attributes.

Public API
----------
``load_pk(pk)``
    Fetch a PK from ARS (and its child messages) and return TranslatorExport.

``parse_ars_response(payload)``
    Parse an already downloaded ARS response, useful for tests/offline work.

``to_ui_dict(export)``
    Convert TranslatorExport to a JSON-serializable dictionary using the same
    top-level and record names as a Translator UI full-results export.

The module uses only Python's standard library.  Keep it beside
``UI_parser.py`` and ``models.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from models import Edge, Node, Path, Publication, QueryResult, ScoreComponent, Trial
from UI_parser import TranslatorExport, TranslatorParseError


DEFAULT_ARS_URL = "https://ars-prod.transltr.io/ars/api/messages"
_PK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{5,127}$")
_PMID_RE = re.compile(r"\bPMID[: ]?(\d+)\b", re.IGNORECASE)
_PMC_RE = re.compile(r"\bPMC[: ]?(\d+)\b", re.IGNORECASE)
_NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


def _tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float(value: Any) -> float | None:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _json_id(*parts: Any, size: int = 16) -> str:
    encoded = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:size]


def _walk(value: Any) -> Iterator[Any]:
    """Depth-first traversal of JSON-like data."""
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk(child)


def _strings(value: Any) -> Iterator[str]:
    for item in _walk(value):
        if isinstance(item, str):
            yield item


def _attributes(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(x for x in _tuple(record.get("attributes")) if isinstance(x, Mapping))


def _attribute_label(attribute: Mapping[str, Any]) -> str:
    return " ".join(
        str(attribute.get(key, ""))
        for key in ("attribute_type_id", "original_attribute_name", "value_type_id")
    ).lower()


def _attribute_values(record: Mapping[str, Any], *terms: str) -> Iterator[Any]:
    for attribute in _attributes(record):
        label = _attribute_label(attribute)
        if any(term.lower() in label for term in terms):
            yield attribute.get("value")


def _first_attribute(record: Mapping[str, Any], *terms: str) -> Any:
    return next(_attribute_values(record, *terms), None)


def _ids_in(value: Any, pattern: re.Pattern[str], prefix: str = "") -> tuple[str, ...]:
    found: dict[str, None] = {}
    for text in _strings(value):
        for match in pattern.finditer(text):
            token = match.group(1) if match.lastindex else match.group(0)
            found[prefix + token.upper()] = None
    return tuple(found)


def _resource_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    found: dict[str, None] = {}
    for source in _tuple(record.get("sources")):
        if isinstance(source, Mapping):
            for key in ("resource_id", "upstream_resource_ids"):
                for resource in _tuple(source.get(key)):
                    if resource:
                        found[str(resource)] = None
    resource = record.get("resource_id")
    if resource:
        found[str(resource)] = None
    return tuple(found)


def _extract_trapi_responses(payload: Any) -> list[Mapping[str, Any]]:
    """Find complete TRAPI Responses inside old and new ARS envelopes."""
    found: list[Mapping[str, Any]] = []
    seen_payloads: set[str] = set()
    for candidate in _walk(payload):
        if not isinstance(candidate, Mapping):
            continue
        message = candidate.get("message")
        if isinstance(message, str) and message.lstrip().startswith("{"):
            try:
                message = json.loads(message)
                candidate = {**candidate, "message": message}
            except json.JSONDecodeError:
                continue
        if not isinstance(message, Mapping):
            continue
        if not any(key in message for key in ("results", "knowledge_graph", "query_graph")):
            continue
        identity = _json_id(message, size=32)
        if identity not in seen_payloads:
            seen_payloads.add(identity)
            found.append(candidate)
    if found:
        return found

    # Also accept a bare TRAPI Message rather than a full TRAPI Response.
    for candidate in _walk(payload):
        if not isinstance(candidate, Mapping):
            continue
        if "results" not in candidate or "knowledge_graph" not in candidate:
            continue
        identity = _json_id(candidate, size=32)
        if identity not in seen_payloads:
            seen_payloads.add(identity)
            found.append({"message": candidate})
    return found


def _child_pks(payload: Any, current_pk: str) -> tuple[str, ...]:
    """Extract explicit child-message identifiers without treating biomedical IDs as PKs."""
    children: dict[str, None] = {}
    child_keys = {"children", "child", "child_pk", "child_pks", "children_ids", "child_messages"}

    def visit(value: Any, parent_key: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_l = str(key).lower()
                if key_l in child_keys or ("child" in key_l and ("pk" in key_l or "message" in key_l)):
                    collect(child)
                visit(child, key_l)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child, parent_key)

    def collect(value: Any) -> None:
        if isinstance(value, str) and _PK_RE.fullmatch(value) and value != current_pk:
            children[value] = None
        elif isinstance(value, Mapping):
            for key in ("pk", "id", "message_id"):
                item = value.get(key)
                if isinstance(item, str) and _PK_RE.fullmatch(item) and item != current_pk:
                    children[item] = None
            for nested in value.values():
                if isinstance(nested, (Mapping, list, tuple)):
                    collect(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                collect(item)

    visit(payload)
    return tuple(children)


class ARSClient:
    """Small standard-library client for the Translator ARS message API."""

    def __init__(
        self,
        base_url: str = DEFAULT_ARS_URL,
        *,
        timeout: float = 120.0,
        retries: int = 2,
        user_agent: str = "translator-pk-parser/1.0",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.user_agent = user_agent

    def fetch(self, pk: str) -> Mapping[str, Any] | list[Any]:
        if not _PK_RE.fullmatch(pk):
            raise TranslatorParseError(f"Invalid ARS message PK: {pk!r}")
        request = Request(
            f"{self.base_url}/{quote(pk, safe='')}",
            headers={"Accept": "application/json", "User-Agent": self.user_agent},
            method="GET",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                if not isinstance(payload, (Mapping, list)):
                    raise TranslatorParseError("ARS returned a non-object/non-array JSON value")
                return payload
            except HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    break
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
            time.sleep(0.5 * (2**attempt))
        raise TranslatorParseError(f"Could not retrieve ARS PK {pk!r}: {last_error}") from last_error

    def fetch_tree(self, pk: str, *, follow_children: bool = True) -> list[Any]:
        """Fetch the root message and every explicitly referenced child exactly once."""
        queue = deque([pk])
        visited: set[str] = set()
        payloads: list[Any] = []
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            payload = self.fetch(current)
            payloads.append(payload)
            if follow_children:
                queue.extend(child for child in _child_pks(payload, current) if child not in visited)
        return payloads


def _make_node(node_id: str, raw: Mapping[str, Any], aras: tuple[str, ...]) -> Node:
    name = raw.get("name")
    descriptions = tuple(
        str(x) for value in _attribute_values(raw, "description") for x in _tuple(value) if x
    )
    synonyms = tuple(
        str(x) for value in _attribute_values(raw, "synonym", "alias") for x in _tuple(value) if x
    )
    equivalent = tuple(
        str(x)
        for value in _attribute_values(raw, "equivalent", "xref", "same_as")
        for x in _tuple(value)
        if x
    )
    provenance = tuple(
        source
        for attribute in _attributes(raw)
        for source in _tuple(attribute.get("attribute_source"))
        if source
    )
    return Node(
        id=node_id,
        names=(str(name),) if name else (),
        types=tuple(str(x) for x in _tuple(raw.get("categories")) if x),
        curies=tuple(dict.fromkeys((node_id, *equivalent))),
        descriptions=descriptions,
        species=_first_attribute(raw, "in_taxon", "species", "organism"),
        provenance=provenance,
        synonyms=synonyms,
        aras=aras,
        raw=raw,
    )


def _publication_url(identifier: str) -> str | None:
    if identifier.startswith("PMID:"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{identifier.split(':', 1)[1]}/"
    if identifier.startswith("PMC"):
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{identifier}/"
    if identifier.upper().startswith("DOI:"):
        return f"https://doi.org/{identifier.split(':', 1)[1]}"
    return None


def _evidence_class(edge: Mapping[str, Any], infores: str | None) -> str:
    resource = (infores or "").lower()
    if "semmed" in resource or "text-mining" in resource or "text_mining" in resource:
        return "ml"
    level = str(edge.get("knowledge_level") or _first_attribute(edge, "knowledge_level") or "").lower()
    return "trusted" if "trusted" in level else "inferred"


def _make_edge(
    edge_id: str,
    raw: Mapping[str, Any],
    aras: tuple[str, ...],
) -> tuple[Edge, dict[str, Publication], dict[str, Trial]]:
    sources = tuple(x for x in _tuple(raw.get("sources")) if isinstance(x, Mapping))
    infores = next((str(x.get("resource_id")) for x in sources if x.get("resource_id")), None)
    publication_ids = tuple(dict.fromkeys((
        *_ids_in(raw, _PMID_RE, "PMID:"),
        *_ids_in(raw, _PMC_RE, "PMC"),
    )))
    trial_ids = _ids_in(raw, _NCT_RE)
    support = _first_attribute(raw, "supporting_text", "support text", "sentence")
    support_text = support if isinstance(support, str) else None
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for publication_id in publication_ids:
        grouped[_evidence_class(raw, infores)].append(
            {
                "id": publication_id,
                "support": {"text": support_text, "subject": None, "object": None}
                if support_text else None,
                "infores": infores,
            }
        )
    predicate = raw.get("predicate")
    predicate_url = None
    if isinstance(predicate, str) and predicate.startswith("biolink:"):
        predicate_url = "https://biolink.github.io/biolink-model/" + predicate.split(":", 1)[1] + "/"
    publications = {
        identifier: Publication(
            id=identifier,
            url=_publication_url(identifier),
            source={"name": infores, "knowledge_level": raw.get("knowledge_level")},
            raw={"id": identifier, "derived_from_edge": edge_id},
        )
        for identifier in publication_ids
    }
    trials = {
        identifier: Trial(
            id=identifier,
            url=f"https://clinicaltrials.gov/study/{identifier}",
            raw={"id": identifier, "derived_from_edge": edge_id},
        )
        for identifier in trial_ids
    }
    edge = Edge(
        id=edge_id,
        subject=str(raw.get("subject", "")),
        object=str(raw.get("object", "")),
        predicate=str(predicate) if predicate is not None else None,
        predicate_url=predicate_url,
        knowledge_level=raw.get("knowledge_level") or _first_attribute(raw, "knowledge_level"),
        provenance=sources,
        publications=dict(grouped),
        trials=trial_ids,
        aras=aras,
        description=_first_attribute(raw, "description"),
        raw=raw,
    )
    return edge, publications, trials


def _bound_ids(binding: Any) -> tuple[str, ...]:
    ids: list[str] = []
    for item in _tuple(binding):
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, Mapping):
            for identifier in _tuple(item.get("id") or item.get("ids")):
                if identifier:
                    ids.append(str(identifier))
    return tuple(dict.fromkeys(ids))


def _result_endpoints(
    result: Mapping[str, Any], query_graph: Mapping[str, Any]
) -> tuple[str, str]:
    bindings = _mapping(result.get("node_bindings"))
    qnodes = _mapping(query_graph.get("nodes"))
    ordered_keys = [key for key in qnodes if key in bindings] or list(bindings)
    if not ordered_keys:
        return "", ""

    # Prefer query-graph leaf nodes. Their insertion order normally preserves
    # the user-selected source and target used by the UI.
    degree: dict[str, int] = defaultdict(int)
    for qedge in _mapping(query_graph.get("edges")).values():
        if isinstance(qedge, Mapping):
            degree[str(qedge.get("subject", ""))] += 1
            degree[str(qedge.get("object", ""))] += 1
    leaves = [key for key in ordered_keys if degree.get(key, 0) <= 1]
    endpoints = leaves if len(leaves) >= 2 else ordered_keys
    subject_ids = _bound_ids(bindings.get(endpoints[0]))
    object_ids = _bound_ids(bindings.get(endpoints[-1]))
    return (subject_ids[0] if subject_ids else "", object_ids[0] if object_ids else "")


def _simple_paths(
    edge_ids: Iterable[str],
    edges: Mapping[str, Edge],
    subject: str,
    object_: str,
) -> list[tuple[str, ...]]:
    """Return every simple alternating node/edge path between two endpoints."""
    usable = tuple(dict.fromkeys(x for x in edge_ids if x in edges))
    if not usable or not subject or not object_:
        return []
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge_id in usable:
        edge = edges[edge_id]
        adjacency[edge.subject].append((edge_id, edge.object))
        adjacency[edge.object].append((edge_id, edge.subject))
    output: list[tuple[str, ...]] = []

    def visit(node: str, used_edges: set[str], used_nodes: set[str], sequence: list[str]) -> None:
        if node == object_:
            output.append(tuple(sequence))
            return
        for edge_id, neighbour in adjacency.get(node, ()):
            if edge_id in used_edges or neighbour in used_nodes:
                continue
            visit(
                neighbour,
                used_edges | {edge_id},
                used_nodes | {neighbour},
                [*sequence, edge_id, neighbour],
            )

    visit(subject, set(), {subject}, [subject])
    return list(dict.fromkeys(output))


def _analysis_edge_groups(
    analysis: Mapping[str, Any],
    auxiliary_graphs: Mapping[str, Any],
    edge_id_map: Mapping[str, str],
) -> Iterator[tuple[str, tuple[str, ...]]]:
    path_bindings = _mapping(analysis.get("path_bindings"))
    yielded: set[tuple[str, ...]] = set()
    for binding_name, bindings in path_bindings.items():
        for auxiliary_id in _bound_ids(bindings):
            auxiliary = _mapping(auxiliary_graphs.get(auxiliary_id))
            ids = tuple(edge_id_map.get(x, x) for x in _tuple(auxiliary.get("edges")))
            if ids and ids not in yielded:
                yielded.add(ids)
                yield f"{binding_name}:{auxiliary_id}", ids
    if yielded:
        return
    ids = tuple(
        edge_id_map.get(edge_id, edge_id)
        for bindings in _mapping(analysis.get("edge_bindings")).values()
        for edge_id in _bound_ids(bindings)
    )
    if ids:
        yield "edge_bindings", tuple(dict.fromkeys(ids))


def _score_components(analyses: Sequence[Mapping[str, Any]]) -> tuple[ScoreComponent, ...]:
    components: list[ScoreComponent] = []
    for analysis in analyses:
        values: dict[str, float | None] = {
            "confidence": None,
            "clinical_evidence": None,
            "novelty": None,
            "normalized_score": None,
        }
        score = _float(analysis.get("score"))
        values["confidence"] = score
        for attribute in _attributes(analysis):
            label = _attribute_label(attribute)
            value = _float(attribute.get("value"))
            if "clinical" in label and "evidence" in label:
                values["clinical_evidence"] = value
            elif "novelty" in label:
                values["novelty"] = value
            elif "normalized" in label and "score" in label:
                values["normalized_score"] = value
            elif "confidence" in label:
                values["confidence"] = value
        components.append(ScoreComponent(**values, raw=analysis))
    return tuple(components)


def _path_name(sequence: Sequence[str], nodes: Mapping[str, Node]) -> str:
    labels = []
    for node_id in sequence[0::2]:
        node = nodes.get(node_id)
        labels.append(node.name if node and node.name else node_id)
    return "/".join(labels)


def parse_ars_response(
    payload: Any,
    *,
    pk: str | None = None,
    strict: bool = False,
) -> TranslatorExport:
    """Parse one ARS payload or a list of fetched root/child payloads."""
    responses = _extract_trapi_responses(payload)
    if not responses:
        raise TranslatorParseError("No TRAPI response was found in the ARS payload")

    nodes: dict[str, Node] = {}
    edges: dict[str, Edge] = {}
    paths: dict[str, Path] = {}
    publications: dict[str, Publication] = {}
    trials: dict[str, Trial] = {}
    results: list[QueryResult] = []
    result_ids: set[str] = set()
    all_aras: dict[str, None] = {}

    for response_index, response in enumerate(responses):
        message = _mapping(response.get("message"))
        knowledge_graph = _mapping(message.get("knowledge_graph"))
        query_graph = _mapping(message.get("query_graph"))
        auxiliary_graphs = _mapping(message.get("auxiliary_graphs"))
        native_results = tuple(x for x in _tuple(message.get("results")) if isinstance(x, Mapping))
        response_aras = tuple(dict.fromkeys(
            resource
            for result in native_results
            for analysis in _tuple(result.get("analyses"))
            if isinstance(analysis, Mapping)
            for resource in _resource_ids(analysis)
        ))
        for resource in response_aras:
            all_aras[resource] = None

        for node_id, raw_node in _mapping(knowledge_graph.get("nodes")).items():
            if isinstance(raw_node, Mapping) and str(node_id) not in nodes:
                nodes[str(node_id)] = _make_node(str(node_id), raw_node, response_aras)

        edge_id_map: dict[str, str] = {}
        for native_edge_id, raw_edge in _mapping(knowledge_graph.get("edges")).items():
            if not isinstance(raw_edge, Mapping):
                continue
            native_edge_id = str(native_edge_id)
            candidate_id = native_edge_id
            if candidate_id in edges and edges[candidate_id].raw != raw_edge:
                candidate_id = f"{native_edge_id}::{_json_id(response_index, raw_edge, size=10)}"
            edge_id_map[native_edge_id] = candidate_id
            if candidate_id not in edges:
                edge, edge_publications, edge_trials = _make_edge(candidate_id, raw_edge, response_aras)
                edges[candidate_id] = edge
                publications.update({k: v for k, v in edge_publications.items() if k not in publications})
                trials.update({k: v for k, v in edge_trials.items() if k not in trials})

        for result_index, raw_result in enumerate(native_results):
            subject, object_ = _result_endpoints(raw_result, query_graph)
            analyses = tuple(x for x in _tuple(raw_result.get("analyses")) if isinstance(x, Mapping))
            path_ids: list[str] = []
            path_names: list[str] = []
            for analysis_index, analysis in enumerate(analyses):
                for group_name, bound_edges in _analysis_edge_groups(analysis, auxiliary_graphs, edge_id_map):
                    sequences = _simple_paths(bound_edges, edges, subject, object_)
                    for sequence_index, sequence in enumerate(sequences):
                        path_id = _json_id(pk, response_index, result_index, analysis_index,
                                           group_name, sequence_index, sequence)
                        if path_id not in paths:
                            paths[path_id] = Path(
                                id=path_id,
                                subgraph=sequence,
                                aras=_resource_ids(analysis) or response_aras,
                                raw={
                                    "source": "ARS/TRAPI",
                                    "binding": group_name,
                                    "edge_ids": list(bound_edges),
                                    "analysis": analysis,
                                },
                            )
                        path_ids.append(path_id)
                        path_names.append(_path_name(sequence, nodes))

            native_result_id = raw_result.get("id")
            result_id = str(native_result_id) if native_result_id is not None else _json_id(
                pk, response_index, result_index, raw_result
            )
            if result_id in result_ids:
                result_id = f"{result_id}::{_json_id(response_index, result_index, raw_result, size=10)}"
            result_ids.add(result_id)
            score = next((_float(a.get("score")) for a in analyses if _float(a.get("score")) is not None), None)
            results.append(QueryResult(
                id=result_id,
                subject=subject,
                object=object_,
                path_ids=tuple(dict.fromkeys(path_ids)),
                score=score,
                name=path_names[0] if path_names else raw_result.get("name"),
                score_components=_score_components(analyses),
                raw=raw_result,
            ))

    meta = {
        "aras": list(all_aras),
        "qid": pk,
        "timestamp": next((x.get("datetime") or x.get("timestamp") for x in responses
                           if x.get("datetime") or x.get("timestamp")), None),
        "export_time": datetime.now(timezone.utc).isoformat(),
        "scope": "full",
        "format": "json",
        "resultCount": len(results),
        "source": "ARS API",
        "trapiResponseCount": len(responses),
    }
    export = TranslatorExport(
        meta=meta,
        results=tuple(results),
        nodes=nodes,
        edges=edges,
        paths=paths,
        publications=publications,
        trials=trials,
        raw={"pk": pk, "ars_payload": payload},
    )
    report = export.validate()
    if strict and not report.ok:
        preview = "; ".join(f"{x.location}: {x.message}" for x in report.errors[:5])
        raise TranslatorParseError(f"Validation failed with {len(report.errors)} error(s): {preview}")
    return export


def load_pk(
    pk: str,
    *,
    base_url: str = DEFAULT_ARS_URL,
    follow_children: bool = True,
    timeout: float = 120.0,
    retries: int = 2,
    strict: bool = False,
) -> TranslatorExport:
    """Fetch an ARS PK and return the same TranslatorExport type as UI_parser."""
    client = ARSClient(base_url, timeout=timeout, retries=retries)
    payloads = client.fetch_tree(pk, follow_children=follow_children)
    return parse_ars_response(payloads, pk=pk, strict=strict)


def _score_component_dict(component: ScoreComponent) -> dict[str, Any]:
    return {
        "confidence": component.confidence,
        "clinical_evidence": component.clinical_evidence,
        "novelty": component.novelty,
        "normalized_score": component.normalized_score,
    }


def to_ui_dict(export: TranslatorExport, *, include_raw: bool = False) -> dict[str, Any]:
    """Serialize using the field names consumed by ``UI_parser.load_export``."""
    output: dict[str, Any] = {
        "meta": dict(export.meta),
        "results": [
            {
                "id": result.id,
                "drug_name": result.name,
                "subject": result.subject,
                "object": result.object,
                "paths": list(result.path_ids),
                "score": result.score,
                "scoreComponents": [_score_component_dict(x) for x in result.score_components],
                **({"raw": result.raw} if include_raw else {}),
            }
            for result in export.results
        ],
        "nodes": {
            node.id: {
                "id": node.id, "names": list(node.names), "types": list(node.types),
                "curies": list(node.curies), "descriptions": list(node.descriptions),
                "species": node.species, "provenance": list(node.provenance),
                "synonyms": list(node.synonyms), "aras": list(node.aras),
                **({"raw": node.raw} if include_raw else {}),
            }
            for node in export.nodes.values()
        },
        "edges": {
            edge.id: {
                "id": edge.id, "subject": edge.subject, "object": edge.object,
                "predicate": edge.predicate, "predicate_url": edge.predicate_url,
                "knowledge_level": edge.knowledge_level,
                "provenance": list(edge.provenance), "publications": dict(edge.publications),
                "trials": list(edge.trials), "aras": list(edge.aras),
                "description": edge.description,
                **({"raw": edge.raw} if include_raw else {}),
            }
            for edge in export.edges.values()
        },
        "paths": {
            path.id: {
                "id": path.id, "subgraph": list(path.subgraph), "aras": list(path.aras),
                **({"raw": path.raw} if include_raw else {}),
            }
            for path in export.paths.values()
        },
        "publications": {
            item.id: {"id": item.id, "url": item.url, "source": dict(item.source),
                      **({"raw": item.raw} if include_raw else {})}
            for item in export.publications.values()
        },
        "trials": {
            item.id: {
                "id": item.id, "title": item.title, "url": item.url, "phase": item.phase,
                "start_date": item.start_date, "status": item.status, "type": item.type,
                "size": item.size, **({"raw": item.raw} if include_raw else {}),
            }
            for item in export.trials.values()
        },
    }
    if include_raw:
        output["raw"] = export.raw
    return output

def load_pk_data(pk: str) -> TranslatorExport:
    """Fetch an ARS PK and return the same TranslatorExport type as UI_parser."""
    return load_pk(pk, follow_children=True, strict=False)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and parse all Translator ARS results for a PK")
    parser.add_argument("pk", help="ARS message primary key")
    parser.add_argument("-o", "--output", help="Output JSON file; stdout when omitted")
    parser.add_argument("--base-url", default=DEFAULT_ARS_URL)
    parser.add_argument("--no-children", action="store_true", help="Do not follow child messages")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--include-raw", action="store_true", help="Include complete native records")
    args = parser.parse_args(argv)

    try:
        export = load_pk(
            args.pk,
            base_url=args.base_url,
            follow_children=not args.no_children,
            strict=args.strict,
        )
        text = json.dumps(to_ui_dict(export, include_raw=args.include_raw), indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as stream:
                stream.write(text + "\n")
        else:
            print(text)
        return 0
    except TranslatorParseError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
