"""Normalized CSV export for analysis with pandas, R, SQL, or spreadsheet tools."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .UI_parser import TranslatorExport


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _write(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_csv(dataset: TranslatorExport, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    meta_path = output / "meta.json"
    meta_path.write_text(json.dumps(dataset.meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written.append(meta_path)

    specs = {
        "results.csv": (["result_id", "name", "subject", "object", "score", "path_count", "path_ids", "score_components"], (
            {"result_id": r.id, "name": r.name, "subject": r.subject, "object": r.object,
             "score": r.score, "path_count": len(r.path_ids), "path_ids": _json(r.path_ids),
             "score_components": _json(r.raw.get("scoreComponents", []))} for r in dataset.results)),
        "nodes.csv": (["node_id", "name", "names", "types", "curies", "descriptions", "species", "synonyms", "provenance", "aras"], (
            {"node_id": n.id, "name": n.name, "names": _json(n.names), "types": _json(n.types),
             "curies": _json(n.curies), "descriptions": _json(n.descriptions), "species": _json(n.species),
             "synonyms": _json(n.synonyms), "provenance": _json(n.provenance), "aras": _json(n.aras)}
            for n in dataset.nodes.values())),
        "edges.csv": (["edge_id", "subject", "object", "predicate", "predicate_url", "knowledge_level", "description", "provenance", "aras"], (
            {"edge_id": e.id, "subject": e.subject, "object": e.object, "predicate": e.predicate,
             "predicate_url": e.predicate_url, "knowledge_level": e.knowledge_level,
             "description": e.description, "provenance": _json(e.provenance), "aras": _json(e.aras)}
            for e in dataset.edges.values())),
        "paths.csv": (["path_id", "subgraph", "node_count", "edge_count", "aras"], (
            {"path_id": p.id, "subgraph": _json(p.subgraph), "node_count": (len(p.subgraph)+1)//2,
             "edge_count": len(p.subgraph)//2, "aras": _json(p.aras)} for p in dataset.paths.values())),
        "path_steps.csv": (["path_id", "step_index", "source_node_id", "edge_id", "target_node_id", "direction"], (
            {"path_id": p.id, "step_index": s.index, "source_node_id": s.source.id,
             "edge_id": s.edge.id, "target_node_id": s.target.id, "direction": s.direction}
            for p in dataset.paths.values() for s in dataset.resolve_path(p).steps)),
        "result_paths.csv": (["result_id", "path_id", "path_rank"], (
            {"result_id": r.id, "path_id": path_id, "path_rank": rank}
            for r in dataset.results for rank, path_id in enumerate(r.path_ids))),
        "publications.csv": (["publication_id", "url", "source"], (
            {"publication_id": p.id, "url": p.url, "source": _json(p.source)}
            for p in dataset.publications.values())),
        "edge_publications.csv": (["edge_id", "evidence_class", "publication_id", "infores", "support_text", "support_subject", "support_object"], (
            {"edge_id": x.edge_id, "evidence_class": x.evidence_class, "publication_id": x.publication_id,
             "infores": x.infores, "support_text": x.support_text, "support_subject": _json(x.support_subject),
             "support_object": _json(x.support_object)} for x in dataset.iter_publication_evidence())),
        "trials.csv": (["trial_id", "title", "url", "phase", "start_date", "status", "type", "size"], (
            {"trial_id": t.id, "title": t.title, "url": t.url, "phase": _json(t.phase),
             "start_date": t.start_date, "status": t.status, "type": t.type, "size": _json(t.size)}
            for t in dataset.trials.values())),
        "edge_trials.csv": (["edge_id", "trial_id"], (
            {"edge_id": e.id, "trial_id": trial_id} for e in dataset.edges.values() for trial_id in e.trials)),
    }
    for filename, (fields, rows) in specs.items():
        path = output / filename
        _write(path, fields, rows)
        written.append(path)
    return written

