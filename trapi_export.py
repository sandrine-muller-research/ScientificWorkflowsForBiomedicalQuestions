"""
Write the reranked UI paths as a TRAPI 1.5 Response so they can be inspected in any
TRAPI-aware viewer (ARAX UI, Translator UI, ...) or with plain JSON tools.

TRAPI mapping
-------------
message.query_graph      one node per side of the export's question (source / answer) and
                         one edge  source --[biolink:related_to]--> answer
message.knowledge_graph  every node and every edge of the exported paths, with the UI's
                         evidence carried as edge attributes: publications (biolink:publications),
                         supporting sentences (biolink:supporting_text), provenance
                         (biolink:primary_knowledge_source / aggregator), knowledge level,
                         clinical trials, plus the influence flow computed here.
message.auxiliary_graphs one per UI path (the ordered edge list of that path), scored.
message.results          one per answer; its single analysis lists the paths of that answer
                         as support_graphs, ordered by influence, with the influence score.

Paths are oriented SOURCE -> ANSWER (intervention -> target for the targets queries),
i.e. the reverse of the UI's storage order (answer -> query node). Edges keep their
asserted direction; the path order is carried by the auxiliary graph.

Usage (from main.py, after step 7)
----------------------------------
    from trapi_export import write_trapi_responses
    write_trapi_responses(dp, records, ui_paths, export, alias, kg, DATA_DIR)

or standalone on the saved CSV:
    python trapi_export.py data/ui_paths_reranked.csv
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from collections import defaultdict
from typing import Any

import pandas as pd

from graph_analysis import BIOLINK_INVERSE, BIOLINK_SYMMETRIC

TRAPI_VERSION = "1.5.0"
BIOLINK_VERSION = "4.2.2"
INFORES_THIS = "infores:context-influence-reranker"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _biolink(x: str | None, default: str) -> str:
    if not x:
        return default
    x = str(x).strip()
    return x if x.startswith("biolink:") else f"biolink:{x.replace(' ', '_')}"


def _attr(type_id: str, value: Any, **extra) -> dict:
    a = {"attribute_type_id": type_id, "value": value}
    a.update({k: v for k, v in extra.items() if v is not None})
    return a


def _pub_curie(pid: str) -> str:
    """
    Normalise a publication reference to a CURIE (PMID:, PMC:, DOI:, ISBN:, or a URL kept as is).
    Handles bare numbers, pubmed / PMC / doi URLs, 'doi:' prefixes and case variants.
    """
    p = str(pid).strip()
    if not p:
        return p
    if re.fullmatch(r"\d+", p):
        return f"PMID:{p}"
    m = re.fullmatch(r"(?i)pmid:?\s*(\d+)", p)
    if m:
        return f"PMID:{m.group(1)}"
    m = re.fullmatch(r"(?i)(?:pmc:?)?\s*(pmc\d+)", p)
    if m:
        return f"PMC:{m.group(1).upper()}"
    m = re.fullmatch(r"(?i)(?:https?://)?(?:www\.)?(?:pubmed\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov/pubmed)/(\d+)/?", p)
    if m:
        return f"PMID:{m.group(1)}"
    m = re.fullmatch(r"(?i)(?:https?://)?(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/(PMC\d+)/?", p)
    if m:
        return f"PMC:{m.group(1).upper()}"
    m = re.fullmatch(r"(?i)(?:https?://)?(?:dx\.)?doi\.org/(.+)", p) or re.fullmatch(r"(?i)doi:\s*(.+)", p)
    if m:
        return f"DOI:{m.group(1)}"
    m = re.fullmatch(r"(?i)isbn:?\s*([\d-]+x?)", p)
    if m:
        return f"ISBN:{m.group(1)}"
    return p


def pub_url(curie: str, known: dict | None = None) -> str | None:
    """
    Link for a publication CURIE. Uses the UI export's own publication URL when it has one,
    otherwise reconstructs it from the prefix:
        PMID:123     -> https://pubmed.ncbi.nlm.nih.gov/123/
        PMC:PMC123   -> https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/
        DOI:10.x/y   -> https://doi.org/10.x/y
        ISBN:...     -> https://www.worldcat.org/isbn/...
        http(s)://.. -> itself
    """
    if known and curie in known and known[curie]:
        return known[curie]
    c = str(curie)
    if c.startswith("PMID:"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{c[5:]}/"
    if c.startswith("PMC:"):
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{c[4:]}/"
    if c.startswith("DOI:"):
        return f"https://doi.org/{c[4:]}"
    if c.startswith("ISBN:"):
        return f"https://www.worldcat.org/isbn/{c[5:].replace('-', '')}"
    if re.match(r"(?i)https?://", c):
        return c
    return None


def infores_url(resource_id: str) -> str:
    """Link to the Translator infores catalog entry for a knowledge source id."""
    r = str(resource_id)
    return f"https://w3id.org/information-resource-registry/{r[len('infores:'):] if r.startswith('infores:') else r}"


def _primary_sources(e: dict) -> list[str]:
    return [s["resource_id"] for s in e.get("sources", []) if s.get("resource_role") == "primary_knowledge_source"]


def _qualifiers(raw) -> list[dict]:
    """
    Qualifiers of a UI-export edge as a TRAPI `qualifiers` list
    [{"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}, ...].
    Accepts the TRAPI list form, a dict {type: value}, or Translator-UI style fields
    (e.g. "object_aspect_qualifier": "activity" at the top level of the edge).
    """
    if not isinstance(raw, dict):
        return []
    out = []
    q = raw.get("qualifiers") or raw.get("qualifier_set")
    if isinstance(q, list):
        for x in q:
            if isinstance(x, dict) and x.get("qualifier_type_id") and x.get("qualifier_value") is not None:
                out.append({"qualifier_type_id": _biolink(x["qualifier_type_id"], "biolink:qualifier"),
                            "qualifier_value": str(x["qualifier_value"])})
    elif isinstance(q, dict):
        for k, v in q.items():
            if v is not None and v != "":
                out.append({"qualifier_type_id": _biolink(k, "biolink:qualifier"), "qualifier_value": str(v)})
    for k, v in raw.items():                      # top-level "<...>_qualifier" / "qualified_predicate" fields
        if (k.endswith("_qualifier") or k == "qualified_predicate") and v not in (None, "", []):
            tid = _biolink(k, "biolink:qualifier")
            if not any(o["qualifier_type_id"] == tid for o in out):
                out.append({"qualifier_type_id": tid, "qualifier_value": str(v)})
    return out


def _invert_qualifiers(quals: list[dict]) -> list[dict]:
    """When a hop is walked against its assertion, subject_* and object_* qualifiers swap ends."""
    out = []
    for q in quals:
        tid = q["qualifier_type_id"]
        if "subject_" in tid:
            tid = tid.replace("subject_", "object_")
        elif "object_" in tid:
            tid = tid.replace("object_", "subject_")
        out.append({"qualifier_type_id": tid, "qualifier_value": q["qualifier_value"]})
    return out


def _edge_attributes(edge, export, flow: float | None) -> tuple[list[dict], list[dict]]:
    """TRAPI attributes + sources for one UI edge."""
    attrs: list[dict] = []
    sources: list[dict] = []

    # provenance -> retrieval sources (primary first if identifiable)
    prov = [str(p) for p in (edge.provenance or ()) if p]
    for i, p in enumerate(prov):
        rid = p if p.startswith("infores:") else f"infores:{p}"
        sources.append({"resource_id": rid,
                        "resource_role": "primary_knowledge_source" if i == 0 else "aggregator_knowledge_source"})
    for a in (edge.aras or ()):
        rid = str(a) if str(a).startswith("infores:") else f"infores:{a}"
        if rid not in {s["resource_id"] for s in sources}:
            sources.append({"resource_id": rid, "resource_role": "aggregator_knowledge_source"})
    if not sources:
        sources.append({"resource_id": "infores:translator-ui-export", "resource_role": "aggregator_knowledge_source"})

    if edge.knowledge_level:
        attrs.append(_attr("biolink:knowledge_level", str(edge.knowledge_level)))
    if edge.description:
        attrs.append(_attr("biolink:description", str(edge.description)))

    # publications, with supporting text as sub-attributes
    pubs, texts = [], []
    for ev in export.iter_publication_evidence(edge):
        if not ev.publication_id:
            continue
        curie = _pub_curie(ev.publication_id)
        pubs.append(curie)
        if ev.support_text:
            texts.append(_attr("biolink:supporting_text", str(ev.support_text),
                               attributes=[_attr("biolink:publications", curie),
                                           *([_attr("biolink:xref", pub_url(curie, _known_pub_urls),
                                                    original_attribute_name="publication_url")]
                                             if pub_url(curie, _known_pub_urls) else []),
                                           _attr("biolink:evidence_class", ev.evidence_class)]
                               + ([_attr("biolink:subject_location_in_text", ev.support_subject)]
                                  if ev.support_subject else [])
                               + ([_attr("biolink:object_location_in_text", ev.support_object)]
                                  if ev.support_object else []),
                               attribute_source=ev.infores))
    if pubs:
        pubs_unique = list(dict.fromkeys(pubs))
        attrs.append(_attr("biolink:publications", pubs_unique, value_type_id="biolink:Publication"))
        urls = [u for u in (pub_url(p, _known_pub_urls) for p in pubs_unique) if u]
        if urls:
            attrs.append(_attr("biolink:xref", urls, original_attribute_name="publication_urls",
                               value_type_id="biolink:Uriorcurie"))
        attrs.append(_attr("biolink:evidence_count", len(pubs_unique)))
    attrs += texts

    trials = [str(t) for t in (edge.trials or ())]
    if trials:
        attrs.append(_attr("biolink:supporting_study", trials,
                           attributes=[_attr("biolink:name", export.trials[t].title)
                                       for t in trials if t in export.trials and export.trials[t].title] or None))
    if flow is not None:
        attrs.append(_attr("biolink:has_numeric_value", float(flow),
                           original_attribute_name="context_influence_flow",
                           description="alpha * pi[u] * P[u,v] from the source diffusion on the "
                                       "context-weighted graph (this run)",
                           attribute_source=INFORES_THIS))
    return attrs, sources


def _node(nid: str, export, kg=None) -> dict:
    n = export.nodes.get(nid)
    if n is not None:
        name = n.names[0] if n.names else nid
        cats = [_biolink(c, "biolink:NamedThing") for c in (n.types or ())] or ["biolink:NamedThing"]
        attrs = []
        if n.descriptions:
            attrs.append(_attr("biolink:description", str(n.descriptions[0])))
        if n.curies:
            attrs.append(_attr("biolink:xref", [str(c) for c in n.curies]))
        if n.synonyms:
            attrs.append(_attr("biolink:synonym", [str(s) for s in n.synonyms]))
        return {"name": name, "categories": cats, "attributes": attrs}
    if kg is not None and nid in kg.names:
        return {"name": kg.names[nid], "categories": [_biolink(kg.types.get(nid), "biolink:NamedThing")],
                "attributes": []}
    return {"name": nid, "categories": ["biolink:NamedThing"], "attributes": []}


# --------------------------------------------------------------------------- #
# main builder
# --------------------------------------------------------------------------- #
def _tct_evidence(details: list[dict]) -> tuple[list[dict], list[dict], str | None, list[dict]]:
    """Merge the TRAPI attributes/sources/qualifiers of the TCT edges supporting one hop."""
    attrs, sources, pubs, pred, quals = [], [], [], None, []
    seen_src = set()
    for e in details:
        pred = pred or e.get("predicate")
        for q in e.get("qualifiers") or []:
            if isinstance(q, dict) and q.get("qualifier_type_id") and q not in quals:
                quals.append({"qualifier_type_id": q["qualifier_type_id"], "qualifier_value": str(q.get("qualifier_value"))})
        for s in e.get("sources") or []:
            key = (s.get("resource_id"), s.get("resource_role"))
            if s.get("resource_id") and key not in seen_src:
                seen_src.add(key)
                sources.append({"resource_id": s["resource_id"], "resource_role": s.get("resource_role", "aggregator_knowledge_source")})
        for a in e.get("attributes") or []:
            tid = a.get("attribute_type_id", "")
            if tid == "biolink:publications":
                vals = a.get("value") if isinstance(a.get("value"), list) else [a.get("value")]
                pubs += [_pub_curie(x) for x in vals if x]
            elif tid in ("biolink:knowledge_level", "biolink:agent_type", "biolink:supporting_text",
                         "biolink:evidence_count", "biolink:supporting_study", "biolink:description"):
                attrs.append({k: v for k, v in a.items() if k in ("attribute_type_id", "value", "value_type_id",
                                                                  "original_attribute_name", "attribute_source",
                                                                  "description", "attributes")})
    if pubs:
        pubs = list(dict.fromkeys(pubs))
        attrs.insert(0, _attr("biolink:publications", pubs, value_type_id="biolink:Publication"))
        urls = [u for u in (pub_url(p, _known_pub_urls) for p in pubs) if u]
        if urls:
            attrs.insert(1, _attr("biolink:xref", urls, original_attribute_name="publication_urls",
                                  value_type_id="biolink:Uriorcurie"))
        attrs.insert(2, _attr("biolink:evidence_count", len(pubs)))
    return attrs, sources, pred, quals


def _walk_edge(u: str, v: str, asserted, export, flow, tag: str, kg_nodes, kg_edges, ensure_node,
               kg=None, tct_details=None) -> str:
    """
    One hop of a path, emitted IN WALK DIRECTION u -> v so the viewer draws the path
    source -> answer. `asserted` is the export edge (id, Edge) supporting the hop, if any.
      * asserted u->v : the export edge itself, with all its evidence.
      * asserted v->u : a new edge u -[inverse predicate]-> v carrying the same evidence, and
                        biolink:original_edge pointing at the asserted edge id (also in the KG).
      * none (TCT)    : a plain edge, predicate from the graph's arc label.
    """
    ensure_node(u); ensure_node(v)
    if asserted is not None:
        eid, e = asserted
        es, eo = export_alias(e.subject), export_alias(e.object)
        attrs, sources = _edge_attributes(e, export, flow)
        quals = _qualifiers(e.raw)
        if es == u and eo == v:
            key = eid if "::" in str(eid) else f"{tag}::{eid}"
            if key not in kg_edges:
                kg_edges[key] = {"subject": u, "object": v, "predicate": _biolink(e.predicate, "biolink:related_to"),
                                 "sources": sources, "attributes": attrs, **({"qualifiers": quals} if quals else {})}
            return key
        # reversed: keep the asserted edge too, for provenance
        okey = eid if "::" in str(eid) else f"{tag}::{eid}"
        if okey not in kg_edges:
            kg_edges[okey] = {"subject": es, "object": eo, "predicate": _biolink(e.predicate, "biolink:related_to"),
                              "sources": sources, "attributes": attrs, **({"qualifiers": quals} if quals else {})}
        pred = str(e.predicate or "related_to").replace("biolink:", "").replace(" ", "_")
        inv = pred if pred in BIOLINK_SYMMETRIC else BIOLINK_INVERSE.get(pred, f"inverse_of_{pred}")
        key = f"{okey}::rev"
        if key not in kg_edges:
            iq = _invert_qualifiers(quals)
            kg_edges[key] = {"subject": u, "object": v, "predicate": _biolink(inv, "biolink:related_to"),
                             "sources": sources,
                             "attributes": attrs + [_attr("biolink:description", okey,
                                                          original_attribute_name="original_edge",
                                                          description="asserted edge this hop is the inverse of")],
                             **({"qualifiers": iq} if iq else {})}
        return key
    key = f"{tag}::tct::{u}::{v}"
    if key not in kg_edges:
        details = (tct_details or {}).get((u, v)) or []
        reversed_ = False
        if not details:
            details = (tct_details or {}).get((v, u)) or []
            reversed_ = bool(details)
        attrs, sources, pred_raw, quals = _tct_evidence(details)
        if reversed_:
            quals = _invert_qualifiers(quals)
        pred = "biolink:related_to"
        if kg is not None:
            lab = kg.arc_label.get((kg.index.get(u, -1), kg.index.get(v, -1)))
            if lab:
                pred = _biolink(sorted(lab)[0], "biolink:related_to")
        elif pred_raw and not reversed_:
            pred = _biolink(pred_raw, "biolink:related_to")
        if not sources:
            sources = [{"resource_id": "infores:translator-component-toolkit", "resource_role": "aggregator_knowledge_source"}]
        attrs.append(_attr("biolink:description", "edge from TCT expansion (not in the UI export)"
                           + ("; asserted in the opposite direction" if reversed_ else "")))
        if flow is not None:
            attrs.append(_attr("biolink:has_numeric_value", float(flow), original_attribute_name="context_influence_flow",
                               attribute_source=INFORES_THIS))
        kg_edges[key] = {"subject": u, "object": v, "predicate": pred, "sources": sources, "attributes": attrs,
                         **({"qualifiers": quals} if quals else {})}
    return key


export_alias = lambda x: x   # replaced per call in build_trapi_response
_known_pub_urls: dict = {}


def _bind_cluster(result: dict, aux: dict, kg_edges: dict, member_sg: list[str]) -> None:
    """
    Make the result's RESULT GRAPH the whole cluster: bind every node and every real edge
    of the union of the member paths (support graphs). Intermediates go to n_intermediate,
    real edges to e_path. The inferred source -> target edge(s) stay bound to e_source_answer.
    Support graphs are untouched (one intervention -> target chain each).
    """
    edges = list(dict.fromkeys(k for sg in member_sg for k in aux[sg]["edges"]))
    src = {b["id"] for b in result["node_bindings"]["n_source"]}
    ans = {b["id"] for b in result["node_bindings"]["n_answer"]}
    nodes = list(dict.fromkeys(n for k in edges for n in (kg_edges[k]["subject"], kg_edges[k]["object"])))
    inter = [n for n in nodes if n not in src and n not in ans]
    if inter:
        result["node_bindings"]["n_intermediate"] = [{"id": n, "attributes": []} for n in inter]
    result["analyses"][0]["edge_bindings"]["e_path"] = [{"id": k, "attributes": []} for k in edges]


def build_trapi_response(path_rows: pd.DataFrame, tag: str, source_id: str, source_role: str,
                         paths: dict, edges: dict, export, alias: dict, kg=None,
                         edge_flow: dict[tuple[str, str], float] | None = None,
                         score_col: str = "bottleneck_flow", tct_details: dict | None = None,
                         group_by_intermediates: bool = True) -> dict:
    """
    One TRAPI Response for one export. ONE RESULT PER PATH.

    path_rows: one row per path with columns
        answer_id, path_node_ids ("a|b|c", already oriented source -> answer),
        score (float), origin ("computed" | "ui"), ui_path_id (or ""), ui_rank (int or nan)
    Each result binds n_source = source (intervention) / drug, n_answer = target / condition,
    one inferred edge between them, and ONE support graph = that path, hops in walk direction.
    """
    global export_alias, _known_pub_urls
    a = lambda x: alias.get(x, x)
    export_alias = a
    _known_pub_urls = {}
    for pid, pub in getattr(export, "publications", {}).items():
        url = getattr(pub, "url", None)
        if url:
            _known_pub_urls[_pub_curie(pid)] = url
            _known_pub_urls[str(pid)] = url
    kg_nodes: dict[str, dict] = {}
    kg_edges: dict[str, dict] = {}
    aux: dict[str, dict] = {}
    results: list[dict] = []

    def ensure_node(nid):
        if nid not in kg_nodes:
            kg_nodes[nid] = _node(nid, export, kg)

    # hop lookup over ALL exports' edges (the null model is the merged graph, so a path for
    # this export may use an edge that another export contributed); own export first
    by_pair: dict[tuple[str, str], list] = {}
    for eid, e in edges.items():
        by_pair.setdefault((a(e.subject), a(e.object)), []).append((eid, e))
    for eid, e in export.edges.items():                     # merged: ids are "<file>::<local id>"
        k = (a(e.subject), a(e.object))
        if not any(x[1] is e for x in by_pair.get(k, [])):
            by_pair.setdefault(k, []).append((eid, e))

    seen = set()
    seq_of: list[list[str]] = []          # node sequence per result index
    rows = path_rows.sort_values("score", ascending=False)
    for i, row in enumerate(rows.itertuples(index=False), start=1):
        seq = str(row.path_node_ids).split("|")
        if len(seq) < 2 or tuple(seq) in seen:
            continue
        seen.add(tuple(seq))
        answer = row.answer_id
        src_node, dst_node = seq[0], seq[-1]
        keys = []
        for u, v in zip(seq, seq[1:]):
            cand = by_pair.get((u, v)) or by_pair.get((v, u))
            f = edge_flow.get((u, v)) if edge_flow else None
            keys.append(_walk_edge(u, v, cand[0] if cand else None, export, f, tag, kg_nodes, kg_edges, ensure_node, kg,
                                   tct_details))
        aux_id = f"{tag}::path::{i}"
        score = float(row.score) if pd.notna(row.score) else 0.0
        names = " -> ".join(kg_nodes[n]["name"] for n in seq)
        aux[aux_id] = {"edges": keys, "attributes": [
            _attr("biolink:has_numeric_value", score, original_attribute_name="rank_score", attribute_source=INFORES_THIS,
                  description=f"{score_col} * (1 + w * type_diversity)"),
            *([_attr("biolink:has_numeric_value", float(row.influence), original_attribute_name=score_col,
                     attribute_source=INFORES_THIS),
               _attr("biolink:has_numeric_value", float(row.type_diversity), original_attribute_name="type_diversity",
                     attribute_source=INFORES_THIS, description="distinct coarse Biolink entity classes / nodes"),
               _attr("biolink:description", str(row.path_node_classes).split("|"), original_attribute_name="node_classes")]
              if hasattr(row, "influence") else []),
            _attr("biolink:description", names, original_attribute_name="path_nodes_source_to_answer"),
            _attr("biolink:description", row.origin, original_attribute_name="path_origin",
                  description="computed = strongest chain on the context-weighted graph; ui = path from the UI export"),
            *([_attr("biolink:description", str(row.ui_path_id), original_attribute_name="ui_path_id"),
               _attr("biolink:has_numeric_value", int(row.ui_rank), original_attribute_name="ui_path_rank")]
              if row.origin == "ui" and pd.notna(row.ui_rank) else []),
        ]}
        inf_key = f"{tag}::inferred::{i}"
        kg_edges[inf_key] = {
            "subject": src_node, "object": dst_node,
            "predicate": "biolink:related_to" if source_role == "intervention" else "biolink:treats_or_applied_or_studied_to_treat",
            "sources": [{"resource_id": INFORES_THIS, "resource_role": "primary_knowledge_source"}],
            "attributes": [_attr("biolink:knowledge_level", "prediction"),
                           _attr("biolink:agent_type", "computational_model"),
                           _attr("biolink:support_graphs", [aux_id]),
                           _attr("biolink:has_numeric_value", score, original_attribute_name=score_col,
                                 attribute_source=INFORES_THIS)]}
        seq_of.append(seq)
        results.append({
            "node_bindings": {"n_source": [{"id": src_node, "attributes": []}],
                              "n_answer": [{"id": dst_node, "attributes": []}]},
            "analyses": [{
                "resource_id": INFORES_THIS,
                "edge_bindings": {"e_source_answer": [{"id": inf_key, "attributes": []}]},
                "support_graphs": [aux_id],
                "score": score,
                "scoring_method": f"rank_score = {score_col} * (1 + w * type_diversity)",
                "attributes": [
                    _attr("biolink:has_numeric_value", len(seq) - 1, original_attribute_name="hops"),
                    *(lambda prims, pubs: [
                        _attr("biolink:primary_knowledge_source", prims, original_attribute_name="path_primary_sources",
                              description="primary knowledge source of each hop's evidence",
                              attributes=[_attr("biolink:xref", [infores_url(p) for p in prims],
                                                original_attribute_name="primary_source_urls")]),
                        _attr("biolink:publications", pubs, original_attribute_name="path_publications",
                              value_type_id="biolink:Publication",
                              attributes=[_attr("biolink:xref", [u for u in (pub_url(p, _known_pub_urls) for p in pubs) if u],
                                                original_attribute_name="publication_urls")]),
                    ] if (prims or pubs) else [])(
                        list(dict.fromkeys(p for k in keys for p in _primary_sources(kg_edges[k]))),
                        list(dict.fromkeys(p for k in keys for a_ in kg_edges[k]["attributes"]
                                           if a_["attribute_type_id"] == "biolink:publications"
                                           for p in (a_["value"] if isinstance(a_["value"], list) else [a_["value"]])))),
                    _attr("biolink:description", [kg_nodes[n]["name"] for n in seq[1:-1]],
                          original_attribute_name="intermediate_nodes"),
                    _attr("biolink:description", " | ".join(kg_nodes[n]["name"] for n in seq[1:-1]) or "(direct)",
                          original_attribute_name="path_group",
                          description="paths sharing this intermediate set form a group"),
                    _attr("biolink:description", answer, original_attribute_name="answer_id"),
                    _attr("biolink:description", names, original_attribute_name="path"),
                    _attr("biolink:description", source_role, original_attribute_name="query_role"),
                ],
            }],
        })

    if group_by_intermediates and results:
        # GROUPING: every path belongs to EXACTLY ONE result.
        #   * a path with intermediates joins the group of its most-shared intermediate node
        #     (ties -> the one closest to the source); a group needs >= 2 paths
        #   * paths whose intermediates are shared by nobody, and direct paths, stay single
        # A group result binds the source and all targets reached through the node, one
        # inferred source -> target edge per target, and lists its member paths as support
        # graphs (each one a single source -> ... -> target chain). No aux graph is shared
        # between results, so nothing appears twice in the list.
        share = {}
        for i in range(len(results)):
            for n in seq_of[i][1:-1]:
                share[n] = share.get(n, 0) + 1
        assign: dict[int, str | None] = {}
        for i in range(len(results)):
            inter = seq_of[i][1:-1]
            cands = [n for n in inter if share[n] >= 2]
            assign[i] = max(cands, key=lambda n: (share[n], -inter.index(n))) if cands else None
        members_of: dict[str, list[int]] = {}
        for i, n in assign.items():
            if n is not None:
                members_of.setdefault(n, []).append(i)
        # a "group" with a single member after assignment stays a single path result
        for n, mem in list(members_of.items()):
            if len(mem) < 2:
                for i in mem:
                    assign[i] = None
                del members_of[n]

        for i, r in enumerate(results):
            r["analyses"][0]["attributes"].append(_attr("biolink:description", "path", original_attribute_name="result_kind"))

        group_results = []
        for nid, members in members_of.items():
            name = kg_nodes[nid]["name"]
            member_paths = [results[i]["analyses"][0]["support_graphs"][0] for i in members]
            targets = list(dict.fromkeys(seq_of[i][-1] for i in members))
            scores = [results[i]["analyses"][0]["score"] for i in members]
            best, total = max(scores), float(sum(scores))
            inf_keys = []
            for tgt in targets:
                paths_to_t = [results[i]["analyses"][0]["support_graphs"][0] for i in members if seq_of[i][-1] == tgt]
                k = f"{tag}::inferred::group::{nid}::{tgt}"
                kg_edges[k] = {
                    "subject": seq_of[members[0]][0], "object": tgt,
                    "predicate": "biolink:related_to" if source_role == "intervention" else "biolink:treats_or_applied_or_studied_to_treat",
                    "sources": [{"resource_id": INFORES_THIS, "resource_role": "primary_knowledge_source"}],
                    "attributes": [_attr("biolink:knowledge_level", "prediction"),
                                   _attr("biolink:agent_type", "computational_model"),
                                   _attr("biolink:support_graphs", paths_to_t),
                                   _attr("biolink:description", f"via {name}", original_attribute_name="via_intermediate")]}
                inf_keys.append(k)
            group_results.append({
                "node_bindings": {"n_source": [{"id": seq_of[members[0]][0], "attributes": []}],
                                  "n_answer": [{"id": tgt, "attributes": []} for tgt in targets]},
                "analyses": [{
                    "resource_id": INFORES_THIS,
                    "edge_bindings": {"e_source_answer": [{"id": k, "attributes": []} for k in inf_keys]},
                    "support_graphs": member_paths,
                    "score": total,
                    "scoring_method": f"sum of {score_col} over the {len(members)} paths through {name} "
                                      f"(total influence routed through this node); best single path = {best:.4g}",
                    "attributes": [
                        _attr("biolink:description", "group", original_attribute_name="result_kind",
                              description="one result gathering all paths through a shared intermediate node"),
                        _attr("biolink:description", nid, original_attribute_name="intermediate_id"),
                        _attr("biolink:description", name, original_attribute_name="intermediate_name"),
                        _attr("biolink:has_numeric_value", len(members), original_attribute_name="n_paths"),
                        _attr("biolink:has_numeric_value", len(targets), original_attribute_name="n_targets"),
                        _attr("biolink:has_numeric_value", best, original_attribute_name=f"best_path_{score_col}",
                              attribute_source=INFORES_THIS),
                        _attr("biolink:has_numeric_value", total, original_attribute_name=f"summed_{score_col}",
                              attribute_source=INFORES_THIS),
                        _attr("biolink:description", [kg_nodes[x]["name"] for x in targets], original_attribute_name="targets"),
                        _attr("biolink:description", f"{kg_nodes[seq_of[members[0]][0]]['name']} -> {name} -> "
                                                     + ", ".join(kg_nodes[x]["name"] for x in targets),
                              original_attribute_name="path"),
                        _attr("biolink:description", name, original_attribute_name="path_group"),
                        _attr("biolink:description", source_role, original_attribute_name="query_role"),
                        _attr("biolink:has_numeric_value", 0, original_attribute_name="hops"),
                    ],
                }],
            })
        grouped = {i for i, n in assign.items() if n is not None}
        results = [r for i, r in enumerate(results) if i not in grouped] + group_results
        results.sort(key=lambda r: -r["analyses"][0]["score"])

    # Result Graph = the whole cluster (union of the result's paths); support graphs = each path
    for r in results:
        _bind_cluster(r, aux, kg_edges, r["analyses"][0]["support_graphs"])

    if source_role == "intervention":
        ensure_node(source_id)
        qg = {"nodes": {"n_source": {"ids": [source_id], "categories": kg_nodes[source_id]["categories"]},
                        "n_answer": {"categories": sorted({c for r in results
                                                           for c in kg_nodes[r["node_bindings"]["n_answer"][0]["id"]]["categories"]})
                                     or ["biolink:NamedThing"], "is_set": True},
                        "n_intermediate": {"categories": ["biolink:NamedThing"], "is_set": True}},
              "edges": {"e_source_answer": {"subject": "n_source", "object": "n_answer",
                                            "predicates": ["biolink:related_to"], "knowledge_type": "inferred"},
                        "e_path": {"subject": "n_source", "object": "n_answer",
                                   "predicates": ["biolink:related_to"]}}}
    else:
        conds = sorted({r["node_bindings"]["n_answer"][0]["id"] for r in results})
        qg = {"nodes": {"n_source": {"categories": sorted({c for r in results
                                                           for c in kg_nodes[r["node_bindings"]["n_source"][0]["id"]]["categories"]})
                                     or ["biolink:NamedThing"]},
                        "n_answer": {"ids": conds, "categories": sorted({c for x in conds for c in kg_nodes[x]["categories"]})
                                     or ["biolink:NamedThing"]}},
              "edges": {"e_source_answer": {"subject": "n_source", "object": "n_answer",
                                            "predicates": ["biolink:treats_or_applied_or_studied_to_treat"],
                                            "knowledge_type": "inferred"},
                        "e_path": {"subject": "n_source", "object": "n_answer",
                                   "predicates": ["biolink:related_to"]}}}
        qg["nodes"]["n_intermediate"] = {"categories": ["biolink:NamedThing"], "is_set": True}
    return {
        "message": {"query_graph": qg, "knowledge_graph": {"nodes": kg_nodes, "edges": kg_edges},
                    "auxiliary_graphs": aux, "results": results},
        "status": "Success",
        "description": f"{len(results)} paths ({tag}), each a result, ranked by {score_col}",
        "schema_version": TRAPI_VERSION, "biolink_version": BIOLINK_VERSION,
        "workflow": [{"id": "score"}],
        "logs": [{"timestamp": datetime.now(timezone.utc).isoformat(), "level": "INFO",
                  "message": f"source={source_id} ({source_role}); one result per path, hops in walk direction"}],
    }


def merge_responses(per_export: dict[str, dict], score_col: str) -> dict:
    """
    One TRAPI Response for all exports, ONE RESULT PER PATH, ranked globally by
    relative_score = score / best score of the path's export (scores are not comparable
    across exports: each source diffusion has mass 1 spread over a different graph).
    """
    kg_nodes, kg_edges, aux, results, logs = {}, {}, {}, [], []
    for tag, resp in per_export.items():
        m = resp["message"]
        kg_nodes.update(m["knowledge_graph"]["nodes"]); kg_edges.update(m["knowledge_graph"]["edges"])
        aux.update(m["auxiliary_graphs"]); logs += resp["logs"]
        best = max((r["analyses"][0]["score"] for r in m["results"]), default=0.0) or 1.0
        logs.append({"timestamp": datetime.now(timezone.utc).isoformat(), "level": "INFO",
                     "message": f"{tag}: query_graph={json.dumps(m['query_graph'])}; best {score_col}={best:.4g}"})
        for r in m["results"]:
            r = json.loads(json.dumps(r))
            an = r["analyses"][0]
            raw = an["score"]
            an["score"] = raw / best
            an["scoring_method"] = f"relative_score = {score_col} / best {score_col} of export {tag}"
            an["attributes"] += [_attr("biolink:has_numeric_value", raw, original_attribute_name=f"raw_{score_col}",
                                       attribute_source=INFORES_THIS),
                                 _attr("biolink:description", tag, original_attribute_name="source_export")]
            results.append(r)
    # global order: by relative score (group results carry their summed score, so a node on
    # many paths ranks above a single path of similar strength)
    results.sort(key=lambda r: -r["analyses"][0]["score"])
    for rank, r in enumerate(results, start=1):
        r["analyses"][0]["attributes"].append(
            _attr("biolink:has_numeric_value", rank, original_attribute_name="global_rank", attribute_source=INFORES_THIS))
    qg = {"nodes": {"n_source": {"categories": ["biolink:NamedThing"]},
                    "n_answer": {"categories": sorted({c for n in kg_nodes.values() for c in n["categories"]}) or ["biolink:NamedThing"],
                                 "is_set": True},
                    "n_intermediate": {"categories": ["biolink:NamedThing"], "is_set": True}},
          "edges": {"e_source_answer": {"subject": "n_source", "object": "n_answer", "predicates": ["biolink:related_to"],
                                        "knowledge_type": "inferred"},
                    "e_path": {"subject": "n_source", "object": "n_answer", "predicates": ["biolink:related_to"]}}}
    return {
        "message": {"query_graph": qg, "knowledge_graph": {"nodes": kg_nodes, "edges": kg_edges},
                    "auxiliary_graphs": aux, "results": results},
        "status": "Success",
        "description": f"{len(results)} paths from {len(per_export)} exports, one result per path, "
                       f"ranked by relative_score; hops in walk direction source -> answer",
        "schema_version": TRAPI_VERSION, "biolink_version": BIOLINK_VERSION,
        "workflow": [{"id": "score"}], "logs": logs,
    }


def results_table(response: dict) -> pd.DataFrame:
    """
    One row per TRAPI result, in the JSON's order, so the CSV mirrors results.trapi.json exactly.
    Path results: the path with its intermediates, node types/classes, predicates, qualifiers,
    per-hop PMIDs and evidence. Group results: the shared intermediate and its targets, with
    the member paths listed.
    """
    m = response["message"]
    N, E, A = m["knowledge_graph"]["nodes"], m["knowledge_graph"]["edges"], m["auxiliary_graphs"]
    name = lambda n: N.get(n, {}).get("name", n)
    typ = lambda n: (N.get(n, {}).get("categories") or ["?"])[0].replace("biolink:", "")

    def hop_info(edge_key):
        e = E[edge_key]
        at = {a.get("attribute_type_id"): a for a in e.get("attributes", [])}
        pubs = at.get("biolink:publications", {}).get("value") or []
        if isinstance(pubs, str):
            pubs = [pubs]
        quals = ";".join(f"{q['qualifier_type_id'].replace('biolink:', '')}={q['qualifier_value']}"
                         for q in e.get("qualifiers", []))
        srcs = ";".join(s["resource_id"].replace("infores:", "") for s in e.get("sources", []))
        kind = "tct" if "::tct::" in edge_key else "inverse" if edge_key.endswith("::rev") else "asserted"
        prim = ";".join(p.replace("infores:", "") for p in _primary_sources(e))
        return (e["predicate"].replace("biolink:", ""), quals, pubs, srcs, kind,
                at.get("biolink:knowledge_level", {}).get("value", ""), prim)

    def path_row(sg_id):
        g = A[sg_id]
        at = {a.get("original_attribute_name"): a["value"] for a in g["attributes"]}
        seq = [E[g["edges"][0]]["subject"]] + [E[k]["object"] for k in g["edges"]]
        hops = [hop_info(k) for k in g["edges"]]
        all_pubs = list(dict.fromkeys(p for h in hops for p in h[2]))
        prims = list(dict.fromkeys(p for h in hops for p in h[6].split(";") if p))
        return {
            "path_node_ids": "|".join(seq),
            "path_node_names": "|".join(name(n) for n in seq),
            "path_node_types": "|".join(typ(n) for n in seq),
            "path_node_classes": "|".join(at.get("node_classes", [])) if isinstance(at.get("node_classes"), list) else "",
            "intermediate_ids": "|".join(seq[1:-1]),
            "intermediate_names": "|".join(name(n) for n in seq[1:-1]),
            "path_predicates": "|".join(h[0] for h in hops),
            "hop_kinds": "|".join(h[4] for h in hops),
            "hop_qualifiers": "|".join(h[1] for h in hops),
            "hop_knowledge_levels": "|".join(h[5] for h in hops),
            "hop_primary_sources": "|".join(h[6] for h in hops),
            "primary_sources": ";".join(prims),
            "primary_source_urls": ";".join(infores_url(p) for p in prims),
            "hop_sources": "|".join(h[3] for h in hops),
            "hop_publications": "|".join(";".join(h[2]) for h in hops),
            "n_publications": len(all_pubs),
            "publications": ";".join(all_pubs),
            "publication_urls": ";".join(u for u in (pub_url(p, _known_pub_urls) for p in all_pubs) if u),
            "path_origin": at.get("path_origin", ""),
            "bottleneck_flow": at.get("bottleneck_flow", float("nan")),
            "type_diversity": at.get("type_diversity", float("nan")),
        }

    rows = []
    for r in m["results"]:
        an = r["analyses"][0]
        at = {a.get("original_attribute_name"): a["value"] for a in an["attributes"]}
        src = r["node_bindings"]["n_source"][0]["id"]
        targets = [b["id"] for b in r["node_bindings"]["n_answer"]]
        base = {"global_rank": at.get("global_rank"), "result_kind": at.get("result_kind"),
                "score": an["score"], "raw_score": at.get(next((k for k in at if k.startswith("raw_")), ""), float("nan")),
                "source_export": at.get("source_export", ""),
                "source_id": src, "source_name": name(src),
                "target_ids": "|".join(targets), "target_names": "|".join(name(x) for x in targets),
                "n_targets": len(targets), "hops": at.get("hops")}
        if at.get("result_kind") == "group":
            members = an["support_graphs"]
            base.update({"intermediate_ids": at.get("intermediate_id", ""),
                         "intermediate_names": at.get("intermediate_name", ""),
                         "n_paths": len(members),
                         "member_paths": " || ".join(next(a["value"] for a in A[s]["attributes"]
                                                          if a.get("original_attribute_name") == "path_nodes_source_to_answer")
                                                     for s in members),
                         "publications": ";".join(dict.fromkeys(p for s in members for k in A[s]["edges"] for p in hop_info(k)[2])),
                         "primary_sources": ";".join(dict.fromkeys(p for s in members for k in A[s]["edges"]
                                                                   for p in hop_info(k)[6].split(";") if p))})
            base["primary_source_urls"] = ";".join(infores_url(p) for p in base["primary_sources"].split(";") if p)
            base["n_publications"] = len(base["publications"].split(";")) if base["publications"] else 0
            base["publication_urls"] = ";".join(u for u in (pub_url(p, _known_pub_urls)
                                                             for p in base["publications"].split(";") if p) if u)
        else:
            base["n_paths"] = 1
            base.update(path_row(an["support_graphs"][0]))
        rows.append(base)
    cols = ["global_rank", "result_kind", "score", "raw_score", "source_export", "source_id", "source_name",
            "target_ids", "target_names", "n_targets", "hops", "n_paths", "intermediate_ids", "intermediate_names",
            "path_node_ids", "path_node_names", "path_node_types", "path_node_classes", "path_predicates",
            "hop_kinds", "hop_qualifiers", "hop_knowledge_levels", "hop_primary_sources", "primary_sources",
            "primary_source_urls", "hop_sources", "hop_publications",
            "n_publications", "publications", "publication_urls", "path_origin", "bottleneck_flow", "type_diversity", "member_paths"]
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]]
    text_cols = [c for c in df.columns if df[c].dtype == object or c in ("intermediate_ids", "intermediate_names", "member_paths")]
    df[text_cols] = df[text_cols].fillna("")
    return df


def write_trapi_responses(path_rows: pd.DataFrame, ui_paths: dict, export, alias: dict, kg, out_dir: str,
                          edge_flow: dict | None = None, targets_dirname: str = "targets",
                          score_col: str = "bottleneck_flow", per_export_files: bool = False,
                          merged_name: str = "results.trapi.json", tct_details: dict | None = None,
                          query_roles: tuple[str, ...] = ("intervention",),
                          group_by_intermediates: bool = True) -> list[str]:
    """
    path_rows: one row per path (source_file, answer_id, path_node_ids, score, origin, ui_path_id, ui_rank).
    Always writes ONE merged file <out_dir>/<merged_name>; per_export_files adds one per export.
    """
    per_export = {}
    for tag, g in path_rows.groupby("source_file", sort=False):
        paths, edges, query = ui_paths[tag]
        if tag.split("/")[0] == targets_dirname and query:
            source_id, role = query, "intervention"
        else:
            source_id, role = "context", "context"
        if role not in query_roles:
            continue
        per_export[tag] = build_trapi_response(g, tag, source_id, role, paths, edges, export, alias, kg,
                                               edge_flow=edge_flow, score_col=score_col, tct_details=tct_details,
                                               group_by_intermediates=group_by_intermediates)
    if not per_export:
        print("  TRAPI: no export matches query_roles", query_roles, "- nothing written")
        return []
    written = []
    if per_export_files:
        os.makedirs(os.path.join(out_dir, "trapi"), exist_ok=True)
        for tag, resp in per_export.items():
            fn = os.path.join(out_dir, "trapi", tag.replace("/", "_").replace(".json", "") + ".trapi.json")
            with open(fn, "w") as f:
                json.dump(resp, f, indent=1)
            written.append(fn)
    merged = merge_responses(per_export, score_col)
    fn = os.path.join(out_dir, merged_name)
    with open(fn, "w") as f:
        json.dump(merged, f, indent=1)
    written.append(fn)
    m = merged["message"]
    print(f"  TRAPI: {fn} ({len(m['results'])} results, {len(m['auxiliary_graphs'])} paths, "
          f"{len(m['knowledge_graph']['edges'])} edges, from {len(per_export)} exports)")
    tbl = results_table(merged)
    tfn = os.path.join(out_dir, merged_name.replace(".trapi.json", "") + "_table.csv")
    tbl.to_csv(tfn, index=False)
    written.append(tfn)
    print(f"  CSV mirror of the TRAPI results: {tfn} ({len(tbl)} rows, same order and global_rank)")
    return written
