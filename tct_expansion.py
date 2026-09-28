"""
TCT one-hop expansion of the context nodes, returning real EDGES (not just ids)
so they can be added to the graph. Works with both TCT APIs:

* new (>= Aug 2026, v0.3.0): get_translator_resources / neighborhood_finder(node, cats, resources=...)
* old (<= v0.2.0):          translator_metakg.load_translator_resources + positional call

Results are cached in a JSON file so re-running the analysis does not re-query Translator.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import pandas as pd
import requests


def _biolink(cats):
    return [c if c.startswith("biolink:") else f"biolink:{c}" for c in cats]


def _retriever_metakg_rows(api_names: dict) -> tuple[str | None, pd.DataFrame]:
    """SmartAPI lists 'Retriever (TRAPI v1.6)' but indexes its MetaKG as 'Retriever'."""
    titles = [k for k in api_names if k.startswith("Retriever") and "v2" not in k]
    if not titles:
        return None, pd.DataFrame()
    title = sorted(titles)[-1]
    try:
        r = requests.get("https://smart-api.info/api/metakg",
                         params={"q": "(api.x-translator.component:KP AND api.name:Retriever)", "size": 5000},
                         timeout=120)
        hits = r.json().get("hits", [])
    except Exception as exc:                                   # network etc.
        print(f"  Retriever MetaKG patch skipped: {exc}")
        return None, pd.DataFrame()
    rows = pd.DataFrame({
        "API": title,
        "Predicate": ["biolink:" + h["predicate"] for h in hits],
        "Subject": ["biolink:" + h["subject"] for h in hits],
        "Object": ["biolink:" + h["object"] for h in hits],
        "URL": api_names[title],
    })
    return title, rows


def _patch(api_names, meta_kg):
    title, rows = _retriever_metakg_rows(api_names)
    if title and not rows.empty:
        meta_kg = pd.concat([meta_kg[meta_kg["API"] != title], rows], ignore_index=True)
        print(f"  Patched {title}: {len(rows)} MetaKG edges")
    api_predicates = {a: sorted(set(meta_kg.loc[meta_kg["API"] == a, "Predicate"]))
                      for a in meta_kg["API"].unique()}
    return meta_kg, api_predicates


def make_finder():
    """Return a function curie, categories -> TRAPI-like knowledge_graph dict."""
    try:
        from TCT import get_translator_resources, neighborhood_finder, TranslatorResources
        res = get_translator_resources()
        meta_kg, api_predicates = _patch(res.api_names, res.meta_kg)
        res = TranslatorResources(api_names=res.api_names, meta_kg=meta_kg, api_predicates=api_predicates)

        def find(curie, categories):
            return neighborhood_finder(curie, _biolink(categories), resources=res).knowledge_graph
        print("  Using TCT >= 0.3 API")
        return find
    except ImportError:
        pass

    from TCT import translator_metakg, TCT_neighborhood_finder
    api_names, meta_kg, _ = translator_metakg.load_translator_resources(use_new_metakg_url=True)
    meta_kg, api_predicates = _patch(api_names, meta_kg)

    def find(curie, categories):
        # returns (input_node_id, raw_result, parsed_result, ranked)
        _, _, parsed, _ = TCT_neighborhood_finder.neighborhood_finder(
            curie, _biolink(categories), api_names, meta_kg, api_predicates)
        return parsed["knowledge_graph"]
    print("  Using TCT <= 0.2 API")
    return find


def expand_context(curies, categories, max_neighbours=50, cache_path=None):
    """
    Returns (edges, nodes):
      edges: list of (subject, object, predicate)
      nodes: {id: {"name":..., "categories": [...]}}
    Keeps, per context node, the `max_neighbours` neighbours supported by the most edges.
    """
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    find = None
    edges, nodes = [], {}
    for curie in curies:
        key = f"{curie}|{','.join(sorted(categories))}"
        if key not in cache:
            find = find or make_finder()
            try:
                kg = find(curie, categories)
                cache[key] = {"nodes": kg.get("nodes", {}),
                              "edges": [{"subject": e["subject"], "object": e["object"],
                                         "predicate": e.get("predicate")}
                                        for e in kg.get("edges", {}).values()]}
            except Exception as exc:
                print(f"  TCT failed for {curie}: {type(exc).__name__}: {exc}")
                continue
        kg = cache[key]
        support = Counter()
        for e in kg["edges"]:
            other = e["object"] if e["subject"] == curie else e["subject"]
            support[other] += 1
        keep = {n for n, _ in support.most_common(max_neighbours)}
        kept = [e for e in kg["edges"]
                if (e["object"] if e["subject"] == curie else e["subject"]) in keep]
        edges += [(e["subject"], e["object"], e["predicate"]) for e in kept]
        nodes.update({n: kg["nodes"].get(n, {}) for n in keep})
        print(f"  {curie}: {len(support)} neighbours from TCT, kept {len(keep)} ({len(kept)} edges)")

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    return edges, nodes
