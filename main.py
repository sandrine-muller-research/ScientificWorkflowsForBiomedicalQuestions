import os
from collections import Counter

import numpy as np
import pandas as pd

from UI_parser import load_export
from graph_analysis import (build_alias_map, detect_query_roles, build_graph,
                            random_walk_with_restart, edge_influence,
                            most_probable_paths, reconstruct, describe_path, path_sources)

# ------------------------------------------------------------------ settings
DATA_DIR = "data/"
USE_TCT = True                         # expand context with TCT one-hop neighbours
TCT_CATEGORIES = ["Gene", "Protein"]
TCT_MAX_NEIGHBOURS = 50                # per context node, ranked by number of supporting edges
ALPHA = 0.85                           # RWR continuation probability (restart = 1 - ALPHA)
BLOCK_QUERY_NODE = True                # forbid paths through the pinned query node
EDGE_FLOW_THRESHOLD = 1e-4


def main():
    ############# STEP 1: load UI export + context #############
    mapping_path = os.path.join(DATA_DIR, "context_mapping.csv")
    if not os.path.exists(mapping_path):
        print("No context_mapping.csv - nothing to analyse.")
        return
    df_context = pd.read_csv(mapping_path)

    json_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json") and not f.startswith("tct_cache"))
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in {DATA_DIR}")
    data_path = os.path.join(DATA_DIR, json_files[0])
    print(f"UI export: {data_path}")
    export = load_export(data_path)

    alias = build_alias_map(export)
    pinned, answer_field = detect_query_roles(export)
    print(f"{len(export.results)} results; pinned query node = {pinned}; answers are result.{answer_field}")

    ############# STEP 2: context seeds #############
    context_curies = df_context["corresponding_id"].dropna().astype(str).unique().tolist()
    seeds_ids = [alias.get(c, c) for c in context_curies]
    in_ui = [c for c in context_curies if c in alias]
    print(f"Context: {len(context_curies)} ids, {len(in_ui)} present in the UI graph")
    missing = sorted(set(context_curies) - set(in_ui))
    if missing:
        print(f"  not in UI graph (need TCT or normalisation): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if pinned in seeds_ids:
        print(f"  WARNING: the query node {pinned} is in the context; removing it from the seeds "
              f"(otherwise every path is just 'query -> answer').")
        seeds_ids = [s for s in seeds_ids if s != pinned]

    ############# STEP 3: TCT expansion (adds real edges) #############
    tct_edges, tct_nodes = [], {}
    if USE_TCT:
        from tct_expansion import expand_context
        print("TCT expansion...")
        tct_edges, tct_nodes = expand_context(
            context_curies, TCT_CATEGORIES, max_neighbours=TCT_MAX_NEIGHBOURS,
            cache_path=os.path.join(DATA_DIR, "tct_cache.json"))
        tct_edges = [(alias.get(s, s), alias.get(o, o), p) for s, o, p in tct_edges]
        bridged = ({n for s, o, _ in tct_edges for n in (s, o)} & set(export.nodes)) - set(seeds_ids) - {pinned}
        print(f"  +{len(tct_edges)} TCT edges; {len(bridged)} TCT nodes already in the UI graph (bridges)")

    ############# STEP 4: graph + influence #############
    kg = build_graph(export, tct_edges, tct_nodes, alias, undirected=True)
    print(f"Graph: {kg.n} nodes, {len(kg.src) // 2} undirected edges")
    seeds = [kg.index[s] for s in seeds_ids if s in kg.index]
    if not seeds:
        print("No context node is connected to the graph - stopping.")
        return

    pi = random_walk_with_restart(kg, seeds, alpha=ALPHA)
    blocked = {kg.index[pinned]} if (BLOCK_QUERY_NODE and pinned in kg.index) else set()
    cost, parent, seed_of = most_probable_paths(kg, seeds, blocked)

    ############# STEP 5: best path per result #############
    rows = []
    for r in export.results:
        answer = alias.get(getattr(r, answer_field), getattr(r, answer_field))
        t = kg.index.get(answer)
        row = {"result_id": r.id, "answer_id": answer,
               "answer_name": r.name or (kg.names.get(answer) if t is not None else answer),
               "ui_score": r.score}
        if t is None or not np.isfinite(cost[t]):
            row.update(context_influence=0.0 if t is None else float(pi[t]),
                       path_probability=0.0, hops=None, context_seed=None,
                       best_path="(no path from context)", hop_sources="")
        else:
            path = reconstruct(parent, t)
            row.update(context_influence=float(pi[t]),
                       path_probability=float(np.exp(-cost[t])),
                       hops=len(path) - 1,
                       context_seed=kg.names[kg.ids[seed_of[t]]],
                       best_path=describe_path(kg, path),
                       hop_sources=path_sources(kg, path))
        rows.append(row)

    df = pd.DataFrame(rows)
    # rank answers by how much context random-walk mass reaches them
    df["context_influence_rank"] = df["context_influence"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values(["context_influence", "path_probability"], ascending=False)
    out = os.path.join(DATA_DIR, "best_influence_paths.csv")
    df.to_csv(out, index=False)
    print(f"\nBest paths saved to {out}")
    with pd.option_context("display.max_colwidth", 90, "display.width", 200):
        print(df[["answer_name", "context_influence", "path_probability", "hops", "best_path"]].head(10))

    # sanity check: are paths actually diverse?
    found = df[df["hops"].notna()]
    if len(found):
        top = Counter(found["best_path"]).most_common(1)[0]
        print(f"\n{found['best_path'].nunique()} distinct paths for {len(found)} reachable answers "
              f"(most frequent used {top[1]}x); {len(df) - len(found)} answers unreachable from context")

    ############# STEP 6: influence graph on REAL edges #############
    flow = edge_influence(kg, pi, ALPHA)
    keep = flow > EDGE_FLOW_THRESHOLD
    ig = pd.DataFrame({
        "source_id": [kg.ids[i] for i in kg.src[keep]],
        "target_id": [kg.ids[j] for j in kg.dst[keep]],
        "influence_flow": flow[keep],
    })
    ig["source_name"] = ig["source_id"].map(kg.names)
    ig["source_type"] = ig["source_id"].map(kg.types)
    ig["target_name"] = ig["target_id"].map(kg.names)
    ig["target_type"] = ig["target_id"].map(kg.types)
    ig["predicates"] = [
        "|".join(sorted(kg.predicates.get((s, t), set()) | kg.predicates.get((t, s), set())))
        for s, t in zip(ig["source_id"], ig["target_id"])]
    ig = ig.sort_values("influence_flow", ascending=False)
    out = os.path.join(DATA_DIR, "influence_graph.csv")
    ig.to_csv(out, index=False)
    print(f"Influence graph saved to {out} ({len(ig)} directed flows on real edges)")

    nodes_out = pd.DataFrame({"id": kg.ids, "name": [kg.names[x] for x in kg.ids],
                              "type": [kg.types[x] for x in kg.ids], "context_influence": pi,
                              "is_context": [i in set(seeds) for i in range(kg.n)]})
    nodes_out.sort_values("context_influence", ascending=False).to_csv(
        os.path.join(DATA_DIR, "node_influence.csv"), index=False)


if __name__ == "__main__":
    main()
