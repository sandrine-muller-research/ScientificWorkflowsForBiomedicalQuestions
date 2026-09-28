import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from UI_parser import load_export, TranslatorExport
from graph_analysis import (build_alias_map, resolve_answers, build_graph, signed_rwr,
                            strongest_chain_paths, path_flows, reconstruct, describe_path,
                            path_columns, expand_both_sides, score_ui_path,
                            context_weighted_graph, _rwr, null_influence, hub_normalise, empirical_z,
                            k_best_chains, path_type_diversity)

# ------------------------------------------------------------------ settings
DATA_DIR = "data/"
EXPORT_DIRS = ["targets", "conditions"]   # sub-folders of DATA_DIR holding UI exports (.json)
TARGETS_DIRNAME = "targets"               # exports whose SOURCE is the query node (the intervention);
                                          # in every other folder the sources are the context seeds
CONTEXT_BETA = 1.0                        # how strongly the context field bends the walk (0 = not at all)
CONTEXT_FLOOR = 0.05                      # minimum multiplier on an arc (keeps repelled arcs > 0)
HUB_CORRECTION = "enrichment"             # "enrichment": score with pi / pi_null (pi_null = PageRank, what a random
                                          #   source gives each node) so hubs are not favoured; "none": raw pi
Z_SCORE_RANDOM_SOURCES = 0                # > 0: also compute an empirical z-score / p-value per answer from this
                                          #   many degree-matched random source sets (slower; ~200 is reasonable)
MAPPING_FILE = "context_mapping_v2.csv"   # columns: corresponding_id, [direction: +1 attractor / -1 repeller]
USE_TCT = True                            # two-sided TCT expansion (context seeds + answers), bridges only
TCT_CATEGORIES = ["Gene", "Protein"]
TCT_MAX_NEIGHBOURS = 50                   # max bridging neighbours kept per seed (after the bridge filter)
REVERSE_WEIGHT = 0.5                      # directed graph; walking an edge against its assertion uses the
                                          # Biolink inverse predicate with this weight (0 = strictly as asserted)
REPULSION_WEIGHT = 0.5                    # each repeller (direction -1) counts this much relative to an
                                          # attractor (+1) of the same |weight|; seeds are weighted per gene
RWR_TOL = 1e-12                           # diffusion runs until the L1 change per iteration < RWR_TOL
ALPHA = 0.85                              # RWR continuation probability (restart = 1 - ALPHA)
BLOCK_QUERY_NODE = True                   # query (pinned) nodes are never path intermediates (they can
                                          # still be path endpoints, and seeds if they are in the context)
EXCLUDE_QUERY_NODES_FROM_SEEDS = False    # True: a query node listed in the context is dropped from the seeds
                                          # (only sensible when the context contains a queried condition)
EDGE_FLOW_THRESHOLD = 1e-4
TYPE_DIVERSITY_WEIGHT = 1.0               # rank_score = influence * (1 + w * type_diversity); type_diversity =
                                          # distinct coarse Biolink entity classes / nodes on the path. 0 = ignore.
PATHS_PER_ANSWER = 5                      # computed chains kept per answer (node-disjoint in their intermediates)
MAX_PATH_HOPS = 6
WRITE_TRAPI = True                        # write data/results.trapi.json: ONE TRAPI 1.5 Response, all exports merged,
                                          # answers ranked globally by relative_score (score / best score of its export)
TRAPI_QUERY_ROLES = ("intervention",)     # which query types go into results.trapi.json: "intervention" (targets/)
                                          # and/or "context" (conditions/). Default: intervention -> target paths only.
TRAPI_GROUP_BY_INTERMEDIATES = True       # order results by shared intermediate nodes and add a "group view" support
                                          # graph (all paths through that node) to each member result
TRAPI_PER_EXPORT_FILES = False            # True: also one file per export in data/trapi/
RERANK_BY = "bottleneck_flow"             # score used to rerank the original UI paths:
                                          # "bottleneck_flow" (weakest hop) | "total_flow" | "mean_intermediate_influence"


def load_and_merge(folders):
    """
    Load every UI export and merge them into one graph.

    Answers are resolved PER EXPORT (each export is its own query with its own
    pinned node) before merging. Edge and path ids are local to each file
    ('e1', 'p0' ... can repeat across files), so they are namespaced by file.
    """
    merged_nodes, merged_edges, merged_paths, merged_pubs, merged_trials = {}, {}, {}, {}, {}
    records, pinned_nodes, metas = [], set(), []
    ui_paths = {}                                   # file -> (paths, edges, query node), local ids
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for fname in sorted(f for f in os.listdir(folder) if f.endswith(".json") and not f.startswith("tct_cache")):
            path = os.path.join(folder, fname)
            try:
                exp = load_export(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
            tag = f"{os.path.basename(folder)}/{fname}"
            pinned, answers = resolve_answers(exp, build_alias_map(exp))
            if pinned:
                pinned_nodes.add(pinned)
            n_ans = len({a for a in answers if a})
            print(f"  {tag}: {len(exp.results)} results -> {n_ans} distinct answers; query node = {pinned}")
            records += [(tag, r, a) for r, a in zip(exp.results, answers)]
            ui_paths[tag] = (exp.paths, exp.edges, pinned)
            for nid, node in exp.nodes.items():
                merged_nodes.setdefault(nid, node)
            merged_edges.update({f"{tag}::{k}": v for k, v in exp.edges.items()})
            merged_paths.update({f"{tag}::{k}": v for k, v in exp.paths.items()})
            merged_pubs.update(exp.publications)
            merged_trials.update(exp.trials)
            metas.append(exp.meta)
    if not records and not merged_nodes:
        return None, [], set(), {}
    export = TranslatorExport(meta=metas[0] if metas else {}, results=tuple(r for _, r, _ in records),
                              nodes=merged_nodes, edges=merged_edges, paths=merged_paths,
                              publications=merged_pubs, trials=merged_trials, raw={})
    return export, records, pinned_nodes, ui_paths


def _rwr_from(kg, source_idx, alpha, tol):
    """RWR restarting uniformly on the given source indices."""
    s = np.zeros(kg.n)
    s[source_idx] = 1.0 / len(source_idx)
    pi, _ = _rwr(kg, s, alpha, tol, 100_000)
    return pi


def save_null_model(kg, export, out_dir):
    """
    Save the null model AFTER TCT expansion: the exact graph the diffusion runs on
    (merged UI exports + TCT edges, CURIEs mapped to UI ids, hierarchical edges removed).
      null_model_nodes.csv : id, name, type, in_ui (False = added by TCT)
      null_model_edges.csv : one row per asserted subject-predicate-object, with source UI/TCT
      null_model_arcs.csv  : directed arcs used by the random walk (asserted / symmetric /
                             inverse) with their transition probability
    """
    pd.DataFrame({"id": kg.ids, "name": [kg.names[x] for x in kg.ids],
                  "type": [kg.types[x] for x in kg.ids],
                  "in_ui": [x in export.nodes for x in kg.ids]}
                 ).to_csv(os.path.join(out_dir, "null_model_nodes.csv"), index=False)
    pd.DataFrame([{"subject": s, "predicate": p, "object": o,
                   "source": "+".join(sorted(kg.origin.get((s, o), set())))}
                  for (s, o), preds in kg.predicates.items() for p in sorted(preds)]
                 ).to_csv(os.path.join(out_dir, "null_model_edges.csv"), index=False)
    pd.DataFrame({"source_id": [kg.ids[u] for u in kg.src], "target_id": [kg.ids[v] for v in kg.dst],
                  "predicate": ["/".join(sorted(kg.arc_label[(u, v)])) for u, v in zip(kg.src, kg.dst)],
                  "arc_kind": [kg.arc_kind[(u, v)] for u, v in zip(kg.src, kg.dst)],
                  "transition_prob": kg.prob}
                 ).to_csv(os.path.join(out_dir, "null_model_arcs.csv"), index=False)


def read_context(path, alias):
    """{node_id: signed weight}. No 'direction' column or empty value -> +1; 0 -> ignored."""
    df = pd.read_csv(path)
    has_dir = "direction" in df.columns
    weights, n_default, n_zero = {}, 0, 0
    for _, row in df.iterrows():
        curie = row.get("corresponding_id")
        if pd.isna(curie):
            continue
        w = row["direction"] if has_dir else np.nan
        if pd.isna(w):
            w, n_default = 1.0, n_default + 1
        w = float(w)
        if w == 0:
            n_zero += 1
            continue
        node = alias.get(str(curie), str(curie))
        weights[node] = weights.get(node, 0.0) + w
    if not has_dir:
        print("  no 'direction' column: all context nodes weighted +1")
    elif n_default:
        print(f"  {n_default} context rows without direction -> +1")
    if n_zero:
        print(f"  {n_zero} context rows with direction 0 ignored")
    return {k: v for k, v in weights.items() if v != 0}


def main():
    ############# STEP 1: load and merge UI exports #############
    print("Loading UI exports...")
    export, records, pinned_nodes, ui_paths = load_and_merge([os.path.join(DATA_DIR, d) for d in EXPORT_DIRS])
    if export is None:
        print("No JSON exports found. Exiting.")
        return
    alias = build_alias_map(export)
    print(f"Merged graph: {len(export.nodes)} nodes, {len(export.edges)} edges, "
          f"{len(records)} results, query nodes: {sorted(pinned_nodes)}")

    ############# STEP 2: signed context #############
    mapping_path = os.path.join(DATA_DIR, MAPPING_FILE)
    if not os.path.exists(mapping_path):
        print(f"No {MAPPING_FILE} found. Exiting.")
        return
    print("Context:")
    context = read_context(mapping_path, alias)
    for q in sorted(pinned_nodes & set(context)):
        if EXCLUDE_QUERY_NODES_FROM_SEEDS:
            print(f"  query node {q} is in the context; removed from the seeds (EXCLUDE_QUERY_NODES_FROM_SEEDS)")
            del context[q]
        else:
            print(f"  query node {q} is in the context; kept as a seed"
                  f"{' (blocked as intermediate)' if BLOCK_QUERY_NODE else ''}")
    n_up = sum(w > 0 for w in context.values())
    n_down = sum(w < 0 for w in context.values())
    print(f"  {len(context)} seeds: {n_up} attractors, {n_down} repellers; "
          f"{sum(c in export.nodes for c in context)} present in the UI graph")

    ############# STEP 3: TCT expansion (adds real edges) #############
    tct_edges, tct_nodes, tct_edge_details = [], {}, {}
    if USE_TCT:
        print("TCT expansion...")
        answer_ids = [a for _, _, a in records]
        rank_by = None
        if HUB_CORRECTION == "enrichment":
            # rank candidate bridges by how enriched the UI graph already is around them
            # (context field / PageRank on the UI-only graph); unknown (TCT-only) nodes -> 1
            kg0 = build_graph(export, [], {}, alias, reverse_weight=REVERSE_WEIGHT)
            f0 = signed_rwr(kg0, context, alpha=ALPHA, tol=RWR_TOL, repulsion_weight=REPULSION_WEIGHT)["net"]
            e0 = np.abs(f0) / np.maximum(null_influence(kg0, ALPHA, RWR_TOL), 1e-15)
            rank_by = lambda x: float(e0[kg0.index[x]]) if x in kg0.index else 1.0
        tct_edges, tct_nodes, rep_, tct_edge_details = expand_both_sides(
            list(context), answer_ids, export.nodes.keys(), TCT_CATEGORIES, alias=alias,
            max_per_seed=TCT_MAX_NEIGHBOURS, cache_path=os.path.join(DATA_DIR, "tct_cache.json"),
            rank_by=rank_by)
        tct_edges = [(alias.get(s, s), alias.get(o, o), p) for s, o, p in tct_edges]
        tct_edge_details = {(alias.get(s, s), alias.get(o, o)): v for (s, o), v in tct_edge_details.items()}
        print(f"  TCT added {len(tct_edges)} edges via {len(tct_nodes)} bridging nodes: "
              f"{rep_.get('bridges_to_ui', 0)} to UI nodes, {rep_.get('bridges_to_answers', 0)} to answers, "
              f"{rep_.get('bridges_seed_seed', 0)} seed-seed; {rep_.get('dropped_dead_ends', 0)} dead ends dropped")

    ############# STEP 4: graph + diffusion #############
    kg = build_graph(export, tct_edges, tct_nodes, alias, reverse_weight=REVERSE_WEIGHT)
    save_null_model(kg, export, DATA_DIR)
    n_tct = sum(x not in export.nodes for x in kg.ids)
    print(f"Null model (after TCT expansion) saved: {kg.n} nodes ({n_tct} from TCT), "
          f"{sum(len(p) for p in kg.predicates.values())} edges -> null_model_*.csv")
    kinds = Counter(kg.arc_kind.values())
    print(f"Graph: {kg.n} nodes, {len(kg.src)} arcs ({kinds['asserted']} asserted, "
          f"{kinds['symmetric']} symmetric, {kinds['inverse']} inverse at weight {REVERSE_WEIGHT})")

    ############# STEP 4b: context field (state of the network) #############
    rwr = signed_rwr(kg, context, alpha=ALPHA, tol=RWR_TOL, repulsion_weight=REPULSION_WEIGHT)
    print(f"Context field: attractors {rwr['mass']['up']:.0%}, repellers {rwr['mass']['down']:.0%} of the restart "
          f"budget (per-gene weights, repellers x {REPULSION_WEIGHT})")
    for side, info in rwr["info"].items():
        print(f"  diffusion ({side}) stabilised after {info['iterations']} iterations (L1 change {info['residual']:.1e})")
    field = rwr["net"]
    ctx_idx = {kg.index[k] for k in context if k in kg.index}
    blocked = {kg.index[q] for q in pinned_nodes if q in kg.index} if BLOCK_QUERY_NODE else set()

    # The context bends the walk: arcs into attractor-influenced nodes become more probable,
    # arcs into repeller-influenced nodes less (CONTEXT_BETA = 0 -> plain graph).
    kg_ctx = context_weighted_graph(kg, field, beta=CONTEXT_BETA, floor=CONTEXT_FLOOR)

    # hub correction: what a NON-specific source gives each node on the same graph
    pi_null = null_influence(kg_ctx, ALPHA, RWR_TOL)
    deg = np.bincount(kg.src, minlength=kg.n)
    top = np.argsort(-deg)[:3]
    print("Hub correction: " + HUB_CORRECTION + "; top-degree nodes: " +
          ", ".join(f"{kg.names[kg.ids[i]]} (deg {deg[i]}, null share {pi_null[i]:.3g})" for i in top))

    ############# STEP 5: per-export SOURCE, diffusion from the source, best paths #############
    # targets/    : the source is the query node (the intervention); the context is the state it acts on
    # conditions/ : the sources are the context seeds
    by_file = defaultdict(list)
    for tag, r, a in records:
        by_file[tag].append((r, a))
    ctx_ids = set(context)

    rows, paths_by_row, prow = [], [], []
    all_path_rows = []                        # one row per path (computed + UI): the TRAPI results
    flow_total = np.zeros(len(kg.src))
    flow_raw_total = np.zeros(len(kg.src))
    node_infl = {}
    for tag, items in by_file.items():
        query = ui_paths[tag][2]
        kind = tag.split("/")[0]
        if kind == TARGETS_DIRNAME and query and query in kg.index:
            sources = {query: 1.0}
            mode = f"source = intervention {kg.names[query]} ({query})"
        else:
            sources = {k: w for k, w in context.items() if k in kg.index and k != query}
            mode = f"sources = {len(sources)} context seeds"
        if not sources:
            print(f"{tag}: no source in the graph, skipped")
            continue
        src_idx = [kg.index[k] for k in sources]
        pi_raw = _rwr_from(kg_ctx, src_idx, ALPHA, RWR_TOL)
        pi = hub_normalise(pi_raw, pi_null, src_idx) if HUB_CORRECTION == "enrichment" else pi_raw
        node_infl[tag] = (pi_raw, pi)
        z = p_emp = None
        if Z_SCORE_RANDOM_SOURCES > 0:
            z, p_emp = empirical_z(kg_ctx, pi_raw, src_idx, ALPHA, n_random=Z_SCORE_RANDOM_SOURCES)
        width, parent, seed_of, _ = strongest_chain_paths(kg_ctx, pi, src_idx, blocked, alpha=ALPHA)
        answers_here = {kg.index[a] for _, a in items if a in kg.index}
        n_reach = sum(width[t] > -np.inf for t in answers_here)
        print(f"{tag}: {mode}; answers reachable from source: {n_reach}/{len(answers_here)}")
        flow_total += ALPHA * pi[kg.src] * kg_ctx.prob
        flow_raw_total += ALPHA * pi_raw[kg.src] * kg_ctx.prob

        for r, answer in items:
            t = kg.index.get(answer) if answer else None
            row = {"source_file": tag, "result_id": r.id, "answer_id": answer,
                   "answer_name": kg.names.get(answer) if t is not None else (r.name or answer),
                   "ui_score": r.score,
                   "source_influence": float(pi[t]) if t is not None else 0.0,      # enrichment if HUB_CORRECTION
                   "raw_influence": float(pi_raw[t]) if t is not None else 0.0,
                   "null_influence": float(pi_null[t]) if t is not None else 0.0,
                   "degree": int(deg[t]) if t is not None else 0,
                   "z_score": float(z[t]) if (z is not None and t is not None) else float("nan"),
                   "empirical_p": float(p_emp[t]) if (p_emp is not None and t is not None) else float("nan"),
                   "context_field_at_answer": float(field[t]) if t is not None else 0.0}
            if t is None or width[t] == -np.inf:
                paths_by_row.append([])
                row.update(bottleneck_flow=0.0, hops=None, path_source="",
                           path_node_ids="", path_node_names="", path_node_types="", path_predicates="",
                           path_hop_kinds="", hop_flows="", context_on_path=0,
                           best_path="(no path from source)")
            else:
                path = reconstruct(parent, t)
                flows = path_flows(kg_ctx, pi, path, ALPHA)
                paths_by_row.append(path)
                row.update(bottleneck_flow=float(min(flows)) if flows else float("nan"),
                           hops=len(path) - 1,
                           path_source=kg.names[kg.ids[seed_of[t]]],
                           **path_columns(kg_ctx, path),
                           hop_flows="|".join(f"{f:.3g}" for f in flows),
                           context_on_path=sum(i in ctx_idx for i in path[1:-1]),
                           best_path=describe_path(kg_ctx, path))
            rows.append(row)

            # k best computed chains source -> answer: each becomes a result in the TRAPI file
            if t is not None:
                for kb_score, kb_path in k_best_chains(kg_ctx, pi, src_idx, t, k=PATHS_PER_ANSWER,
                                                       blocked=blocked, alpha=ALPHA, max_hops=MAX_PATH_HOPS):
                    seq = [kg.ids[i] for i in kb_path]
                    if kind != TARGETS_DIRNAME:
                        seq = seq[::-1]          # conditions: present drug -> ... -> context node
                    div, cls = path_type_diversity(kg, seq)
                    all_path_rows.append({"source_file": tag, "answer_id": answer,
                                          "answer_name": kg.names.get(answer, answer),
                                          "path_node_ids": "|".join(seq),
                                          "path_node_names": "|".join(str(kg.names.get(n, n)) for n in seq),
                                          "path_node_classes": "|".join(cls),
                                          "influence": kb_score, "type_diversity": div,
                                          "score": kb_score * (1 + TYPE_DIVERSITY_WEIGHT * div),
                                          "hops": len(seq) - 1,
                                          "origin": "computed", "ui_path_id": "", "ui_rank": float("nan")})
            # original UI paths of this result, scored with the SOURCE diffusion on the bent graph
            paths, edges, _ = ui_paths[tag]
            for ui_rank, pid in enumerate(r.path_ids, start=1):
                p = paths.get(pid)
                if p is None or not p.subgraph:
                    continue
                nodes = [alias.get(x, x) for x in p.subgraph[0::2]]
                preds, dirs = [], []
                for a_, eid, b_ in zip(p.subgraph[0::2], p.subgraph[1::2], p.subgraph[2::2]):
                    e = edges.get(eid)
                    pred = str(e.predicate or "").replace("biolink:", "") if e else "?"
                    fwd = e is not None and e.subject == a_ and e.object == b_
                    preds.append(pred if fwd else f"^{pred}")
                    dirs.append("forward" if fwd else "reverse")
                sc = score_ui_path(kg_ctx, pi, nodes, ALPHA, ctx_ids, pinned_nodes)
                seq_ui = nodes if not (kind == TARGETS_DIRNAME and nodes and nodes[0] == answer and nodes[-1] != answer) else nodes[::-1]
                div, cls = path_type_diversity(kg, seq_ui)
                all_path_rows.append({"source_file": tag, "answer_id": answer,
                                      "answer_name": kg.names.get(answer, answer),
                                      "path_node_ids": "|".join(seq_ui),
                                      "path_node_names": "|".join(str(kg.names.get(n, n)) for n in seq_ui),
                                      "path_node_classes": "|".join(cls),
                                      "influence": sc[RERANK_BY], "type_diversity": div,
                                      "score": sc[RERANK_BY] * (1 + TYPE_DIVERSITY_WEIGHT * div),
                                      "hops": len(seq_ui) - 1,
                                      "origin": "ui", "ui_path_id": pid, "ui_rank": float(ui_rank)})
                prow.append({
                    "source_file": tag, "result_id": r.id, "answer_id": answer,
                    "answer_name": kg.names.get(answer, r.name or answer), "ui_score": r.score,
                    "path_id": pid, "ui_path_rank": ui_rank, "hops": len(nodes) - 1,
                    "path_node_ids": "|".join(nodes),
                    "path_node_names": "|".join(str(kg.names.get(n, n)) for n in nodes),
                    "path_node_types": "|".join(str(kg.types.get(n, "Unknown")).replace("biolink:", "") for n in nodes),
                    "path_predicates": "|".join(preds),
                    "path_step_directions": "|".join(dirs),
                    "n_hierarchical_steps": sum(any(k in x.lower() for k in ("subclass", "superclass", "is_a"))
                                                for x in preds),
                    **sc,
                })

    share = Counter(n for p in paths_by_row for n in set(p[1:-1]))
    for row, p in zip(rows, paths_by_row):
        inter = p[1:-1]
        hub = max(inter, key=lambda n: share[n]) if inter else None
        row["most_shared_intermediate"] = kg.names[kg.ids[hub]] if hub is not None else ""
        row["n_results_sharing_it"] = share[hub] if hub is not None else 0

    df = pd.DataFrame(rows)
    df = df.sort_values(["source_file", "source_influence", "bottleneck_flow"], ascending=[True, False, False])
    out = os.path.join(DATA_DIR, "best_path_per_answer.csv")   # ONE best chain per answer (diagnostic);
                                                                 # results_table.csv mirrors results.trapi.json
    df.to_csv(out, index=False)
    print(f"\nBest paths saved to {out}")
    found = df[df["hops"].notna()]
    if len(found):
        print(f"{found['best_path'].nunique()} distinct paths for {len(found)} reachable results; "
              f"{len(df) - len(found)} unreachable; hops: {found['hops'].value_counts().sort_index().to_dict()}; "
              f"{int((found['context_on_path'] > 0).sum())} paths pass through a context node")
        for n, c in share.most_common(3):
            print(f"  shared intermediate: {kg.names[kg.ids[n]]} ({kg.ids[n]}) in {c} paths")

    ############# STEP 6: influence graph (flow from the sources, on the bent graph) #############
    keep = np.abs(flow_total) > EDGE_FLOW_THRESHOLD
    ig = pd.DataFrame({
        "source_id": [kg.ids[i] for i in kg.src[keep]],
        "target_id": [kg.ids[j] for j in kg.dst[keep]],
        "influence_flow": flow_total[keep],
        "raw_flow": flow_raw_total[keep],
        "context_weighted_prob": kg_ctx.prob[keep],
        "plain_prob": kg.prob[keep],
        "predicate": ["/".join(sorted(kg.arc_label[(u, v)])) for u, v in zip(kg.src[keep], kg.dst[keep])],
        "arc_kind": [kg.arc_kind[(u, v)] for u, v in zip(kg.src[keep], kg.dst[keep])],
    })
    ig["source_name"] = ig["source_id"].map(kg.names)
    ig["target_name"] = ig["target_id"].map(kg.names)
    ig = ig.sort_values("influence_flow", key=lambda s: s.abs(), ascending=False)
    ig.to_csv(os.path.join(DATA_DIR, "influence_graph.csv"), index=False)

    all_sources = {kg.index[q] for q in pinned_nodes if q in kg.index} | ctx_idx
    missing_arcs = [kg.ids[i] for i in all_sources if not keep[kg.src == i].any()]
    if missing_arcs:
        print(f"  NOTE: {len(missing_arcs)} source/context node(s) have no arc above EDGE_FLOW_THRESHOLD "
              f"in influence_graph.csv: {missing_arcs[:8]}")

    nd = pd.DataFrame({"id": kg.ids, "name": [kg.names[x] for x in kg.ids], "type": [kg.types[x] for x in kg.ids],
                       "context_field": field, "is_context": [i in ctx_idx for i in range(kg.n)]})
    nd["null_influence"] = pi_null
    nd["degree"] = deg
    for tag, (pr, pe) in node_infl.items():
        t_ = tag.replace('/', '_')
        nd[f"raw_influence_from_{t_}"] = pr
        nd[f"enrichment_from_{t_}"] = pe
    nd.sort_values("context_field", key=lambda s: s.abs(), ascending=False
                   ).to_csv(os.path.join(DATA_DIR, "node_influence.csv"), index=False)
    print(f"Influence graph ({len(ig)} arcs) and node influence saved in {DATA_DIR}")

    ############# STEP 7: rerank the ORIGINAL UI paths by source influence #############
    dp = pd.DataFrame(prow)
    if dp.empty:
        print("No UI paths to rerank.")
        return
    key = dp[RERANK_BY].abs().fillna(-1)
    dp["_key"] = key
    # within each result: 1 = most influenced path (ties -> UI order)
    dp = dp.sort_values(["source_file", "result_id", "_key", "ui_path_rank"],
                        ascending=[True, True, False, True])
    dp["influence_path_rank"] = dp.groupby(["source_file", "result_id"]).cumcount() + 1
    dp["path_rank_change"] = dp["ui_path_rank"] - dp["influence_path_rank"]    # > 0 = moved up
    dp["global_path_rank"] = dp["_key"].rank(ascending=False, method="min").astype(int)
    out = os.path.join(DATA_DIR, "ui_paths_reranked.csv")
    dp.drop(columns="_key").sort_values(["global_path_rank", "ui_path_rank"]).to_csv(out, index=False)

    # results reranked by their most influenced UI path, next to the UI's own order
    best = dp[dp["influence_path_rank"] == 1].copy()
    order = {(t, r.id): i for i, (t, r, _) in enumerate(records)}
    best["ui_result_rank"] = [order[(t, rid)] for t, rid in zip(best["source_file"], best["result_id"])]
    best["ui_result_rank"] = best.groupby("source_file")["ui_result_rank"].rank(method="first").astype(int)
    best["influence_result_rank"] = best.groupby("source_file")["_key"].rank(
        ascending=False, method="min").astype(int)
    best["result_rank_change"] = best["ui_result_rank"] - best["influence_result_rank"]
    cols = ["source_file", "result_id", "answer_id", "answer_name", "ui_score", "ui_result_rank",
            "influence_result_rank", "result_rank_change", RERANK_BY, "path_id", "ui_path_rank",
            "path_node_names", "path_predicates"]
    cols = list(dict.fromkeys(cols))
    out2 = os.path.join(DATA_DIR, "results_reranked.csv")
    best[cols].sort_values(["source_file", "influence_result_rank"]).to_csv(out2, index=False)

    ############# STEP 8: TRAPI export of the reranked paths #############
    if WRITE_TRAPI:
        from trapi_export import write_trapi_responses
        edge_flow = {(kg.ids[u], kg.ids[v]): float(f) for u, v, f in zip(kg.src, kg.dst, flow_total)}
        pr = pd.DataFrame(all_path_rows)
        # dedupe identical node sequences (keep the best-scored copy; note both origins)
        pr = pr.sort_values("score", ascending=False)
        pr["origin"] = pr.groupby(["source_file", "path_node_ids"])["origin"].transform(lambda s: "+".join(sorted(set(s))))
        pr = pr.drop_duplicates(["source_file", "path_node_ids"], keep="first")
        pr["rank_in_export"] = pr.groupby("source_file")["score"].rank(ascending=False, method="first").astype(int)
        pr.to_csv(os.path.join(DATA_DIR, "influential_paths.csv"), index=False)
        print(f"\nInfluential paths (one per future TRAPI result): {len(pr)} "
              f"({int((pr.origin.str.contains('computed')).sum())} computed, {int((pr.origin == 'ui').sum())} UI-only); "
              f"rank_score = {RERANK_BY} * (1 + {TYPE_DIVERSITY_WEIGHT} * type_diversity)")
        if TYPE_DIVERSITY_WEIGHT:
            r_inf = pr["influence"].rank(ascending=False); r_sc = pr["score"].rank(ascending=False)
            print(f"  type diversity: mean {pr['type_diversity'].mean():.2f}; "
                  f"{int((abs(r_inf - r_sc) >= 5).sum())} paths moved >= 5 ranks vs influence alone")
        write_trapi_responses(pr, ui_paths, export, alias, kg, DATA_DIR,
                              edge_flow=edge_flow, targets_dirname=TARGETS_DIRNAME, score_col=RERANK_BY,
                              per_export_files=TRAPI_PER_EXPORT_FILES, tct_details=tct_edge_details,
                              query_roles=TRAPI_QUERY_ROLES, group_by_intermediates=TRAPI_GROUP_BY_INTERMEDIATES)
        # path groups table: which paths share which intermediate node
        rows_g = []
        for _, r in pr.iterrows():
            for n in str(r["path_node_ids"]).split("|")[1:-1]:
                rows_g.append({"source_file": r["source_file"], "intermediate_id": n,
                               "intermediate_name": kg.names.get(n, n), "answer_id": r["answer_id"],
                               "answer_name": r["answer_name"], "path_node_names": r["path_node_names"],
                               "score": r["score"], "hops": r["hops"]})
        if rows_g:
            pg = pd.DataFrame(rows_g)
            cnt = pg.groupby(["source_file", "intermediate_id"])["answer_id"].transform("nunique")
            pg["n_targets_via_this_node"] = cnt
            pg = pg.sort_values(["source_file", "n_targets_via_this_node", "intermediate_id", "score"],
                                ascending=[True, False, True, False])
            pg.to_csv(os.path.join(DATA_DIR, "path_groups.csv"), index=False)
            top = pg.drop_duplicates(["source_file", "intermediate_id"]).head(5)
            print("  intermediates shared by most targets: " +
                  ", ".join(f"{r.intermediate_name} ({r.n_targets_via_this_node})" for r in top.itertuples()))

    print(f"\nUI paths reranked by {RERANK_BY}: {out} ({len(dp)} paths); results: {out2}")
    moved = (dp["path_rank_change"] != 0).groupby([dp["source_file"], dp["result_id"]]).any()
    print(f"  top path changed for {int((best['ui_path_rank'] != 1).sum())}/{len(best)} results; "
          f"{int(moved.sum())} results have their paths reordered")
    for tag, g in best.groupby("source_file"):
        if len(g) > 2:
            # Spearman = Pearson on the ranks (both columns already are ranks) -> no scipy needed
            rho = float(np.corrcoef(g["ui_result_rank"], g["influence_result_rank"])[0, 1])
            print(f"  {tag}: Spearman(UI rank, influence rank) = {rho:.2f} over {len(g)} results")


if __name__ == "__main__":
    main()
