"""
Context-influence analysis on Translator UI exports (+ optional TCT edges).

Model
-----
* DIRECTED graph. Each asserted edge  s -p-> o  gives an arc s -> o (weight 1).
  Walking it backwards uses the Biolink inverse predicate (affects -> affected_by)
  with weight REVERSE_WEIGHT; symmetric predicates (interacts_with, related_to...)
  go both ways with weight 1. REVERSE_WEIGHT = 0 -> strictly as asserted.
* Hierarchical edges (subclass_of / superclass_of / is_a) are dropped.
* Transition matrix is ROW-stochastic: P[u, v] = w(u, v) / sum_v w(u, v).
* Node influence = Random Walk with Restart iterated UNTIL STABILISATION.
  Signed context: direction +1 = attractor, -1 = repeller. Each seed counts by its
  own |weight| (repellers x REPULSION_WEIGHT) in one restart budget of mass 1; the
  two sides are diffused SEPARATELY and combined as
  net = m_att * pi_att - m_rep * pi_rep  (m = each side's share of the budget).
* Edge influence = flow through a real arc:  alpha * pi[u] * P[u, v].
* Best path to an answer = strongest chain: maximise the weakest hop's flow,
  from the seeds of the sign that dominates at the answer. Query (pinned) nodes
  are never intermediates.

Only numpy / pandas are required (requests + TCT for the optional expansion).
"""
from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import json
import os

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Identifier handling
# --------------------------------------------------------------------------- #
def build_alias_map(export) -> dict[str, str]:
    """Map every equivalent CURIE listed on a UI node to that node's primary id."""
    alias = {}
    for node_id, node in export.nodes.items():
        for curie in (node_id, *node.curies):
            alias.setdefault(str(curie), node_id)
    return alias


def resolve_answers(export, alias: dict[str, str]) -> tuple[str | None, list[str | None]]:
    """
    Find the answer node of every result from the result's own UI paths.

    Every UI path runs answer -> ... -> query node, so its two endpoints are the
    answer and the pinned query node. The pinned node is the endpoint shared by
    (almost) all results; the answer is the other endpoint. result.subject /
    result.object are used only as a tie-breaker, and result.name (drug_name)
    as a last resort.

    Returns (pinned_node_id, [answer_id per result, in export.results order]).
    """
    def a(x):
        return alias.get(x, x)

    ends_per_result = []
    for r in export.results:
        ends = Counter()
        for pid in r.path_ids:
            p = export.paths.get(pid)
            if p is not None and p.subgraph:
                ends[a(p.subgraph[0])] += 1
                ends[a(p.subgraph[-1])] += 1
        for x in (r.subject, r.object):
            if x:
                ends[a(x)] += 0.5
        ends_per_result.append(ends)

    presence = Counter(n for ends in ends_per_result for n in ends)
    pinned = None
    if presence and len(export.results) > 1:
        node, count = presence.most_common(1)[0]
        if count >= 0.8 * len(export.results):
            pinned = node

    name_index = {}
    for node_id, node in export.nodes.items():
        for nm in node.names:
            name_index.setdefault(str(nm).lower(), node_id)

    answers = []
    for r, ends in zip(export.results, ends_per_result):
        cands = [(c, n) for n, c in ends.items() if n != pinned]
        if cands:
            answers.append(max(cands)[1])
        elif r.name and str(r.name).lower() in name_index:
            answers.append(name_index[str(r.name).lower()])
        else:
            answers.append(None)
    return pinned, answers


def detect_query_roles(export) -> tuple[str | None, str]:
    """
    Return (pinned_node_id, answer_field).

    In UI exports one end of every result is the pinned query node (e.g. the
    disease in "what drugs may treat X") and the other end is the answer.
    The pinned end is the one that is identical across all results.
    """
    subjects = {r.subject for r in export.results if r.subject}
    objects = {r.object for r in export.results if r.object}
    if len(objects) == 1 and len(subjects) > 1:
        return next(iter(objects)), "subject"
    if len(subjects) == 1 and len(objects) > 1:
        return next(iter(subjects)), "object"
    if len(objects) == 1 and len(subjects) == 1:
        return None, "subject"          # single result, ambiguous
    return None, "subject"              # both vary: no single pinned node


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
@dataclass
class KG:
    ids: list[str]
    index: dict[str, int]
    names: dict[str, str]
    types: dict[str, str]
    src: np.ndarray            # directed arc arrays (both directions for undirected)
    dst: np.ndarray
    prob: np.ndarray           # P[src, dst], row-stochastic
    predicates: dict[tuple[str, str], set[str]] = field(default_factory=dict)   # as asserted
    origin: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    arc_label: dict[tuple[int, int], set[str]] = field(default_factory=dict)    # predicate read in walk direction
    arc_kind: dict[tuple[int, int], str] = field(default_factory=dict)          # asserted / symmetric / inverse

    @property
    def n(self) -> int:
        return len(self.ids)

    def neighbours(self):
        """Adjacency list of (v, P[u,v]) per u, built lazily."""
        if not hasattr(self, "_adj"):
            adj = [[] for _ in range(self.n)]
            for u, v, p in zip(self.src, self.dst, self.prob):
                adj[u].append((int(v), float(p)))
            self._adj = adj
        return self._adj


def build_graph(export, extra_edges=(), extra_nodes=None, alias=None,
                reverse_weight: float = 0.5) -> KG:
    """
    reverse_weight : weight of an inverse arc (walking an asymmetric edge against
                  its assertion, using the Biolink inverse predicate), relative to 1
                  for asserted and symmetric arcs.
    extra_edges : iterable of (subject, object, predicate) from TCT, ids already
                  mapped with `alias` where possible.
    extra_nodes : {id: {"name": ..., "categories": [...]}} for TCT-only nodes.
    """
    alias = alias or {}
    extra_nodes = extra_nodes or {}
    predicates: dict[tuple[str, str], set[str]] = defaultdict(set)
    origin: dict[tuple[str, str], set[str]] = defaultdict(set)

    def add(s, o, p, where):
        s, o = alias.get(s, s), alias.get(o, o)
        if not s or not o or s == o:
            return
        pl = str(p or "").lower().replace(" ", "_")
        if "subclass" in pl or "superclass" in pl or pl.endswith("is_a"):
            return                       # hierarchical edges: not functional interactions
        predicates[(s, o)].add(str(p or "related_to").replace("biolink:", "").strip().replace(" ", "_"))
        origin[(s, o)].add(where)

    for e in export.edges.values():
        add(e.subject, e.object, e.predicate, "UI")
    for s, o, p in extra_edges:
        add(s, o, p, "TCT")

    node_ids = set(export.nodes)
    for s, o in predicates:
        node_ids.update((s, o))
    ids = sorted(node_ids)
    index = {x: i for i, x in enumerate(ids)}

    names, types = {}, {}
    for x in ids:
        node = export.nodes.get(x)
        if node is not None:
            names[x] = node.names[0] if node.names else x
            types[x] = node.types[0] if node.types else "Unknown"
        else:
            info = extra_nodes.get(x, {})
            names[x] = info.get("name") or x
            cats = info.get("categories") or []
            types[x] = cats[0] if cats else "Unknown"

    # Directed arcs. Every asserted edge s -p-> o gives:
    #   s -> o  labelled p                      weight 1          (asserted)
    #   o -> s  labelled p       if p symmetric weight 1          (symmetric)
    #   o -> s  labelled inverse(p) otherwise   weight reverse_weight (inverse)
    # Biolink defines an inverse for most predicates (affects <-> affected_by,
    # treats <-> treated_by, ...), which is how Translator/UI paths traverse
    # edges "backwards". reverse_weight = 0 -> strictly follow assertions;
    # reverse_weight = 1 -> inverse statements count as much as asserted ones.
    if not 0 <= reverse_weight <= 1:
        raise ValueError("reverse_weight must be in [0, 1]")
    rank = {"asserted": 3, "symmetric": 2, "inverse": 1}
    weight: dict[tuple[int, int], float] = {}
    arc_label: dict[tuple[int, int], set[str]] = defaultdict(set)
    arc_kind: dict[tuple[int, int], str] = {}

    def put(i, j, w, label, kind):
        if w <= 0:
            return
        weight[(i, j)] = max(weight.get((i, j), 0.0), w)
        arc_label[(i, j)].add(label)
        if rank[kind] > rank.get(arc_kind.get((i, j)), 0):
            arc_kind[(i, j)] = kind

    for (s, o), preds in predicates.items():
        i, j = index[s], index[o]
        for p in preds:
            put(i, j, 1.0, p, "asserted")
            if p in BIOLINK_SYMMETRIC:
                put(j, i, 1.0, p, "symmetric")
            else:
                put(j, i, reverse_weight, BIOLINK_INVERSE.get(p, f"inverse_of_{p}"), "inverse")

    if weight:
        arcs = sorted(weight)
        src = np.array([a for a, _ in arcs])
        dst = np.array([b for _, b in arcs])
        w = np.array([weight[a] for a in arcs])
    else:
        src = dst = np.array([], dtype=int)
        w = np.array([])
    out_w = np.bincount(src, weights=w, minlength=len(ids)) if len(src) else np.zeros(len(ids))
    prob = w / out_w[src] if len(src) else np.array([])

    return KG(ids, index, names, types, src, dst, prob, dict(predicates), dict(origin),
              dict(arc_label), arc_kind)


# --------------------------------------------------------------------------- #
# Influence
# --------------------------------------------------------------------------- #
def _rwr(kg: KG, s: np.ndarray, alpha: float, tol: float, max_iter: int):
    """RWR for ONE non-negative restart vector s (sum 1). Runs until stabilisation."""
    dangling = np.bincount(kg.src, minlength=kg.n) == 0   # nodes with no outgoing arc
    pi = s.copy()
    residual = np.inf
    for it in range(1, max_iter + 1):
        spread = np.bincount(kg.dst, weights=pi[kg.src] * kg.prob, minlength=kg.n)
        new = alpha * spread + (1 - alpha) * s + alpha * pi[dangling].sum() * s
        residual = float(np.abs(new - pi).sum())
        pi = new
        if residual < tol:
            return pi, {"iterations": it, "residual": residual, "converged": True}
    raise RuntimeError(f"RWR did not stabilise in {max_iter} iterations (residual {residual:.2e})")


def signed_rwr(kg: KG, context_weights, alpha: float = 0.85, tol: float = 1e-12,
               max_iter: int = 100_000, repulsion_weight: float = 1.0):
    """
    context_weights : {node_id: weight}  weight > 0 = ATTRACTOR, < 0 = REPELLER, 0 = ignored
                      (or a list of node ids / indices, all attractors with weight 1).
    repulsion_weight : multiplier applied to every repeller's |weight| (< 1 = attractors pull
                      harder than repellers push).

    Weighting is PER SEED: each seed contributes its own |weight| (x repulsion_weight for
    repellers) to one restart budget of total mass 1. So the attractor side gets
    share m_att = A / (A + R), with A = sum of attractor weights and R = repulsion_weight *
    sum of repeller weights; more attractor genes -> more pull.

    Each side is diffused separately (required for correctness with dangling nodes):
        net = m_att * pi_att - m_rep * pi_rep
    Returns "net", "up" (= m_att * pi_att), "down" (= m_rep * pi_rep), "mass", "info".
    """
    if not isinstance(context_weights, dict):
        context_weights = {x: 1.0 for x in context_weights}
    if repulsion_weight < 0:
        raise ValueError("repulsion_weight must be >= 0")
    att, rep = np.zeros(kg.n), np.zeros(kg.n)
    for node, w in context_weights.items():
        i = node if isinstance(node, (int, np.integer)) else kg.index.get(node)
        if i is None or not w or not np.isfinite(w):
            continue
        if w > 0:
            att[i] += w
        else:
            rep[i] += abs(w) * repulsion_weight
    total = att.sum() + rep.sum()
    mass = {"up": att.sum() / total if total else 0.0, "down": rep.sum() / total if total else 0.0}
    out = {"up": np.zeros(kg.n), "down": np.zeros(kg.n), "info": {}, "mass": mass}
    for name, s in (("up", att), ("down", rep)):
        if s.sum() > 0:
            pi_side, out["info"][name] = _rwr(kg, s / s.sum(), alpha, tol, max_iter)
            out[name] = mass[name] * pi_side
    out["net"] = out["up"] - out["down"]
    return out


def random_walk_with_restart(kg: KG, context_weights, alpha: float = 0.85,
                             tol: float = 1e-12, max_iter: int = 100_000,
                             return_info: bool = False, repulsion_weight: float = 1.0):
    """Signed RWR (see signed_rwr). Returns net influence, or (net, info) if return_info."""
    r = signed_rwr(kg, context_weights, alpha, tol, max_iter, repulsion_weight)
    return (r["net"], r["info"]) if return_info else r["net"]


def context_weighted_graph(kg: KG, field: np.ndarray, beta: float = 1.0, floor: float = 0.05) -> KG:
    """
    Bend the random walk by a context field (e.g. net context influence):
        w'(u, v) = w(u, v) * max(floor, 1 + beta * field[v] / max|field|)
    then re-normalise rows. Arcs INTO attractor-influenced nodes get more probable,
    arcs into repeller-influenced nodes less; beta = 0 -> unchanged graph.
    Returns a new KG sharing everything but `prob`.
    """
    import copy
    scale = float(np.abs(field).max()) if kg.n and np.abs(field).max() > 0 else 1.0
    factor = np.maximum(floor, 1.0 + beta * field[kg.dst] / scale)
    # original weights: prob * out-degree-weight of u; recover w from prob by row sums
    w = kg.prob * factor
    row = np.bincount(kg.src, weights=w, minlength=kg.n)
    new = copy.copy(kg)
    new.prob = w / row[kg.src]
    for attr in ("_adj", "_p"):
        if hasattr(new, attr):
            delattr(new, attr)
    return new


def null_influence(kg: KG, alpha: float = 0.85, tol: float = 1e-12) -> np.ndarray:
    """
    Expected influence at each node for a NON-SPECIFIC source: RWR with a uniform
    restart on all nodes (= PageRank of the graph). Hubs get a large value simply
    because they are well connected. Used to turn raw influence into an enrichment.
    """
    s = np.full(kg.n, 1.0 / kg.n)
    pi, _ = _rwr(kg, s, alpha, tol, 100_000)
    return pi


def hub_normalise(pi: np.ndarray, pi_null: np.ndarray, sources=(), eps: float = 1e-15) -> np.ndarray:
    """
    Enrichment  e[v] = pi[v] / pi_null[v]   (1 = what a random source would give this
    node; > 1 = the source reaches v MORE than v's connectivity alone explains).
    Sources are set to the max enrichment so paths can still start there.
    """
    e = pi / np.maximum(pi_null, eps)
    if len(sources):
        e[list(sources)] = max(float(e.max()), 1.0)
    return e


def empirical_z(kg: KG, pi: np.ndarray, source_idx: list[int], alpha: float = 0.85,
                tol: float = 1e-10, n_random: int = 200, seed: int = 0,
                degree_matched: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    z-score of pi against RWRs from n_random random source sets of the same size,
    drawn from nodes of similar degree (degree_matched) or uniformly.
    Returns (z, empirical_p) with p = fraction of random runs >= pi (one-sided).
    """
    rng = np.random.default_rng(seed)
    deg = np.bincount(kg.src, minlength=kg.n)
    k = len(source_idx)
    if degree_matched:
        order = np.argsort(deg)
        rank = np.empty(kg.n, dtype=int)
        rank[order] = np.arange(kg.n)
        band = max(5, kg.n // 20)
        pools = [order[max(0, rank[s] - band): rank[s] + band + 1] for s in source_idx]
    samples = np.zeros((n_random, kg.n))
    for i in range(n_random):
        if degree_matched:
            picks = [int(rng.choice(p)) for p in pools]
        else:
            picks = list(rng.choice(kg.n, size=k, replace=False))
        s = np.zeros(kg.n)
        np.add.at(s, picks, 1.0 / k)
        samples[i], _ = _rwr(kg, s, alpha, tol, 100_000)
    mu, sd = samples.mean(0), samples.std(0)
    z = (pi - mu) / np.where(sd > 0, sd, np.inf)
    p = ((samples >= pi[None, :]).sum(0) + 1) / (n_random + 1)
    return z, p


def edge_influence(kg: KG, pi: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """Expected probability flow along each arc: alpha * pi[u] * P[u, v]."""
    return alpha * pi[kg.src] * kg.prob


def get_top_influence_edges(kg: KG, pi: np.ndarray, target_node_id: str,
                            alpha: float = 0.85, top_k: int = 5) -> list[dict]:
    """Incoming arcs of a node ranked by |influence flow| alpha * pi[u] * P[u, target]."""
    t = kg.index.get(target_node_id)
    if t is None:
        return []
    mask = kg.dst == t
    flows = alpha * pi[kg.src[mask]] * kg.prob[mask]
    order = np.argsort(-np.abs(flows))[:top_k]
    return [{"source_id": kg.ids[u], "source_name": kg.names[kg.ids[u]],
             "target_id": target_node_id, "target_name": kg.names[target_node_id],
             "predicate": "/".join(sorted(kg.arc_label.get((int(u), t), set()))),
             "flow_weight": float(f)}
            for u, f in zip(kg.src[mask][order], flows[order])]


# --------------------------------------------------------------------------- #
# Best paths
# --------------------------------------------------------------------------- #
def most_probable_paths(kg: KG, sources: list[int], blocked: set[int] = frozenset()):
    """
    Multi-source Dijkstra with cost -log P[u, v] on the real graph.
    Nodes in `blocked` can be reached (as endpoints) but never expanded,
    so they never appear as intermediates.

    Returns (cost, parent, origin_seed) arrays for all nodes at once.
    """
    adj = kg.neighbours()
    cost = np.full(kg.n, np.inf)
    parent = np.full(kg.n, -1, dtype=int)
    seed_of = np.full(kg.n, -1, dtype=int)
    pq = []
    for s in sources:
        cost[s] = 0.0
        seed_of[s] = s
        heapq.heappush(pq, (0.0, s))
    while pq:
        c, u = heapq.heappop(pq)
        if c > cost[u]:
            continue
        if u in blocked and seed_of[u] != u:
            continue
        for v, p in adj[u]:
            nc = c - math.log(p)
            if nc < cost[v] - 1e-15:
                cost[v] = nc
                parent[v] = u
                seed_of[v] = seed_of[u]
                heapq.heappush(pq, (nc, v))
    return cost, parent, seed_of


def strongest_chain_paths(kg: KG, pi: np.ndarray, sources: list[int],
                          blocked: set[int] = frozenset(), alpha: float = 0.85,
                          blocked_arcs: set[tuple[int, int]] = frozenset()):
    """
    Maximum-bottleneck ("widest") paths on the influence-flow graph.

    Each hop u->v carries the RWR flow  f(u,v) = alpha * pi[u] * P[u,v]
    (expected share of context influence moving along that edge). A chain is
    as strong as its weakest hop, so for every node we find the path from any
    context seed that MAXIMISES THE MINIMUM hop flow. Length is not penalised;
    among equally strong chains the one with fewer hops is kept.

    Nodes in `blocked` can be endpoints but never intermediates.
    Returns (width, parent, seed_of, hops); width = +inf on seeds, -inf if unreachable.
    """
    adj = kg.neighbours()
    width = np.full(kg.n, -np.inf)
    hops = np.full(kg.n, np.iinfo(np.int64).max, dtype=np.int64)
    parent = np.full(kg.n, -1, dtype=int)
    seed_of = np.full(kg.n, -1, dtype=int)
    pq = []
    sources = [s if isinstance(s, (int, np.integer)) else kg.index.get(s) for s in sources]
    for s in (int(x) for x in sources if x is not None):
        width[s], hops[s], seed_of[s] = np.inf, 0, s
        heapq.heappush(pq, (-np.inf, 0, s))
    while pq:
        neg_w, h, u = heapq.heappop(pq)
        w = -neg_w
        if w < width[u] or (w == width[u] and h > hops[u]):
            continue                                    # stale entry
        if u in blocked and seed_of[u] != u:
            continue                                    # never pass through the query node
        for v, p in adj[u]:
            if blocked_arcs and (u, v) in blocked_arcs:
                continue
            nw = min(w, alpha * pi[u] * p)
            nh = h + 1
            if nw > width[v] or (nw == width[v] and nh < hops[v]):
                width[v], hops[v], parent[v], seed_of[v] = nw, nh, u, seed_of[u]
                heapq.heappush(pq, (-nw, nh, v))
    return width, parent, seed_of, hops


def k_best_chains(kg: KG, pi: np.ndarray, sources: list[int], target: int, k: int = 5,
                  blocked: set[int] = frozenset(), alpha: float = 0.85, max_hops: int = 6):
    """
    Up to k distinct strongest chains from the sources to `target`.
    After each chain is found, its intermediate nodes are forbidden (or, for a direct
    source->target chain, that arc), and the search is re-run, so chains are
    node-disjoint in their intermediates. Returns [(bottleneck, path)] by bottleneck desc.
    """
    found, forbid_nodes, forbid_arcs = [], set(blocked), set()
    for _ in range(k):
        width, parent, _, _ = strongest_chain_paths(kg, pi, sources, forbid_nodes, alpha=alpha,
                                                    blocked_arcs=forbid_arcs)
        if width[target] == -np.inf:
            break
        path = reconstruct(parent, target)
        if len(path) - 1 > max_hops:
            break
        found.append((float(width[target]), path))
        inter = path[1:-1]
        if inter:
            forbid_nodes |= set(inter)
        else:
            forbid_arcs.add((path[0], path[-1]))
    found.sort(key=lambda x: -x[0])
    return found


def path_flows(kg: KG, pi: np.ndarray, path: list[int], alpha: float = 0.85) -> list[float]:
    """Influence flow alpha * pi[u] * P[u,v] of each hop of a path."""
    adj = kg.neighbours()
    out = []
    for u, v in zip(path, path[1:]):
        p = next(pp for vv, pp in adj[u] if vv == v)
        out.append(alpha * pi[u] * p)
    return out


def arc_prob(kg: KG, u: int, v: int) -> float:
    """P[u, v] (0 if there is no arc u -> v)."""
    if not hasattr(kg, "_p"):
        kg._p = {(int(a), int(b)): float(p) for a, b, p in zip(kg.src, kg.dst, kg.prob)}
    return kg._p.get((u, v), 0.0)


def score_ui_path(kg: KG, pi: np.ndarray, node_ids: list[str], alpha: float = 0.85,
                  context: set[str] = frozenset(), query_nodes: set[str] = frozenset()) -> dict:
    """
    Influence score of an ORIGINAL UI path (node sequence answer -> ... -> query).

    Context influence enters a UI path wherever it touches the diffused graph and
    can move along each hop in either direction, so each hop is scored with the
    flow through that edge both ways:
        hop_flow(u, v) = alpha * (pi[u] P[u,v] + pi[v] P[v,u])      (signed with pi)
    * bottleneck  = min |hop_flow|  -> a chain is as strong as its weakest hop
                    (same logic as the strongest-chain best paths)
    * mean_intermediate = mean pi of the intermediate nodes (not answer / query node)
    A hop whose edge is not in the graph (e.g. a filtered subclass_of edge) has flow 0.
    """
    idx = [kg.index.get(n) for n in node_ids]
    hops, missing = [], 0
    for u, v in zip(idx, idx[1:]):
        if u is None or v is None:
            hops.append(0.0)
            missing += 1
            continue
        puv, pvu = arc_prob(kg, u, v), arc_prob(kg, v, u)
        if puv == 0 and pvu == 0:
            missing += 1
        hops.append(alpha * (pi[u] * puv + pi[v] * pvu))
    inter = [i for n, i in zip(node_ids[1:-1], idx[1:-1]) if i is not None and n not in query_nodes]
    node_vals = [pi[i] for i in idx if i is not None]
    return {
        "bottleneck_flow": float(min(abs(h) for h in hops)) if hops else 0.0,
        "total_flow": float(sum(hops)),
        "mean_intermediate_influence": float(np.mean([pi[i] for i in inter])) if inter else float("nan"),
        "max_abs_node_influence": float(max(abs(x) for x in node_vals)) if node_vals else 0.0,
        "n_context_nodes_on_path": sum(n in context for n in node_ids),
        "n_missing_hops": missing,
        "hop_flows": "|".join(f"{h:.3g}" for h in hops),
    }


def reconstruct(parent: np.ndarray, target: int) -> list[int]:
    path, cur = [], target
    while cur != -1:
        path.append(int(cur))
        cur = parent[cur]
    return path[::-1]


def _hop(kg: KG, a: int, b: int) -> tuple[str, str]:
    return "/".join(sorted(kg.arc_label.get((a, b), {"?"}))), kg.arc_kind.get((a, b), "?")


def describe_path(kg: KG, path: list[int]) -> str:
    """'GENE2 -[affected_by*]-> drug21': predicates read in walk direction, * = inverse arc."""
    if not path:
        return ""
    out = [str(kg.names[kg.ids[path[0]]])]
    for a, b in zip(path, path[1:]):
        label, kind = _hop(kg, a, b)
        out.append(f" -[{label}{'*' if kind == 'inverse' else ''}]-> {kg.names[kg.ids[b]]}")
    return "".join(out)


def path_columns(kg: KG, path: list[int], sep: str = "|") -> dict[str, str]:
    """Pipe-separated node ids / names / types, predicate and arc kind of each hop."""
    ids = [kg.ids[i] for i in path]
    hops = [_hop(kg, a, b) for a, b in zip(path, path[1:])]
    return {
        "path_node_ids": sep.join(ids),
        "path_node_names": sep.join(str(kg.names[x]) for x in ids),
        "path_node_types": sep.join(str(kg.types[x]).replace("biolink:", "") for x in ids),
        "path_predicates": sep.join(h[0] for h in hops),
        "path_hop_kinds": sep.join(h[1] for h in hops),
    }


def hop_sources(kg: KG, path: list[int]) -> str:
    """UI / TCT provenance of each hop."""
    tags = []
    for a, b in zip(path, path[1:]):
        u, v = kg.ids[a], kg.ids[b]
        tags.append("+".join(sorted(kg.origin.get((u, v), set()) | kg.origin.get((v, u), set()))))
    return ",".join(tags)


def path_sources(kg: KG, path: list[int]) -> list[str]:
    """Nodes with an arc into the first node of the path."""
    if not path:
        return []
    return sorted({kg.ids[u] for u in kg.src[kg.dst == path[0]]})


# Biolink 4.4.5 category -> coarse entity class (for path type diversity)
BIOLINK_TYPE_CLASS = {'OrganismTaxon': 'organism_taxon', 'StudyVariable': 'information_content_entity', 'CommonDataElement': 'information_content_entity', 'InformationContentEntity': 'information_content_entity', 'Dataset': 'information_content_entity', 'DatasetDistribution': 'information_content_entity', 'DatasetVersion': 'information_content_entity', 'DatasetSummary': 'information_content_entity', 'ConfidenceLevel': 'information_content_entity', 'Evidence': 'information_content_entity', 'Publication': 'information_content_entity', 'Book': 'information_content_entity', 'BookChapter': 'information_content_entity', 'Serial': 'information_content_entity', 'Article': 'information_content_entity', 'JournalArticle': 'information_content_entity', 'Patent': 'information_content_entity', 'WebPage': 'information_content_entity', 'PreprintPublication': 'information_content_entity', 'DrugLabel': 'information_content_entity', 'RetrievalSource': 'information_content_entity', 'Procedure': 'procedure', 'Device': 'device', 'EnvironmentalFeature': 'environmental_feature', 'MolecularEntity': 'chemical_entity', 'ChemicalEntity': 'chemical_entity', 'SmallMolecule': 'chemical_entity', 'ChemicalMixture': 'chemical_entity', 'NucleicAcidEntity': 'chemical_entity', 'MolecularMixture': 'chemical_entity', 'ComplexMolecularMixture': 'chemical_entity', 'BiologicalProcessOrActivity': 'biological_process_or_activity', 'MolecularActivity': 'biological_process_or_activity', 'BiologicalProcess': 'biological_process_or_activity', 'Pathway': 'biological_process_or_activity', 'PhysiologicalProcess': 'biological_process_or_activity', 'Behavior': 'biological_process_or_activity', 'ProcessedMaterial': 'chemical_entity', 'Drug': 'chemical_entity', 'EnvironmentalFoodContaminant': 'chemical_entity', 'FoodAdditive': 'chemical_entity', 'Food': 'chemical_entity', 'DiseaseOrPhenotypicFeature': 'disease_or_phenotypic_feature', 'Disease': 'disease_or_phenotypic_feature', 'PhenotypicFeature': 'disease_or_phenotypic_feature', 'BehavioralFeature': 'disease_or_phenotypic_feature', 'AnatomicalEntity': 'anatomical_entity', 'CellularComponent': 'anatomical_entity', 'Cell': 'anatomical_entity', 'CellLine': 'cell_line', 'GrossAnatomicalStructure': 'anatomical_entity', 'Gene': 'gene_or_protein', 'Polypeptide': 'gene_or_protein', 'Protein': 'gene_or_protein', 'ProteinIsoform': 'gene_or_protein', 'ProteinFamily': 'gene_or_protein', 'GeneFamily': 'gene_or_protein', 'SequenceVariant': 'sequence_variant', 'Snv': 'sequence_variant', 'ClinicalFinding': 'disease_or_phenotypic_feature', 'PathologicalProcess': 'biological_process_or_activity', 'PathologicalAnatomicalStructure': 'anatomical_entity', 'Treatment': 'treatment'}


def path_type_diversity(kg: KG, node_ids: list[str]) -> tuple[float, list[str]]:
    """
    Fraction of DISTINCT coarse entity classes along a path, in (0, 1]:
        n_distinct_classes / n_nodes
    Classes come from the Biolink hierarchy (Gene/Protein -> gene_or_protein,
    SmallMolecule/Drug -> chemical_entity, Disease/PhenotypicFeature -> disease_or_phenotypic_feature,
    Pathway/BiologicalProcess -> biological_process_or_activity, ...), so a repeated Gene->Gene
    hop adds no diversity while chemical -> gene -> pathway -> disease scores 1.
    Returns (diversity, [class per node]).
    """
    classes = []
    for n in node_ids:
        t = str(kg.types.get(n, "")).replace("biolink:", "")
        classes.append(BIOLINK_TYPE_CLASS.get(t, t.lower() or "unknown"))
    return (len(set(classes)) / len(classes)) if classes else 0.0, classes


# Biolink model 4.4.5: inverse predicate pairs and symmetric predicates
# (from biolink-model.yaml slots: inverse / symmetric)
_BIOLINK_INVERSE_PAIRS = [
    ('active_in', 'has_active_component'),
    ('actively_involved_in', 'actively_involves'),
    ('acts_upstream_of', 'has_upstream_actor'),
    ('acts_upstream_of_negative_effect', 'has_negative_upstream_actor'),
    ('acts_upstream_of_or_within', 'has_upstream_or_within_actor'),
    ('acts_upstream_of_or_within_negative_effect', 'has_negative_upstream_or_within_actor'),
    ('acts_upstream_of_or_within_positive_effect', 'has_positive_upstream_or_within_actor'),
    ('acts_upstream_of_positive_effect', 'has_positive_upstream_actor'),
    ('adverse_event_of', 'has_adverse_event'),
    ('affected_by', 'affects'),
    ('affects_likelihood_of', 'likelihood_affected_by'),
    ('affects_sensitivity_to', 'sensitivity_affected_by'),
    ('ameliorates_condition', 'condition_ameliorated_by'),
    ('amount_or_activity_decreased_by', 'decreases_amount_or_activity_of'),
    ('amount_or_activity_increased_by', 'increases_amount_or_activity_of'),
    ('applied_to_treat', 'treatment_applications_from'),
    ('associated_with_decreased_likelihood_of', 'decreased_likelihood_associated_with'),
    ('associated_with_increased_likelihood_of', 'increased_likelihood_associated_with'),
    ('associated_with_likelihood_of', 'likelihood_associated_with'),
    ('associated_with_resistance_to', 'resistance_associated_with'),
    ('associated_with_response_to', 'response_associated_with'),
    ('associated_with_sensitivity_to', 'sensitivity_associated_with'),
    ('author', 'has_author'),
    ('beneficial_in_models_for', 'models_demonstrating_benefits_for'),
    ('biomarker_for', 'has_biomarker'),
    ('broad_match', 'narrow_match'),
    ('can_be_carried_out_by', 'capable_of'),
    ('catalyzes', 'has_catalyst'),
    ('caused_by', 'causes'),
    ('completed_by', 'has_completed'),
    ('composed_primarily_of', 'primarily_composed_of'),
    ('condition_associated_with_gene', 'gene_associated_with_condition'),
    ('condition_exacerbated_by', 'exacerbates_condition'),
    ('condition_predisposed_by', 'predisposes_to_condition'),
    ('condition_promoted_by', 'promotes_condition'),
    ('consumed_by', 'consumes'),
    ('contains_process', 'occurs_in'),
    ('contraindicated_in', 'has_contraindication'),
    ('contributes_to', 'contribution_from'),
    ('contributor', 'has_contributor'),
    ('decreased_amount_in', 'has_decreased_amount'),
    ('decreases_sensitivity_to', 'sensitivity_decreased_by'),
    ('derives_from', 'derives_into'),
    ('develops_from', 'develops_into'),
    ('diagnoses', 'is_diagnosed_by'),
    ('disease_has_basis_in', 'occurs_in_disease'),
    ('disease_has_location', 'location_of_disease'),
    ('disrupted_by', 'disrupts'),
    ('editor', 'has_editor'),
    ('enabled_by', 'enables'),
    ('expressed_in', 'expresses'),
    ('food_component_of', 'has_food_component'),
    ('gene_product_of', 'has_gene_product'),
    ('has_active_ingredient', 'is_active_ingredient_of'),
    ('has_chemical_role', 'is_chemical_role_of'),
    ('has_excipient', 'is_excipient_of'),
    ('has_frameshift_variant', 'is_frameshift_variant_of'),
    ('has_increased_amount', 'increased_amount_of'),
    ('has_input', 'is_input_of'),
    ('has_manifestation', 'manifestation_of'),
    ('has_member', 'member_of'),
    ('has_metabolite', 'is_metabolite_of'),
    ('has_missense_variant', 'is_missense_variant_of'),
    ('has_mode_of_inheritance', 'mode_of_inheritance_of'),
    ('has_molecular_consequence', 'is_molecular_consequence_of'),
    ('has_nearby_variant', 'is_nearby_variant_of'),
    ('has_non_coding_variant', 'is_non_coding_variant_of'),
    ('has_nonsense_variant', 'is_nonsense_variant_of'),
    ('has_not_completed', 'not_completed_by'),
    ('has_nutrient', 'nutrient_of'),
    ('has_output', 'is_output_of'),
    ('has_part', 'part_of'),
    ('has_participant', 'participates_in'),
    ('has_phenotype', 'phenotype_of'),
    ('has_plasma_membrane_part', 'plasma_membrane_part_of'),
    ('has_preventative_intervention', 'preventative_for_condition'),
    ('has_provider', 'provider'),
    ('has_publisher', 'publisher'),
    ('has_sequence_location', 'sequence_location_of'),
    ('has_sequence_variant', 'is_sequence_variant_of'),
    ('has_side_effect', 'is_side_effect_of'),
    ('has_splice_site_variant', 'is_splice_site_variant_of'),
    ('has_substrate', 'is_substrate_of'),
    ('has_synonymous_variant', 'is_synonymous_variant_of'),
    ('has_target', 'target_for'),
    ('has_variant_part', 'variant_part_of'),
    ('in_clinical_trials_for', 'tested_by_clinical_trials_of'),
    ('in_preclinical_trials_for', 'tested_by_preclinical_trials_of'),
    ('in_taxon', 'taxon_of'),
    ('increases_sensitivity_to', 'sensitivity_increased_by'),
    ('lacks_part', 'missing_from'),
    ('located_in', 'location_of'),
    ('mentioned_by', 'mentions'),
    ('model_of', 'models'),
    ('preceded_by', 'precedes'),
    ('produced_by', 'produces'),
    ('regulated_by', 'regulates'),
    ('studied_to_treat', 'treated_in_studies_by'),
    ('subclass_of', 'superclass_of'),
    ('subject_of_treatment_application_or_study_for_treatment_by', 'treats_or_applied_or_studied_to_treat'),
    ('transcribed_from', 'transcribed_to'),
    ('translates_to', 'translation_of'),
    ('treated_by', 'treats'),
    ('was_tested_for_effect_of', 'was_tested_for_effect_on'),
]
BIOLINK_INVERSE = {**{a: b for a, b in _BIOLINK_INVERSE_PAIRS}, **{b: a for a, b in _BIOLINK_INVERSE_PAIRS}}
BIOLINK_SYMMETRIC = {
    'associated_with', 'binds', 'chemically_similar_to', 'close_match', 'coexists_with',
    'coexpressed_with', 'colocalizes_with', 'correlated_with',
    'directly_physically_interacts_with', 'exact_match', 'gene_fusion_with',
    'genetic_association', 'genetic_neighborhood_of', 'genetically_associated_with',
    'genetically_interacts_with', 'homologous_to', 'in_cell_population_with',
    'in_complex_with', 'in_linkage_disequilibrium_with', 'in_pathway_with',
    'indirectly_physically_interacts_with', 'interacts_with',
    'negatively_correlated_with', 'occurs_together_in_literature_with', 'opposite_of',
    'orthologous_to', 'overlaps', 'paralogous_to', 'pharmacologically_interacts_with',
    'physically_interacts_with', 'positively_correlated_with', 'related_condition',
    'related_to', 'related_to_at_concept_level', 'related_to_at_instance_level',
    'same_as', 'similar_to', 'temporally_related_to', 'xenologous_to',
}


# --------------------------------------------------------------------------- #
# TCT one-hop expansion of the context (old and new TCT APIs, cached)
# --------------------------------------------------------------------------- #
import requests  # noqa: E402


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


def query_neighbourhoods(curies, categories, cache_path=None):
    """
    One TCT one-hop query per CURIE (cached per CURIE + category list).
    Returns {curie: {"nodes": {...}, "edges": [{"subject","object","predicate"}, ...]}}.
    """
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    find, out, n_new = None, {}, 0
    for curie in curies:
        key = f"{curie}|{','.join(sorted(categories))}"
        if key not in cache:
            find = find or make_finder()
            try:
                kg = find(curie, categories)
                cache[key] = {"nodes": kg.get("nodes", {}),
                              "edges": [{"subject": e["subject"], "object": e["object"],
                                         "predicate": e.get("predicate"),
                                         "attributes": e.get("attributes") or [],     # TRAPI attributes as returned
                                         "sources": e.get("sources") or [],           # by the knowledge providers
                                         "qualifiers": e.get("qualifiers") or []}
                                        for e in kg.get("edges", {}).values()]}
                n_new += 1
            except Exception as exc:
                print(f"  TCT failed for {curie}: {type(exc).__name__}: {exc}")
                continue
        out[curie] = cache[key]
    if cache_path and n_new:
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    print(f"  {len(out)} neighbourhoods ({n_new} newly queried, {len(out) - n_new} from cache)")
    return out


def expand_both_sides(context_curies, answer_curies, ui_node_ids, categories, alias=None,
                      max_per_seed=50, cache_path=None, rank_by=None):
    """
    Two-sided TCT expansion with a BRIDGE filter.

    * Every context seed and every answer is queried one hop (categories = Gene/Protein...).
    * A neighbour X of a seed is kept only if it BRIDGES, i.e. X is
        - a node of the UI graph (other than that seed), or
        - a neighbour of at least one answer (seed -> X -> answer), or
        - a neighbour of at least one other seed (keeps the context internally connected).
      Dead ends (X touches only the seed it came from) are dropped.
    * Edges answer -> X are added only for kept X (they complete the bridge).
    * max_per_seed applies AFTER the filter. Bridges are ranked by `rank_by(node_id)`
      if given (e.g. an enrichment score, so hubs are not favoured), else by the
      number of supporting edges.

    Returns (edges, nodes, report, edge_details) with edges as (subject, object, predicate),
    report = {"bridges_to_ui", ...} and edge_details = {(subject, object): [full TRAPI edge dicts]}
    (attributes + sources from the knowledge providers, for evidence in the TRAPI export).
    """
    alias = alias or {}
    a = lambda x: alias.get(x, x)
    context_curies = list(dict.fromkeys(context_curies))
    answer_curies = [x for x in dict.fromkeys(answer_curies) if x]
    ui = {a(x) for x in ui_node_ids}
    ctx = {a(x) for x in context_curies}

    print("  querying context seeds...")
    seed_nb = query_neighbourhoods(context_curies, categories, cache_path)
    print("  querying answers...")
    ans_nb = query_neighbourhoods(answer_curies, categories, cache_path)

    def neighbours(curie, kg):
        nb = defaultdict(list)
        for e in kg["edges"]:
            other = e["object"] if e["subject"] == curie else e["subject"]
            nb[a(other)].append(e)
        nb.pop(a(curie), None)
        return nb

    ans_of = defaultdict(set)                       # X -> answers it touches
    ans_edges = defaultdict(list)                   # X -> edges answer <-> X
    for ans, kg in ans_nb.items():
        for x, es in neighbours(ans, kg).items():
            ans_of[x].add(a(ans))
            ans_edges[x] += es
    seed_of = defaultdict(set)                      # X -> seeds it touches
    seed_edges = {s: neighbours(s, kg) for s, kg in seed_nb.items()}
    for s, nb in seed_edges.items():
        for x in nb:
            seed_of[x].add(a(s))

    edges, nodes = [], {}
    edge_details = defaultdict(list)
    report = Counter()
    for s, nb in seed_edges.items():
        kept = []
        for x, es in nb.items():
            to_ui = x in ui and x not in ctx
            to_ans = bool(ans_of.get(x))
            seed_seed = len(seed_of[x]) > 1        # links two context seeds (kept: keeps the context connected)
            if to_ui or to_ans or seed_seed:
                key = rank_by(x) if rank_by else len(es) + len(ans_edges.get(x, ()))
                kept.append((key, x, es, to_ui, to_ans))
            else:
                report["dropped_dead_ends"] += 1
        kept.sort(key=lambda t: -t[0])
        for _, x, es, to_ui, to_ans in kept[:max_per_seed]:
            report["bridges_to_ui"] += to_ui
            report["bridges_to_answers"] += to_ans
            report["bridges_seed_seed"] += (not to_ui and not to_ans)
            for e in list(es) + list(ans_edges.get(x, ())):
                edges.append((e["subject"], e["object"], e["predicate"]))
                edge_details[(e["subject"], e["object"])].append(e)
            nodes[x] = seed_nb[s]["nodes"].get(x) or next(
                (ans_nb[an]["nodes"].get(x) for an in ans_nb if x in ans_nb[an]["nodes"]), {})
        print(f"  {s}: {len(nb)} neighbours, {len(kept)} bridge, kept {min(len(kept), max_per_seed)}")
    edges = list(dict.fromkeys(edges))
    return edges, nodes, dict(report), dict(edge_details)


def expand_context(curies, categories, max_neighbours=50, cache_path=None):
    """One-sided expansion (kept for compatibility): seeds' top neighbours, no bridge filter."""
    nb = query_neighbourhoods(curies, categories, cache_path)
    edges, nodes = [], {}
    for curie, kg in nb.items():
        support = Counter(e["object"] if e["subject"] == curie else e["subject"] for e in kg["edges"])
        keep = {n for n, _ in support.most_common(max_neighbours)}
        edges += [(e["subject"], e["object"], e["predicate"]) for e in kg["edges"]
                  if (e["object"] if e["subject"] == curie else e["subject"]) in keep]
        nodes.update({n: kg["nodes"].get(n, {}) for n in keep})
    return edges, nodes
