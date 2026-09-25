import numpy as np
import pandas as pd
import heapq
from UI_parser import TranslatorExport

def build_adjacency_matrix(export: TranslatorExport, df_context: pd.DataFrame):
    """
    Construit une matrice d'adjacence à partir de l'objet TranslatorExport et des entités de contexte.
    """
    # 1. Extraction des nœuds depuis l'objet export et le contexte
    ui_nodes = set(export.nodes.keys())
    # On ajoute aussi les nœuds mentionnés dans les edges qui pourraient manquer dans export.nodes
    for edge in export.edges.values():
        ui_nodes.add(edge.subject)
        ui_nodes.add(edge.object)
        
    context_nodes = set(df_context['canonical_name'].tolist())
    all_nodes = sorted(list(ui_nodes.union(context_nodes)))
    
    # Mapping entité -> index
    entity_mapping = {node: i for i, node in enumerate(all_nodes)}
    node_count = len(all_nodes)
    
    # Initialisation de la matrice d'adjacence (en float pour les calculs de transition)
    adj_matrix = np.zeros((node_count, node_count), dtype=float)
    
    # 2. Remplissage de la matrice à partir des edges de l'export
    for edge in export.edges.values():
        src = edge.subject
        tgt = edge.object
        if src in entity_mapping and tgt in entity_mapping:
            adj_matrix[entity_mapping[src], entity_mapping[tgt]] = 1.0
            
    return adj_matrix, entity_mapping

def compute_context_influence(adj_matrix, df_context, entity_mapping, iterations=20, alpha=0.85):
    """
    Calcule l'influence via un Random Walk avec téléportation vers le contexte.
    L'influence est distribuée sur le graphe selon la probabilité de visite.
    """
    node_count = adj_matrix.shape[0]
    
    # 1. Création de la matrice de transition stochastique
    column_sums = adj_matrix.sum(axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_matrix = adj_matrix / column_sums
    
    transition_matrix = np.nan_to_num(transition_matrix)

    # 2. Vecteur de distribution initial (v_start)
    v = np.zeros(node_count)
    context_indices = [entity_mapping[e] for e in df_context['canonical_name'] if e in entity_mapping]
    
    if not context_indices:
        return v
        
    v[context_indices] = 1.0 / len(context_indices)
    v_start = v.copy()

    # 3. Itérations du Random Walk
    for _ in range(iterations):
        v = alpha * np.dot(transition_matrix, v) + (1 - alpha) * v_start
        
    return v

def generate_influence_graph(adj_matrix, iterations=10, alpha=0.85):
    """
    Génère un graphe d'influence où les poids des arêtes représentent 
    la probabilité de propagation du signal.
    S = sum_{t=1}^k alpha^t * M^t
    """
    node_count = adj_matrix.shape[0]
    
    # 1. Matrice de transition stochastique
    column_sums = adj_matrix.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        M = adj_matrix / column_sums
    M = np.nan_to_num(M)
    
    # 2. Calcul de la matrice de diffusion cumulée S
    S = np.zeros((node_count, node_count))
    M_pow = np.eye(node_count) # M^0
    
    for t in range(1, iterations + 1):
        M_pow = np.dot(M_pow, M)
        S += (alpha**t) * M_pow
        
    return S

def filter_significant_influence(df_influence: pd.DataFrame):
    """
    Filtre le DataFrame pour ne garder que les entités ayant un score d'influence strictement positif.
    """
    return df_influence[df_influence['influence_score'] > 0]

def find_best_influence_paths(adj_matrix, influence_matrix, entity_mapping, start_nodes, target_node, top_k=1):
    """
    Trouve le(s) chemin(s) maximisant l'influence entre un ensemble de nœuds de départ 
    (contexte) et un nœud cible.
    Utilise une variante de Dijkstra pour maximiser le produit des poids.
    """
    node_count = adj_matrix.shape[0]
    
    # Initialisation des distances (scores) à -infini
    max_influence = np.full(node_count, -np.inf)
    parent = {i: None for i in range(node_count)}
    
    # File de priorité (max-heap) : (-score, node)
    pq = []
    
    for start in start_nodes:
        if start < node_count:
            max_influence[start] = 0 # log(1) = 0
            heapq.heappush(pq, (0, start))
            
    while pq:
        curr_dist, u = heapq.heappop(pq)
        curr_dist = -curr_dist
        
        if u == target_node:
            break
            
        if curr_dist < max_influence[u]:
            continue
            
        # Explorer les voisins via la matrice d'influence
        for v in range(node_count):
            weight = influence_matrix[u, v]
            if weight > 0:
                # On utilise log(weight) pour transformer le produit en somme
                new_dist = curr_dist + np.log(weight)
                if new_dist > max_influence[v]:
                    max_influence[v] = new_dist
                    parent[v] = u
                    heapq.heappush(pq, (-new_dist, v))
                    
    # Reconstruction du chemin
    path = []
    curr = target_node
    while curr is not None:
        path.append(curr)
        curr = parent[curr]
    
    path.reverse()
    
    if len(path) <= 1 and target_node not in start_nodes:
        return None, -np.inf
        
    return path, max_influence[target_node]
