# Step 1: Remove redudant triplets that are duplicates or self-loop

import ast
import re
import pandas as pd

df_kg = pd.read_csv("kg_results_sample.csv")
df_kg.head()

pattern = re.compile(r"\[\d+\]")


def clean_triplets(triplets):
    try:
        # Convert string to list of tuples
        triplets = ast.literal_eval(triplets)
    except Exception as e:
        print(f"Error parsing string: {e}")
        return []
    seen = set()
    cleaned = []
    for triplet in triplets:
        if len(triplet) != 3:
            continue  # ignore malformed triplets
        subj, pred, obj = triplet

        # Skip duplicates
        if triplet in seen:
            continue
        seen.add(triplet)

        # Skip self-loops
        if subj == obj:
            continue

        # Skip if any part contains [number]
        if pattern.search(subj) or pattern.search(pred) or pattern.search(obj):
            continue

        cleaned.append(triplet)
    return cleaned


df_kg["article_graph_cleaned"] = df_kg["article_graph"].apply(clean_triplets)


# Step 2: Gather all triplets in a list

triplets_collection = []
for index, row in df_kg.iterrows():
    triplets = row["article_graph_cleaned"]
    for triplet in triplets:
        triplets_collection.append(triplet)

# Step 3: Create a knowledge graph using networkx library

import networkx as nx


def build_kg(triplets_collection):
    G = nx.MultiDiGraph()
    for subj, pred, obj in triplets_collection:
        # print(f"Adding triplet: {subj} - {pred} -> {obj}")
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, relation=pred)

    return G


kg_graph = build_kg(triplets_collection)


# Step 4: Visualize results

import matplotlib.pyplot as plt


def visualize_existing_graph(G, figsize=(14, 10)):
    pos = nx.spring_layout(G, k=0.7, iterations=100)

    plt.figure(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000)

    # Draw edges with arrows
    nx.draw_networkx_edges(
        G, pos, arrowstyle="->", arrowsize=20, connectionstyle="arc3,rad=0.1"
    )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # Draw edge labels (predicates)
    edge_labels = {
        (u, v, k): d["relation"]
        for u, v, k, d in G.edges(keys=True, data=True)
        if "relation" in d
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="gray", font_size=9
    )

    plt.title("Knowledge Graph Visualization", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        "../example_outputs/knowledge_graph_sample.png", dpi=300, bbox_inches="tight"
    )


# visualize_existing_graph(kg_graph)


def visualize_subgraph(G, center_node=None, depth=1, max_nodes=20):
    if center_node and center_node in G:
        nodes = set([center_node])
        for _ in range(depth):
            neighbors = set()
            for node in nodes:
                neighbors.update(G.successors(node))
                neighbors.update(G.predecessors(node))
            nodes.update(neighbors)
        subgraph = G.subgraph(nodes)
    else:
        # Show a random subset of nodes (up to max_nodes)
        nodes = list(G.nodes())[:max_nodes]
        subgraph = G.subgraph(nodes)

    visualize_existing_graph(subgraph)


# Focus around a key concept, like 'Pledge'
visualize_subgraph(kg_graph, center_node="Pledge", depth=2)


# Step 5: Store the graph in a neo4j database
