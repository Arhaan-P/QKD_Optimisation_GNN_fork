import networkx as nx

def analyze_network_reliability(network_data):
    G = nx.Graph()
    edge_index = network_data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])

    metrics = {
        'avg_clustering': nx.average_clustering(G),
        'avg_shortest_path': nx.average_shortest_path_length(G),
        'algebraic_connectivity': nx.algebraic_connectivity(G),
        'edge_connectivity': nx.edge_connectivity(G),
        'node_connectivity': nx.node_connectivity(G)
    }

    def simulate_attacks(G, n_removals):
        G_copy = G.copy()
        connectivities = []
        for _ in range(n_removals):
            if len(G_copy.nodes()) > 1:
                node_to_remove = max(G_copy.degree(), key=lambda x: x[1])[0]
                G_copy.remove_node(node_to_remove)
                if nx.is_connected(G_copy):
                    connectivities.append(nx.edge_connectivity(G_copy))
                else:
                    connectivities.append(0)
        return connectivities

    metrics['attack_resilience'] = simulate_attacks(G, min(10, G.number_of_nodes() - 1))

    return metrics