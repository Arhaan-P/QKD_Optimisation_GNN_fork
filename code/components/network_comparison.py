import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
from tqdm import tqdm
from components.network_generator import AdvancedQKDNetwork
from components.channel_simulator import AdvancedQuantumChannelSimulator

class QKDNetworkComparison:
    def __init__(self, config):
        self.config = config
        self.base_network = None
        self.gnn_network = None
        
    def generate_baseline_network(self):
        print("Generating baseline network...")
        network = AdvancedQKDNetwork(self.config)
        
        print("Simulating quantum channels...")
        self.base_network = self._generate_graph_data_with_progress(network)
        
        print("Calculating network metrics...")
        return self._calculate_network_metrics(self.base_network, "baseline")
    
    def _generate_graph_data_with_progress(self, network):
        positions = network.positions
        num_nodes = self.config['num_nodes']
        distance_threshold = self.config['distance_threshold']
        
        # Pre-calculate all distances
        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distances[i, j] = distances[j, i] = np.linalg.norm(
                    positions[i] - positions[j]
                )
        
        edges = []
        edge_attrs = []
        
        # Count potential connections for progress bar
        possible_edges = sum(1 for i in range(num_nodes) 
                           for j in range(i + 1, num_nodes) 
                           if distances[i, j] < distance_threshold)
        
        print(f"Processing {possible_edges} potential connections...")
        
        with tqdm(total=possible_edges) as pbar:
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if distances[i, j] < distance_threshold:
                        simulator = AdvancedQuantumChannelSimulator(
                            distance=distances[i, j],
                            wavelength=self.config['wavelength'],
                            fiber_loss=self.config['fiber_loss'],
                            detector_efficiency=self.config['detector_efficiency'],
                            dark_count_rate=self.config['dark_count_rate'],
                            atmospheric_visibility=self.config['atmospheric_visibility'] if np.random.random() < self.config['atmospheric_visibility_probability'] else None,
                            mean_photon_number=self.config['mean_photon_number'],
                            num_pulses=self.config['num_pulses']
                        )
                        
                        results = simulator.simulate_bb84_protocol()
                        
                        if results['final_key_rate'] > 0:
                            edges.append([i, j])
                            edge_attrs.append([
                                results['final_key_rate'],
                                results['qber'],
                                distances[i, j],
                                results['channel_loss_db'],
                                results['dark_count_probability']
                            ])
                        
                        pbar.update(1)
        
        if not edges:
            raise ValueError("No valid edges were generated. Try adjusting the distance threshold or other parameters.")
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # Create node features
        G = nx.Graph()
        G.add_edges_from(edges)
        
        node_features = []
        degree_dict = dict(G.degree())
        betweenness_dict = nx.betweenness_centrality(G)
        
        for i in range(num_nodes):
            features = [
                positions[i, 0],
                positions[i, 1],
                degree_dict.get(i, 0),
                betweenness_dict.get(i, 0)
            ]
            node_features.append(features)
        
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.cat([edge_index, edge_index.flip(0)], dim=1),  # Make undirected
            edge_attr=torch.cat([edge_attr, edge_attr], dim=0),  # Duplicate attributes for undirected
            pos=torch.tensor(positions, dtype=torch.float)
        )
    
    def _calculate_network_metrics(self, network_data, network_type):
        # Convert to NetworkX graph for metric calculation
        G = nx.Graph()
        edge_index = network_data.edge_index.cpu().numpy()
        edge_attr = network_data.edge_attr.cpu().numpy()
        
        # Only use unique edges (avoiding duplicates from undirected graph)
        edge_dict = {}
        for i in range(0, edge_index.shape[1], 2):
            src, dst = edge_index[0, i], edge_index[1, i]
            weight = float(edge_attr[i][0])  # Use key rate as weight
            edge_dict[(min(src, dst), max(src, dst))] = weight
            
        G.add_weighted_edges_from((src, dst, weight) for (src, dst), weight in edge_dict.items())
        
        # Calculate metrics with error handling
        metrics = {
            "network_type": network_type,
            "topology_metrics": {
                "num_nodes": network_data.num_nodes,
                "num_edges": len(edge_dict),
                "avg_degree": float(2 * len(edge_dict)) / network_data.num_nodes,
                "density": nx.density(G),
                "avg_clustering": nx.average_clustering(G),
            },
            "performance_metrics": {
                "avg_key_rate": float(np.mean(edge_attr[::2, 0])),  # Use only unique edges
                "total_key_rate": float(np.sum(edge_attr[::2, 0])),
                "avg_qber": float(np.mean(edge_attr[::2, 1])),
                "max_qber": float(np.max(edge_attr[::2, 1])),
                "avg_distance": float(np.mean(edge_attr[::2, 2])),
                "max_distance": float(np.max(edge_attr[::2, 2])),
                "avg_channel_loss": float(np.mean(edge_attr[::2, 3])),
                "dark_count_prob": float(np.mean(edge_attr[::2, 4]))
            }
        }
        
        # Add path-based metrics with safe handling
        if nx.is_connected(G):
            try:
                metrics["topology_metrics"]["avg_shortest_path"] = nx.average_shortest_path_length(G)
                metrics["topology_metrics"]["diameter"] = nx.diameter(G)
            except nx.NetworkXError:
                metrics["topology_metrics"]["avg_shortest_path"] = float('inf')
                metrics["topology_metrics"]["diameter"] = float('inf')
        else:
            metrics["topology_metrics"]["avg_shortest_path"] = float('inf')
            metrics["topology_metrics"]["diameter"] = float('inf')
            
        try:
            metrics["topology_metrics"]["algebraic_connectivity"] = nx.algebraic_connectivity(G)
        except nx.NetworkXError:
            metrics["topology_metrics"]["algebraic_connectivity"] = 0.0
            
        return metrics
        
    def generate_gnn_optimized_network(self, model):
        print("Generating GNN optimized network...")
        network = AdvancedQKDNetwork(self.config)
        initial_data = network.generate_graph_data()
        
        model.eval()
        with torch.no_grad():
            z, edge_features = model(initial_data.x, initial_data.edge_index, initial_data.edge_attr)
            
            # Generate all possible edges
            all_edges = []
            for i in range(self.config['num_nodes']):
                for j in range(i + 1, self.config['num_nodes']):
                    all_edges.append([i, j])
            
            all_edges = torch.tensor(all_edges).t()
            predictions = model.decode(z, edge_features, all_edges)
            predictions = torch.sigmoid(predictions)
            
            # Select top-k edges based on predictions
            k = initial_data.edge_index.size(1) // 2  # Number of unique edges in initial network
            top_k_indices = torch.topk(predictions, k).indices
            optimized_edges = all_edges[:, top_k_indices]
        
        # Simulate channels for selected edges
        optimized_edge_attrs = []
        for i in range(optimized_edges.size(1)):
            src, dst = optimized_edges[:, i]
            distance = torch.norm(initial_data.pos[src] - initial_data.pos[dst])
            
            if distance < self.config['distance_threshold']:
                simulator = AdvancedQuantumChannelSimulator(
                    distance=distance.item(),
                    wavelength=self.config['wavelength'],
                    fiber_loss=self.config['fiber_loss'],
                    detector_efficiency=self.config['detector_efficiency'],
                    dark_count_rate=self.config['dark_count_rate'],
                    atmospheric_visibility=self.config['atmospheric_visibility'],
                    mean_photon_number=self.config['mean_photon_number'],
                    num_pulses=self.config['num_pulses']
                )
                results = simulator.simulate_bb84_protocol()
                
                optimized_edge_attrs.append([
                    results['final_key_rate'],
                    results['qber'],
                    distance.item(),
                    results['channel_loss_db'],
                    results['dark_count_probability']
                ])
        
        optimized_edge_attrs = torch.tensor(optimized_edge_attrs, dtype=torch.float)
        
        self.gnn_network = Data(
            x=initial_data.x,
            edge_index=torch.cat([optimized_edges, optimized_edges.flip(0)], dim=1),
            edge_attr=torch.cat([optimized_edge_attrs, optimized_edge_attrs], dim=0),
            pos=initial_data.pos
        )
        
        print("Calculating network metrics...")
        return self._calculate_network_metrics(self.gnn_network, "GNN-optimized")
    
    def generate_comparison_report(self):
        if self.base_network is None or self.gnn_network is None:
            raise ValueError("Both baseline and GNN-optimized networks must be generated first")
            
        base_metrics = self._calculate_network_metrics(self.base_network, "baseline")
        gnn_metrics = self._calculate_network_metrics(self.gnn_network, "GNN-optimized")
        
        # Calculate percentage improvements
        improvements = {
            "topology_metrics": {},
            "performance_metrics": {}
        }
        
        for metric_type in ["topology_metrics", "performance_metrics"]:
            for key in base_metrics[metric_type]:
                if isinstance(base_metrics[metric_type][key], (int, float)) and base_metrics[metric_type][key] != 0:
                    improvements[metric_type][key] = (
                        (gnn_metrics[metric_type][key] - base_metrics[metric_type][key]) / 
                        base_metrics[metric_type][key] * 100
                    )
        
        return {
            "baseline": base_metrics,
            "gnn_optimized": gnn_metrics,
            "improvements": improvements
        }