import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from components.attacks import apply_mitm, apply_dos, apply_passive_eavesdrop
import numpy as np
import networkx as nx
from components.channel_simulator import AdvancedQuantumChannelSimulator  # Import the simulator

class AdvancedQKDNetwork:
    def __init__(self, config):
        self.num_nodes = config['num_nodes']
        self.distance_threshold = config['distance_threshold']
        self.config = config
        self.positions = self._generate_realistic_topology()

    def _generate_realistic_topology(self):
        centers = np.random.multivariate_normal(
            mean=[0, 0],
            cov=[[self.config['simulation']['center_covariance'], 0], [0, self.config['simulation']['center_covariance']]],
            size=3
        )

        positions = []
        for _ in range(self.num_nodes):
            center = centers[np.random.randint(0, 3)]
            pos = center + np.random.multivariate_normal(
                mean=[0, 0],
                cov=[[self.config['simulation']['node_covariance'], 0], [0, self.config['simulation']['node_covariance']]]
            )
            positions.append(pos)

        return np.array(positions)

    def generate_graph_data(self):
        distances = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                distances[i, j] = distances[j, i] = np.linalg.norm(
                    self.positions[i] - self.positions[j]
                )

        edges = []
        edge_attrs = []

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if distances[i, j] < self.distance_threshold:
                    simulator = AdvancedQuantumChannelSimulator(
                        distance=distances[i, j],
                        wavelength=self.config['wavelength'],
                        fiber_loss=self.config['fiber_loss'],
                        detector_efficiency=self.config['detector_efficiency'],
                        dark_count_rate=self.config['dark_count_rate'],
                        atmospheric_visibility=self.config['atmospheric_visibility'] if np.random.random() < self.config['atmospheric_visibility_probability'] else None,
                        mean_photon_number = self.config['mean_photon_number'],
                        num_pulses = self.config['num_pulses']
                    )
                    results = simulator.simulate_bb84_protocol()
                    
                    attack_cfg = self.config.get('attack', {"enabled": False})
                    if attack_cfg.get("enabled", False):
                        # Decide if this edge should be attacked
                        params = attack_cfg.get("params", {})
                        edge_is_target = False

                        # 1) node-based targeting (if provided)
                        if attack_cfg.get("target_nodes"):
                            if (i in attack_cfg["target_nodes"]) or (j in attack_cfg["target_nodes"]):
                                edge_is_target = True
                        # 2) or random fraction of edges
                        if not edge_is_target and np.random.rand() < float(attack_cfg.get("edge_fraction", 0.0)):
                            edge_is_target = True

                        if edge_is_target:
                            t = attack_cfg.get("type", "").lower()
                            if t == "mitm":
                                results = apply_mitm(results, strength=float(params.get("strength", 1.0)))
                            elif t == "dos":
                                mode = params.get("mode", "block")
                                results = apply_dos(results, mode=mode, extra_loss_db=float(params.get("extra_loss_db", 60.0)))
                            elif t == "passive":
                                results = apply_passive_eavesdrop(
                                    results,
                                    leak_fraction=float(params.get("leak_fraction", 0.3)),
                                    qber_bump=float(params.get("qber_bump", 0.0))
                                )

                    if results['final_key_rate'] > 0:
                        edges.append([i, j])
                        edge_attrs.append([
                            results['final_key_rate'],
                            results['qber'],
                            distances[i, j],
                            results['channel_loss_db'],
                            results['dark_count_probability']
                        ])

        edge_index = torch.tensor(edges).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        G = nx.Graph()
        G.add_edges_from(edges)

        node_features = []
        for i in range(self.num_nodes):
            features = [
                self.positions[i, 0],
                self.positions[i, 1],
                G.degree(i) if i in G else 0,
                nx.betweenness_centrality(G).get(i, 0) if i in G else 0
            ]
            node_features.append(features)

        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=to_undirected(edge_index),
            edge_attr=edge_attr,
            pos=torch.tensor(self.positions, dtype=torch.float)
        )