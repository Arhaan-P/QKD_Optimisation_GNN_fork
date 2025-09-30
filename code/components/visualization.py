import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

def to_networkx(data):
    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])
    return G

def visualize_results(results, network_data, save_path='qkd_results'):
    os.makedirs(save_path, exist_ok=True)

    min_epochs = min(len(result['train_losses']) for result in results)

    truncated_results = []
    for result in results:
        truncated_result = {
            'train_losses': result['train_losses'][:min_epochs],
            'val_metrics': {
                'auc': result['val_metrics']['auc'][:min_epochs],
                'ap': result['val_metrics']['ap'][:min_epochs],
                'loss': result['val_metrics']['loss'][:min_epochs]
            },
            'final_auc': result['final_auc'],
            'final_ap': result['final_ap']
        }
        truncated_results.append(truncated_result)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    for result in truncated_results:
        plt.plot(result['train_losses'], alpha=0.3)
    mean_train_loss = np.mean([r['train_losses'] for r in truncated_results], axis=0)
    plt.plot(mean_train_loss, 'r-', label='Mean')
    plt.title('Training Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    for result in truncated_results:
        plt.plot(result['val_metrics']['auc'], alpha=0.3)
    mean_val_auc = np.mean([r['val_metrics']['auc'] for r in truncated_results], axis=0)
    plt.plot(mean_val_auc, 'r-', label='Mean')
    plt.title('Validation AUC Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    distances = network_data.edge_attr[:, 2].cpu().numpy()
    key_rates = network_data.edge_attr[:, 0].cpu().numpy()

    plt.subplot(2, 2, 3)
    plt.scatter(distances, key_rates, alpha=0.5)
    plt.xlabel('Distance (km)')
    plt.ylabel('Key Rate (bits/s)')
    plt.yscale('log')
    plt.title('Key Rate vs Distance')

    x_fit = np.linspace(min(distances), max(distances), 100)
    y_fit = np.exp(-0.2 * x_fit)
    plt.plot(x_fit, y_fit * max(key_rates), 'r--', label='Theoretical')
    plt.legend()

    plt.subplot(2, 2, 4)
    qber_values = network_data.edge_attr[:, 1].cpu().numpy()
    sns.histplot(qber_values, bins=20)
    plt.xlabel('QBER')
    plt.ylabel('Count')
    plt.title('QBER Distribution')

    plt.tight_layout()
    plt.savefig(f"{save_path}/training_metrics.png")
    plt.close()

    plt.figure(figsize=(12, 10))
    adj_matrix = np.zeros((network_data.num_nodes, network_data.num_nodes))
    edge_index = network_data.edge_index.cpu().numpy()
    key_rates = network_data.edge_attr[:, 0].cpu().numpy()

    edge_dict = {}
    for i in range(len(key_rates)):
        src, dst = edge_index[0, i], edge_index[1, i]
        edge_dict[(src, dst)] = i
        edge_dict[(dst, src)] = i  # Add reverse direction

    for i in range(network_data.num_nodes):
        for j in range(network_data.num_nodes):
            if (i, j) in edge_dict:
                idx = edge_dict[(i, j)]
                adj_matrix[i, j] = key_rates[idx]

    sns.heatmap(adj_matrix, cmap='YlOrRd',
                xticklabels=range(network_data.num_nodes),
                yticklabels=range(network_data.num_nodes))
    plt.title('Link Quality Heatmap (Key Rates)')
    plt.savefig(f"{save_path}/link_quality_heatmap.png")
    plt.close()

    report = {
        'network_stats': {
            'num_nodes': int(network_data.num_nodes),
            'num_edges': int(len(key_rates)),  # Use actual number of unique edges
            'avg_degree': float(2 * len(key_rates) / network_data.num_nodes),  # Account for bidirectional edges
            'avg_key_rate': float(np.mean(key_rates)),
            'avg_qber': float(np.mean(qber_values)),
            'max_distance': float(np.max(distances))
        },
        'model_performance': {
            'final_metrics': {
                'auc_mean': float(np.mean([r['final_auc'] for r in results])),
                'auc_std': float(np.std([r['final_auc'] for r in results])),
                'ap_mean': float(np.mean([r['final_ap'] for r in results])),
                'ap_std': float(np.std([r['final_ap'] for r in results]))
            },
            'convergence': {
                'final_train_loss_mean': float(np.mean([r['train_losses'][-1] for r in truncated_results])),
                'best_epoch_mean': float(np.mean([np.argmin(r['val_metrics']['loss']) for r in truncated_results]))
            }
        }
    }

    with open(f"{save_path}/performance_report.json", 'w') as f:
        json.dump(report, f, indent=4)

    return report