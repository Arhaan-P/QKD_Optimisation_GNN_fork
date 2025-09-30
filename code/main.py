import torch
import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import negative_sampling
from components.network_generator import AdvancedQKDNetwork
from components.link_predictor import AdvancedQKDLinkPredictor
from components.metrics import bce_with_logits_loss
from components.visualization import visualize_results
from components.report_generator import generate_latex_report
from components.network_analyzer import analyze_network_reliability
from components.network_comparison import QKDNetworkComparison
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def train_and_evaluate(model, data, config):
    num_epochs = config['training']['num_epochs']
    k_folds = config['training']['k_folds']

    # device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = model.to(device)
    data = data.to(device)

    print(f"Using device: {device}")

    all_results = []
    kf = KFold(n_splits=k_folds, shuffle=True)

    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    unique_edges = set()
    edge_to_idx = {}

    for i in range(edge_index.shape[1]):
        edge = tuple(sorted([edge_index[0, i], edge_index[1, i]]))
        if edge not in unique_edges:
            unique_edges.add(edge)
            edge_to_idx[edge] = len(edge_to_idx)

    unique_edges = list(unique_edges)

    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_edges)):
        print(f"\nFold {fold + 1}/{k_folds}")

        train_edges = [unique_edges[i] for i in train_idx]
        val_edges = [unique_edges[i] for i in val_idx]

        train_edge_index = np.array([[edge[0], edge[1]] for edge in train_edges]).T
        train_edge_attr = np.array([edge_attr[edge_to_idx[edge]] for edge in train_edges])
        val_edge_index = np.array([[edge[0], edge[1]] for edge in val_edges]).T
        val_edge_attr = np.array([edge_attr[edge_to_idx[edge]] for edge in val_edges])

        train_edge_index = torch.from_numpy(train_edge_index).to(device)
        train_edge_attr = torch.from_numpy(train_edge_attr).float().to(device)
        val_edge_index = torch.from_numpy(val_edge_index).to(device)
        val_edge_attr = torch.from_numpy(val_edge_attr).float().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        best_val_loss = float('inf')
        early_stopping_counter = 0
        train_losses = []
        val_metrics = {'auc': [], 'ap': [], 'loss': []}

        # placeholders for confusion matrix
        cm_true, cm_preds = None, None

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            z, edge_features = model(data.x, train_edge_index, train_edge_attr)
            pos_out = model.decode(z, edge_features, train_edge_index)

            neg_edge_index = negative_sampling(
                train_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=train_edge_index.size(1)
            )

            neg_out = model.decode(z, edge_features, neg_edge_index)
            loss = bce_with_logits_loss(pos_out, neg_out)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                z, edge_features = model(data.x, val_edge_index, val_edge_attr)
                pos_out = model.decode(z, edge_features, val_edge_index)

                neg_edge_index = negative_sampling(
                    val_edge_index,
                    num_nodes=data.num_nodes,
                    num_neg_samples=val_edge_index.size(1)
                )

                neg_out = model.decode(z, edge_features, neg_edge_index)
                val_loss = bce_with_logits_loss(pos_out, neg_out)

                pred = torch.cat([pos_out, neg_out]).cpu().numpy()
                true = torch.cat([
                    torch.ones(pos_out.size(0)),
                    torch.zeros(neg_out.size(0))
                ]).numpy()

                # Convert probabilities to binary (threshold = 0.5)
                binary_preds = (pred >= 0.5).astype(int)

                # Save first fold's preds/labels for confusion matrix later
                if fold == 0:
                    cm_true, cm_preds = true, binary_preds

                auc = roc_auc_score(true, pred)
                ap = average_precision_score(true, pred)

                val_metrics['auc'].append(auc)
                val_metrics['ap'].append(ap)
                val_metrics['loss'].append(val_loss.item())

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: Train Loss = {loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}, AUC = {auc:.4f}, AP = {ap:.4f}")

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= config['training']['early_stopping_patience']:
                    print("Early stopping triggered")
                    break

        # generate confusion matrix once after training (fold 0 only)
        if fold == 0 and cm_true is not None and cm_preds is not None:
            cm = confusion_matrix(cm_true, cm_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Link", "Link"])
            disp.plot(cmap=plt.cm.Blues, values_format="d")
            plt.title(f"Confusion Matrix - {config['attack']['type'].upper()} Attack")
            plt.savefig(f"{config['report']['save_path']}/confusion_matrix_{config['attack']['type']}.png")
            plt.close()

        fold_results = {
            'fold': fold + 1,
            'train_losses': train_losses,
            'val_metrics': val_metrics,
            'final_auc': auc,
            'final_ap': ap
        }
        all_results.append(fold_results)

    return all_results


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    torch.manual_seed(42)
    np.random.seed(42)

    print("Generating QKD network...")
    network = AdvancedQKDNetwork(config)
    network_data = network.generate_graph_data()

    print("Creating and training model...")
    model = AdvancedQKDLinkPredictor(
        in_channels=network_data.x.size(1),
        edge_attr_channels=network_data.edge_attr.size(1),
        hidden_channels=config['link_predictor']['hidden_channels'],
        dropout_rate=config['link_predictor']['dropout_rate']
    )

    results = train_and_evaluate(model, network_data, config)

    print("Generating visualizations and analysis...")
    save_path = config['report']['save_path']
    performance_report = visualize_results(results, network_data, save_path)

    print("Analyzing network reliability...")
    network_metrics = analyze_network_reliability(network_data)

    print("Generating LaTeX report...")
    generate_latex_report(performance_report, network_metrics, save_path)

    print(f"\nAnalysis complete. Results saved in: {save_path}")

    print("\nSummary Statistics:")
    print("===========================")
    print(f"Number of nodes: {network_data.num_nodes}")
    print(f"Number of edges: {network_data.edge_index.size(1) // 2}")
    print(f"Average degree: {network_data.edge_index.size(1) / network_data.num_nodes:.2f}")
    print(f"Average key rate: {performance_report['network_stats']['avg_key_rate']:.2e} bits/s")
    print(f"Average QBER: {performance_report['network_stats']['avg_qber']:.3f}")
    print(f"Model AUC: {performance_report['model_performance']['final_metrics']['auc_mean']:.3f} ± "
          f"{performance_report['model_performance']['final_metrics']['auc_std']:.3f}")
    
    comparison = QKDNetworkComparison(config)

    print("Analyzing baseline network...")
    baseline_metrics = comparison.generate_baseline_network()

    print("Generating GNN optimized network...")
    gnn_metrics = comparison.generate_gnn_optimized_network(model)

    print("Generating comparison report...")
    report = comparison.generate_comparison_report()

    print("\nNetwork Comparison Summary:")
    print("===========================")
    print(f"\nBaseline Network:")
    print(f"Total Key Rate: {baseline_metrics['performance_metrics']['total_key_rate']:.2e} bits/s")
    print(f"Average QBER: {baseline_metrics['performance_metrics']['avg_qber']:.3f}")
    print(f"Average Distance: {baseline_metrics['performance_metrics']['avg_distance']:.2f} km")

    print(f"\nGNN-Optimized Network:")
    print(f"Total Key Rate: {gnn_metrics['performance_metrics']['total_key_rate']:.2e} bits/s")
    print(f"Average QBER: {gnn_metrics['performance_metrics']['avg_qber']:.3f}")
    print(f"Average Distance: {gnn_metrics['performance_metrics']['avg_distance']:.2f} km")

    print(f"\nImprovements:")
    for metric, value in report['improvements']['performance_metrics'].items():
        print(f"{metric}: {value:.1f}%")
