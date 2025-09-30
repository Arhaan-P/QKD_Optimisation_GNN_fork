from datetime import datetime

def generate_latex_report(results, network_metrics, save_path):
    latex_content = f"""
    \\documentclass{{article}}
    \\usepackage{{graphicx}}
    \\usepackage{{booktabs}}
    \\usepackage{{float}}
    \\title{{QKD Network Analysis Report}}
    \\date{{{datetime.now().strftime('%Y-%m-%d')}}}
    \\begin{{document}}
    \\maketitle

    \\section{{Network Characteristics}}
    Number of nodes: {results['network_stats']['num_nodes']} \\\\
    Number of edges: {results['network_stats']['num_edges']} \\\\
    Average degree: {results['network_stats']['avg_degree']:.2f} \\\\
    Average clustering coefficient: {network_metrics['avg_clustering']:.3f} \\\\
    Average shortest path length: {network_metrics['avg_shortest_path']:.2f}

    \\section{{QKD Performance}}
    Average key rate: {results['network_stats']['avg_key_rate']:.2e} bits/s \\\\
    Average QBER: {results['network_stats']['avg_qber']:.3f} \\\\
    Maximum link distance: {results['network_stats']['max_distance']:.1f} km

    \\section{{Link Prediction Performance}}
    \\begin{{table}}[H]
    \\centering
    \\begin{{tabular}}{{lcc}}
    \\toprule
    Metric & Mean & Std. Dev. \\\\
    \\midrule
    AUC & {results['model_performance']['final_metrics']['auc_mean']:.3f} &
        {results['model_performance']['final_metrics']['auc_std']:.3f} \\\\
    AP & {results['model_performance']['final_metrics']['ap_mean']:.3f} &
        {results['model_performance']['final_metrics']['ap_std']:.3f} \\\\
    \\bottomrule
    \\end{{tabular}}
    \\caption{{Link Prediction Performance Metrics}}
    \\end{{table}}

    \\section{{Network Reliability}}
    Edge connectivity: {network_metrics['edge_connectivity']} \\\\
    Node connectivity: {network_metrics['node_connectivity']} \\\\
    Algebraic connectivity: {network_metrics['algebraic_connectivity']:.3f}

    \\section{{Figures}}
    \\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{network_topology.png}}
    \\caption{{QKD Network Topology}}
    \\end{{figure}}

    \\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{training_metrics.png}}
    \\caption{{Training and Validation Metrics}}
    \\end{{figure}}

    \\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{link_quality_heatmap.png}}
    \\caption{{Link Quality Heatmap}}
    \\end{{figure}}

    \\end{{document}}
    """

    with open(f"{save_path}/report.tex", 'w') as f:
        f.write(latex_content)