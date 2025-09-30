import json
import os
from typing import Dict, List

def read_json_data(folder_path: str) -> Dict:
    json_path = os.path.join(folder_path, 'performance_report.json')
    with open(json_path, 'r') as f:
        return json.load(f)

def read_tex_data(folder_path: str) -> Dict:
    tex_path = os.path.join(folder_path, 'report.tex')
    with open(tex_path, 'r') as f:
        content = f.read()
    
    reliability_data = {
        'edge_connectivity': float(content.split('Edge connectivity: ')[1].split(' ')[0]),
        'node_connectivity': float(content.split('Node connectivity: ')[1].split(' ')[0]),
        'algebraic_connectivity': float(content.split('Algebraic connectivity: ')[1].split('\n')[0])
    }
    return reliability_data

def generate_comparative_report(base_path: str, node_counts: List[int]) -> str:
    data = {}
    
    for nodes in node_counts:
        folder = f'results_for_{nodes}_nodes'
        folder_path = os.path.join(base_path, folder)
        
        json_data = read_json_data(folder_path)
        tex_data = read_tex_data(folder_path)
        
        data[nodes] = {
            **json_data['network_stats'],
            **json_data['model_performance']['final_metrics'],
            **tex_data
        }
    
    latex_content = r"""
    \documentclass{article}
    \usepackage{booktabs}
    \usepackage{siunitx}
    \usepackage{caption}
    \usepackage{graphicx}
    \usepackage{float}

    \title{Comparative QKD Network Analysis}
    \date{\today}

    \begin{document}
    \maketitle

    \section{Network Characteristics Comparison}
    \begin{table}[H]
    \centering
    \caption{Network Characteristics Across Different Node Counts}
    \begin{tabular}{lrrrrr}
    \toprule
    Metric & \multicolumn{5}{c}{Number of Nodes} \\
    \cmidrule(lr){2-6}
    & 10 & 20 & 50 & 100 & 250 \\
    \midrule
    """
    
    metrics = [
        ('Number of Edges', 'num_edges'),
        ('Average Degree', 'avg_degree'),
        ('Average Key Rate (bits/s)', 'avg_key_rate'),
        ('Average QBER', 'avg_qber'),
        ('Maximum Distance (km)', 'max_distance'),
        ('Edge Connectivity', 'edge_connectivity'),
        ('Node Connectivity', 'node_connectivity'),
        ('Algebraic Connectivity', 'algebraic_connectivity')
    ]
    
    for metric_name, metric_key in metrics:
        latex_content += f"{metric_name} & "
        latex_content += " & ".join(f"{data[n][metric_key]:.3f}" for n in node_counts)
        latex_content += r" \\" + "\n"
    
    latex_content += r"""\bottomrule
    \end{tabular}
    \end{table}

    \section{Link Prediction Performance Comparison}
    \begin{table}[H]
    \centering
    \caption{Link Prediction Performance Metrics Across Different Node Counts}
    \begin{tabular}{lrrrrr}
    \toprule
    Metric & \multicolumn{5}{c}{Number of Nodes} \\
    \cmidrule(lr){2-6}
    & 10 & 20 & 50 & 100 & 250 \\
    \midrule
    """
    
    metrics = [
        ('AUC (mean)', 'auc_mean'),
        ('AUC (std)', 'auc_std'),
        ('AP (mean)', 'ap_mean'),
        ('AP (std)', 'ap_std')
    ]
    
    for metric_name, metric_key in metrics:
        latex_content += f"{metric_name} & "
        latex_content += " & ".join(f"{data[n][metric_key]:.3f}" for n in node_counts)
        latex_content += r" \\" + "\n"
    
    latex_content += r"""\bottomrule
    \end{tabular}
    \end{table}

    \end{document}
    """
    
    return latex_content

def main():
    base_path = "."
    node_counts = [10, 20, 50, 100, 250]
    latex_content = generate_comparative_report(base_path, node_counts)
    
    with open('comparative_analysis.tex', 'w') as f:
        f.write(latex_content)

if __name__ == "__main__":
    main()