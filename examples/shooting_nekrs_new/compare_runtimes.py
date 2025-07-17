#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from log_parser import get_runtime
import json
import scienceplots

# Use scienceplots for better aesthetics
plt.style.use(['science', 'no-latex'])

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
})

def get_backend_runtime(node_dir, backend):
    """Get runtime for a specific backend in a node directory."""
    base_path = os.path.join(node_dir, f"logs_{backend}")
    
    if not os.path.exists(base_path):
        return None
    
    # Look for train_ai_0.log file specifically
    train_ai_log = os.path.join(base_path, "train_ai_0.log")
    if not os.path.exists(train_ai_log):
        return None
    
    try:
        runtime = get_runtime(train_ai_log)
        return runtime
    except Exception as e:
        print(f"Error parsing {train_ai_log}: {e}")
        return None

def collect_runtime_data():
    """Collect runtime data for all backends and node counts."""
    backends = ['redis', 'filesystem', 'nodelocal', 'dragon']
    data = {backend: {'nodes': [], 'runtimes': []} for backend in backends}

    
    node_dirs = ["2nodes", "8nodes", "32nodes", "128nodes", "512nodes", "2048nodes"]
    for node_dir in node_dirs:
        node_count = int(node_dir.replace('nodes', ''))
        print(f"\nProcessing {node_dir} ({node_count} nodes):")
        
        for backend in backends:
            runtime = get_backend_runtime(node_dir, backend)
            if runtime is not None:
                data[backend]['nodes'].append(node_count)
                data[backend]['runtimes'].append(runtime)
                print(f"  {backend}: {runtime:.2f}s")
            else:
                print(f"  {backend}: No data")
    
    return data

def plot_runtime_comparison(data):
    """Create plot comparing runtimes across backends."""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors and markers for each backend
    colors = {
        'redis': 'red',
        'filesystem': 'blue', 
        'nodelocal': 'green',
        'dragon': 'purple'
    }
    
    markers = {
        'redis': 'o',
        'filesystem': 'o',
        'nodelocal': 'o',
        'dragon': 'o'
    }
    
    # Plot runtime comparison
    for backend in ['redis', 'filesystem', 'nodelocal', 'dragon']:
        if data[backend]['nodes']:
            ax.plot(data[backend]['nodes'], data[backend]['runtimes'], 
                   color=colors[backend], marker=markers[backend], 
                   linewidth=2, markersize=8, label=backend.capitalize())
    
    ax.set_xlabel('Number of Nodes')
    ax.set_ylim([0, 7])
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison Across Backends')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('runtime_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the runtime comparison analysis."""
    print("Collecting runtime data...")
    data = collect_runtime_data()
    
    # Filter out backends with no data
    data = {backend: values for backend, values in data.items() if values['nodes']}
    
    if not data:
        print("No runtime data found!")
        return
    
    print(f"\nFound data for backends: {list(data.keys())}")
    
    # Create plot
    print("\nCreating runtime comparison plot...")
    plot_runtime_comparison(data)
    
    print("Plot saved as 'runtime_comparison.png'")

if __name__ == "__main__":
    main()