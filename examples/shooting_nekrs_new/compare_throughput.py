#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from log_parser import compute_io_throughput
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

def get_backend_throughput(node_dir, backend, num_elements, dtype, operation='read'):
    """Get throughput for a specific backend in a node directory."""
    base_path = os.path.join(node_dir, f"logs_{backend}")
    
    if not os.path.exists(base_path):
        return None
    
    # Choose log file based on operation
    if operation == 'read':
        log_file = os.path.join(base_path, "train_ai_0.log")
    elif operation == 'write':
        log_file = os.path.join(base_path, "sim_0.log")
    else:
        return None
    
    if not os.path.exists(log_file):
        return None
    
    throughput_data = compute_io_throughput(log_file, num_elements, dtype)
    # Extract node count from directory name
    node_count = int(node_dir.replace('nodes', ''))
    try:
        # Total throughput = nnodes * 6 * single_rank_throughput
        total_throughput = node_count * 6 * throughput_data['mean_throughput_MB_per_sec']
        return total_throughput
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return None

def collect_throughput_data(num_elements, dtype):
    """Collect throughput data for all backends and node counts."""
    backends = ['redis', 'filesystem', 'nodelocal', 'dragon']
    data = {
        'read': {backend: {'nodes': [], 'throughput': []} for backend in backends},
        'write': {backend: {'nodes': [], 'throughput': []} for backend in backends}
    }
    
    node_dirs = ["2nodes", "8nodes", "32nodes", "128nodes", "512nodes", "2048nodes"]
    for node_dir in node_dirs:
        node_count = int(node_dir.replace('nodes', ''))
        print(f"\nProcessing {node_dir} ({node_count} nodes):")
        
        for backend in backends:
            # Get read throughput (from train_ai_0.log)
            read_throughput = get_backend_throughput(node_dir, backend, num_elements, dtype, 'read')
            if read_throughput is not None:
                data['read'][backend]['nodes'].append(node_count)
                data['read'][backend]['throughput'].append(read_throughput/1024)  # Convert to GB/s
                print(f"  {backend} read: {read_throughput:.2f} MB/s")
            else:
                print(f"  {backend} read: No data")
            
            # Get write throughput (from sim_0.log)
            write_throughput = get_backend_throughput(node_dir, backend, num_elements, dtype, 'write')
            if write_throughput is not None:
                data['write'][backend]['nodes'].append(node_count)
                data['write'][backend]['throughput'].append(write_throughput/1024)  # Convert to GB/s
                print(f"  {backend} write: {write_throughput:.2f} MB/s")
            else:
                print(f"  {backend} write: No data")
    
    return data

def plot_throughput_comparison(data):
    """Create plots comparing read and write throughput across backends."""
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
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
    
    # Plot 1: Read throughput
    for backend in ['redis', 'filesystem', 'nodelocal', 'dragon']:
        if data['read'][backend]['nodes']:
            ax1.loglog(data['read'][backend]['nodes'], data['read'][backend]['throughput'], 
                    color=colors[backend], marker=markers[backend], 
                    linewidth=2, markersize=8, label=backend.capitalize())
    
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Read Throughput (GB/s)')
    ax1.set_title('Read Throughput Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Write throughput
    for backend in ['redis', 'filesystem', 'nodelocal', 'dragon']:  # 'dragon' commented out like in runtime script
        if data['write'][backend]['nodes']:
            ax2.loglog(data['write'][backend]['nodes'], data['write'][backend]['throughput'], 
                    color=colors[backend], marker=markers[backend], 
                    linewidth=2, markersize=8, label=backend.capitalize())
    
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Write Throughput (GB/s)')
    ax2.set_title('Write Throughput Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the throughput comparison analysis."""
    # Default parameters - you can modify these as needed
    num_elements = 319488  # Adjust based on your actual data
    dtype = 'float32'       # Adjust based on your actual data type
    
    print(f"Collecting throughput data for {num_elements} elements of type {dtype}...")
    data = collect_throughput_data(num_elements, dtype)
    
    # Filter out backends with no data
    filtered_data = {
        'read': {backend: values for backend, values in data['read'].items() if values['nodes']},
        'write': {backend: values for backend, values in data['write'].items() if values['nodes']}
    }
    
    if not filtered_data['read'] and not filtered_data['write']:
        print("No throughput data found!")
        return
    
    print(f"\nFound read data for backends: {list(filtered_data['read'].keys())}")
    print(f"Found write data for backends: {list(filtered_data['write'].keys())}")
    
    # Create plot
    print("\nCreating throughput comparison plots...")
    plot_throughput_comparison(filtered_data)
    
    print("Plot saved as 'throughput_comparison.png'")

if __name__ == "__main__":
    main()
