import ipywidgets as widgets
from IPython.display import display, clear_output
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Modified function to include noise_type
def create_and_display_network(noise_level, num_simulations, p, noise_type):
    avg_path_lengths = []
    clustering_coeffs = []
    
    for _ in range(num_simulations):
        N = 35  # Number of nodes
        k = 6   # Each node is connected to k nearest neighbors in ring topology
        network = nx.watts_strogatz_graph(N, k, p)
        
        if noise_type == 'Structured Noise':
            possible_edge_changes = int(noise_level * N * (N - 1) / 2)
            edges_to_change = np.random.choice(range(N), size=(possible_edge_changes, 2), replace=True)
            for u, v in edges_to_change:
                if network.has_edge(u, v):
                    if np.random.rand() < noise_level:
                        network.remove_edge(u, v)
                else:
                    if np.random.rand() < noise_level:
                        network.add_edge(u, v)
        elif noise_type == 'Random Noise':
            all_possible_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
            np.random.shuffle(all_possible_edges)
            noise_level = np.random.normal(loc=0.5, scale=0.01)  # mean = 0.5, standard deviation = 0.15
            noise_level = max(0, min(noise_level, 1))  # Ensure noise_level stays within [0, 1]
            edge_changes = int(noise_level * len(all_possible_edges))
            for edge in all_possible_edges[:edge_changes]:
                if network.has_edge(*edge):
                    network.remove_edge(*edge)
                else:
                    network.add_edge(*edge)
        
        path_length = nx.average_shortest_path_length(network)
        clustering_coeff = nx.average_clustering(network)
        avg_path_lengths.append(path_length)
        clustering_coeffs.append(clustering_coeff)
    
    plt.figure(figsize=(8, 8))
    pos = nx.circular_layout(network)
    nx.draw(network, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title(f'Small-World Network with {noise_type}')
    plt.show()
    
    return np.mean(avg_path_lengths), np.mean(clustering_coeffs)

# Adding a dropdown for noise type selection
noise_type_dropdown = widgets.Dropdown(
    options=['Structured Noise', 'Random Noise'],
    value='Structured Noise',
    description='Noise Type:',
)

# Updating the widgets section
noise_level_slider = widgets.FloatSlider(value=0.3, min=0, max=1.0, step=0.005, description='Noise Level:')
num_simulations_slider = widgets.IntSlider(value=1, min=1, max=1000, step=1, description='Num Simulations:')
p_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=.001, description='P:')

button = widgets.Button(description="Show Graph")
output = widgets.Output()

def on_button_clicked(b):
    with output:
        clear_output(wait=True)
        mean_path_length, mean_clustering_coeff = create_and_display_network(
            noise_level_slider.value, num_simulations_slider.value, p_slider.value, noise_type_dropdown.value)
        print(f"Average Path Length: {mean_path_length:.4f}")
        print(f"Average Clustering Coefficient: {mean_clustering_coeff:.4f}")

button.on_click(on_button_clicked)

display(noise_level_slider, num_simulations_slider, p_slider, noise_type_dropdown, button, output)
