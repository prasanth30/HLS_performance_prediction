import networkx as nx
import torch

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Example: Load DGX file
def parse_dgx_file(dgx_file_path):
    # Parse the DGX format using NetworkX
    G = nx.read_gml(dgx_file_path)
    
    # Convert to PyTorch Geometric format
    graph_data = from_networkx(G)
    return graph_data

# Example usage: Parse graph for a kernel
graph_data = parse_dgx_file('kernel_graph.dgx')


