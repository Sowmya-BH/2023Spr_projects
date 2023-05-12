'''
IS597-PR
Team - Sowmya Bhupatiraju

Data : The data released is in the form of a network:
      1. a collection of nodes - relate to entities, addresses, officers and intermediaries
      2.and a collection of edges - which give information about the relationships between these nodes
'''

# load libraries
import pandas as pd
import numpy as np

import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns # this isn't actually required, but it makes our plots look nice
import ipywidgets
#%matplotlib inline

import random
import itertools

from networkx.drawing.nx_agraph import graphviz_layout
import pickle
import csv



def create_graph(node_file, edge_file)->nx.DiGraph:
    """
    Create a directed graph using the specified node and edge files.
    :param node_file: List of tuples, each containing a file path and node type identifier.
    :param edge_file: File path of the file containing edges.
    :return: Directed graph object

    >>> grph = create_graph(node_file=[('Entities.csv', 'entities'),('Intermediaries.csv', 'intermediaries'), ('Officers.csv', 'officers'), ('address.csv','address')],edge_file='Edges.csv')
    >>> len(grph.nodes) > 100000
    True
    >>> grph = create_graph(node_file=[('Entities.csv', 'entities')],edge_file='Edges.csv')
    >>> grph.nodes[10000001]['name']
    'TIANSHENG INDUSTRY AND TRADING CO., LTD.'
    >>> edge_file = 'Edges.csv'
    >>> len(grph.edges) <= len(edge_file)
    True
    >>> len(grph.edges) <= len(edge_file)
    True
    >>> grph = create_graph(node_file=[('Entities.csv', 'entities'),('Intermediaries.csv', 'intermediaries'), ('Officers.csv', 'officers'), ('address.csv','address')],edge_file='Edges.csv')
    >>> grph.has_edge(15006801, 10014683)
    True
    >>> grph = create_graph(node_file=[('Entities.csv', 'entities'),('Intermediaries.csv', 'intermediaries'), ('Officers.csv', 'officers'), ('address.csv','address')],edge_file='Edges.csv')
    >>> grph[12217695][10003083]['rel_type']
    'officer_of'
    """
    for file, node_type in node_file:
        df = pd.read_csv(file,low_memory=False)

    G = nx.DiGraph()

    # read node files and create nodes with node_type attribute
    for file, node_type in node_file:
        df = pd.read_csv(file,low_memory=False)
        for row in df.itertuples(index=False):
            G.add_node(row.node_id,node_type=node_type,name=row.name)

    # read edge file and add edges to graph
    edges = pd.read_csv(edge_file, low_memory=False)

    for row in edges.itertuples(index=False):
        if G.has_node(row.START_ID) and G.has_node(row.END_ID):
            G.add_edge(row.START_ID, row.END_ID, rel_type=row.TYPE, details=row._asdict())
    return G


def plot_graph(X : nx.DiGraph):
    """
    This function plots a directed graph which is the connected subcomponent which we want to visualize
    using the NetworkX library and saves the figure as a PNG image.
    :param X: nx.DiGraph
    :return: None
    """
    #node betweenness centrality, a measure that quantifies how much a node lies on paths between other nodes
    betCent = nx.betweenness_centrality(X, normalized=True, endpoints=True)# middle node -->greatest centrality - doctests
    # set your figure size
    plt.figure(figsize=(20, 20))
    # set your position
    pos = nx.kamada_kawai_layout(X)
    # set your node_options dictionary and pass it to nx.draw_networkx_nodes()
    node_options = {"node_color": "black", "node_size": [v * 10000 for v in betCent.values()]}
    # set your edge_options dictionary and pass it to nx.draw_networkx_edges()
    edge_options = {"width": 1, "alpha": 0.5, "edge_color": "gray"}
    # set your node label
    node_label_options = {"font_size": 10, "font_color": "blue", "verticalalignment": "bottom",
                          "horizontalalignment": "left"}
    # Draw nodes
    nx.draw_networkx_nodes(X, pos, nx.get_node_attributes(X, 'name'), **node_options)
    # Draw edges
    nx.draw_networkx_edges(X, pos, **edge_options)
    # Draw labels
    nx.draw_networkx_labels(X, pos, **node_label_options, labels=nx.get_node_attributes(X, 'name'))
    # Draw edgelabels
    nx.draw_networkx_edge_labels(X, pos, edge_labels=nx.get_edge_attributes(X, 'rel_type'))

    #plt.show()
    #plot as a PNG image with the filename "figure.png"
    plt.savefig("figure.png")
    #close the plot window
    plt.close()



def calculate_components(graph: nx.Graph, min_size: int) -> nx.Graph:
    """
    Calculates connected components of a graph with at least `min_size` nodes.

    :param graph: networkx.Graph object
    :param min_size: minimum size of a component to be included in the returned list
    :return: List of networkx.Graph objects representing connected components with at least `min_size` nodes.

    >>> G = nx.DiGraph()
    >>> isinstance(G, nx.DiGraph)
    True

    >>> G = nx.Graph()  # Test with a graph with 2 connected components
    >>> G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    >>> G.add_edges_from([(6, 7), (7, 8), (8, 9), (9, 10)])
    >>> components = calculate_components(G, min_size=2)
    >>> len(components)
    2
    >>> components[0].nodes()
    NodeView((1, 2, 3, 4, 5))
    >>> components[1].nodes()
    NodeView((6, 7, 8, 9, 10))

    # Test with a graph with 1 connected component
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (2, 3), (6, 7), (7, 8), (8, 9)])
    >>> components = calculate_components(G, min_size=1)
    >>> len(components)
    2
    >>> components[0].nodes()
    NodeView((6, 7, 8, 9))
    """
    # Convert the input graph to an undirected graph
    undirected_graph = graph.to_undirected()
    # Find all connected components in the undirected graph that have at least min_size nodes or edges
    components = [nx.subgraph(undirected_graph, p) for p in nx.connected_components(undirected_graph) if
                  len(p) >= min_size or undirected_graph.subgraph(p).size() >= min_size]
    # Sort the connected components in descending order of size (number of nodes)
    return sorted(components, key=lambda x: x.number_of_nodes(), reverse=True)




def calculate_degree(largest_subgraph: nx.DiGraph, panama : nx.DiGraph, page_rank) -> pd.DataFrame:
    """
    Calculate degree for nodes in largest subgraph
    :param largest_subgraph: Graph object
    :param panama: Graph object
    :page_rank : boolean
    :return: Pandas dataframe with degree for each node

    """
    if page_rank==True:
        # Calculate PageRank for each node
        pr = dict(nx.pagerank(largest_subgraph))
        node_attrs = compute_node_attributes(pr, largest_subgraph, panama)

        # Create dataframe with node types, degrees, and names
        pr_degree_df = pd.DataFrame.from_dict({"name": node_attrs[n]['name'], "node_type": node_attrs[n]['node_type'], "degree": pr[n]} for n in pr.keys() if n in node_attrs.keys())
        pr_degree_df = pr_degree_df.set_index(pr.keys())

        return pr_degree_df

    else:
        # Get node types for each node
        types = [largest_subgraph.nodes[n]["node_type"] for n in largest_subgraph.nodes()]
        # Get degree for each node
        degrees = [largest_subgraph.degree(n) for n in largest_subgraph.nodes()]
        # Get name for each node
        names = [largest_subgraph.nodes[n]['name'] for n in largest_subgraph.nodes()]
        # Create dataframe with node types, degrees, and names
        node_degree_df = pd.DataFrame(data={"node_type": types, "degree": degrees, "name": names},index=largest_subgraph.nodes())

        return node_degree_df

def compute_node_attributes(sorted_dict:dict,largest_subgraph:nx.DiGraph, panama:nx.DiGraph)->dict:
    """
    Computes node attributes for nodes in the largest connected component of the Panama Papers graph.

    :param sorted_dict: a dictionary containing the pagerank scores of each node in the graph
    :param largest_subgraph: the largest connected component of the Panama Papers graph
    :param panama: the Panama Papers graph
    :return: a dictionary containing the node attributes of nodes in the largest connected component
    >>> G = nx.DiGraph()
    >>> G.add_node(1, name='node1', node_type='entity')
    >>> G.add_node(2, name='node2', node_type='officer')
    >>> G.add_edges_from([(1, 2), (2, 1)])
    >>> largest_subgraph = max(nx.weakly_connected_components(G), key=len)
    >>> largest_subgraph = G.subgraph(largest_subgraph)
    >>> pagerank = nx.pagerank(largest_subgraph)
    >>> node_attributes = compute_node_attributes(pagerank, largest_subgraph, G)
    >>> assert isinstance(node_attributes, dict)
    >>> assert set(node_attributes.keys()) == {1, 2}
    >>> assert node_attributes[1] == {'name': 'node1', 'node_type': 'entity'}
    >>> assert node_attributes[2] == {'name': 'node2', 'node_type': 'officer'}
    """
    # Create an empty dictionary to store the node attributes
    node_attributes = dict()
    # Loop through each node in the sorted dictionary
    for node in sorted_dict.keys():
        # Check if the node is in the largest connected component
        if node in largest_subgraph:

            # If the node is in the largest connected component, add its attributes to the node attributes dictionary
            node_attributes[node] = panama.nodes[node]

    # Return the dictionary containing the node attributes
    return node_attributes



if __name__ == '__main__':
    pass
