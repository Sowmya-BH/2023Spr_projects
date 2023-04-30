
'''


'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx


#### Data preparation

df = pd.DataFrame({'ID':[1,2,3,4,5,6],
                   'First Name':['Raju', '  Madhuri', 'Rahul', 'John', 'Madleine', 'Peter'],
                   'Family Name': ['Bhupati', 'Ghaulat', 'Guntagani', 'Hull', 'Conrad', 'Donovan'],
                   'Phone number': ['+33 6 12 34 56 78', '+33 7 00 00 00 00', '+33 6 12 34 56 78', '+33 6 99 99 99 99', '+852 0123 4567', '+852 0123 4567'],
                   'Email': ['felix.revert@gmail.com', 'jean.durand@gmail.com', 'j.custom@gmail.com', np.nan, 'j.custom@gmail.com', np.nan]})

column_edge = 'Phone number'
column_ID = 'ID'

data_to_merge = df[[column_ID, column_edge]].dropna(subset=[column_edge]).drop_duplicates() # select columns, remove NaN

# To create connections between people who have the same number,
# join data with itself on the 'ID' column.
data_to_merge = data_to_merge.merge(
    data_to_merge[[column_ID, column_edge]].rename(columns={column_ID:column_ID+"_2"}),
    on=column_edge
)

print(data_to_merge)

# By joining the data with itself, people will have a connection with themselves.
# Remove self connections, to keep only connected people who are different.
d = data_to_merge[~(data_to_merge[column_ID] == data_to_merge[column_ID + "_2"])] \
    .dropna()[[column_ID, column_ID + "_2", column_edge]]

# To avoid counting twice the connections (person 1 connected to person 2 and person 2 connected to person 1)
# we force the first ID to be "lower" then ID_2
d.drop(d.loc[d[column_ID + "_2"] < d[column_ID]].index.tolist(), inplace=True)


####Creation of the graph

G = nx.from_pandas_edgelist(df=d, source=column_ID, target=column_ID+'_2', edge_attr=column_edge)

G.add_nodes_from(nodes_for_adding=df.ID.tolist())

G.nodes()


#### Emails
column_edge = 'Email'
column_ID = 'ID'

data_to_merge = df[[column_ID, column_edge]].dropna(subset=[column_edge]).drop_duplicates()

data_to_merge = data_to_merge.merge(
    data_to_merge[[column_ID, column_edge]].rename(columns={column_ID:column_ID+"_2"}),
    on=column_edge
)

d = data_to_merge[~(data_to_merge[column_ID]==data_to_merge[column_ID+"_2"])] \
    .dropna()[[column_ID, column_ID+"_2", column_edge]]

d.drop(d.loc[d[column_ID+"_2"]<d[column_ID]].index.tolist(), inplace=True)
d

# Create the connections in the graph
links_attributes = {tuple(row[[column_ID, column_ID+"_2"]]): {column_edge: row[column_edge]} for i,row in d.iterrows()}

G.add_edges_from(links_attributes) # create the connection, without attribute. Check it with G.get_edge_data(3,5)
nx.set_edge_attributes(G=G, values=links_attributes) # adds the attribute. Check it with G.get_edge_data(3,5)


#### Graph analysis
nx.draw(G)


{row[column_ID]:row['First Name'] +' '+ row['Family Name'] for i,row in df.iterrows()}

nx.set_node_attributes(G, {row[column_ID]:{'Name': row['First Name'] +' '+ row['Family Name']} for i,row in df.iterrows()})

EDGE_SIZE = {
    'Phone number': 2,
    'Email': 1,
}

EDGE_COLOR = {
    'Phone number': 'purple',
    'Email': 'red',
}


def clean_edge(edge):
    s.edge[edge[0], edge[1]].values()


def c_(list_edges): return [a for a in list_edges if a in list(EDGE_COLOR.keys())]


# For nx.Graph()
def edge_sizes(s): return [EDGE_SIZE[c_(list(s.edges[edge[0], edge[1]].keys()))[-1]] for edge in
                           s.edges()]  # /!\ multiple links => one size


def edge_colors(s): return [EDGE_COLOR[c_(list(s.edges[edge[0], edge[1]].keys()))[-1]] for edge in
                            s.edges()]  # /!\ multiple links => one color


# For nx.MultiDiGraph()
# def edge_sizes(s): return [EDGE_SIZE[s.edge[edge[0]][edge[1]][0]['label']] for edge in s.edges()] # /!\ multiple links => one size
# def edge_colors(s): return [EDGE_COLOR[s.edge[edge[0]][edge[1]][0]['label']] for edge in s.edges()] # /!\ multiple links => one color

# def draw(s):
#     pos = nx.spring_layout(s, scale=0.5)
#     node_labels = dict(print(n, d['Name']) for n, d in s.nodes(data=True))
#     #     labels = {**node_labels, **edge_labels}
#     nx.draw(s, pos=pos, width=edge_sizes(s), edge_color=edge_colors(s), alpha=0.8, arrows=False, node_color='lightgrey',
#             node_size=400,
#             labels=node_labels,
#             font_color='black', font_size=8, font_weight='bold',
#             )
#     edge_labels = dict(((u, v), list(d.values())[0]) for u, v, d in G.edges(data=True))
#
def draw(s):
    pos = nx.spring_layout(s, scale=0.5)
    if isinstance(s, nx.MultiGraph) or isinstance(s, nx.MultiDiGraph):
        node_labels = dict((n, d['Name']) for n, d in s.nodes(data=True))
    else:
        node_labels = dict((n, d.get('Name', '')) for n, d in s.nodes(data=True))
    nx.draw(s, pos=pos, width=edge_sizes(s), edge_color=edge_colors(s), alpha=0.8, arrows=False, node_color='lightgrey', node_size=400,
            labels=node_labels,
            font_color='black', font_size=8, font_weight='bold',
           )
    if isinstance(s, nx.MultiGraph) or isinstance(s, nx.MultiDiGraph):
        edge_labels = dict(((u,v), list(d.values())[0]) for u, v, d in s.edges(data=True))
    else:
        edge_labels = dict(((u, v), d.get('label', '')) for u, v, d in s.edges(data=True))
    nx.draw_networkx_edge_labels(s, pos, edge_labels=edge_labels, font_size=8, font_weight='bold')


print(draw(nx.ego_graph(G=G, n=1, radius=3)))



## Automated detection of strange patterns
degrees = list(nx.degree(G))

pd.DataFrame([list(d) for d in degrees], columns=['ID', 'degree']).sort_values('degree', ascending=False).head(5)


draw(nx.ego_graph(G=G, n=3, radius=2))