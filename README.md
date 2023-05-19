# Exploring the Business Value of Advanced Analytics on Anti-Money Laundering  Investigations:
## The Problem:
Financial crimes are more about how criminals are linked by relationships - whether it's relationship to other criminals, locations or ofcourse, bank accounts.Conceptually money laudering is simple. Dirty money is passed around to blend it with legitimate funds and then turned into hard assets. This is the kind of analysis used in Panama Paper Analysis which I used to demonstrate my hypothesis.


## Data:
* The original data is available under creative common license and can be downloaded from https://offshoreleaks.icij.org/pages/database.
* Here I used only a sample of the data which I uploaded as zip files in CSV format.

## Hypothesis 1 : Advanced analytics like graph can improve the accuracy and efficiency of Anti-Money Laundering investigations of financial crimes:
Graphs capture relationships and connections between data entities that can be used in data analysis - whether it's relationship to other criminals, locations or ofcourse, bank accounts. Much of data is connected, and graphs are becoming increasingly important because they make it easier to explore those connections and draw new conclusions. Because graph databases explicitly store the relationships, queries and algorithms utilizing the connectivity between vertices can be run in sub- seconds rather than hours or days. Users don’t need to execute countless join and the data can more easily be used for analysis and machine learning which makes it more efficient than traditional data analytics
This is kind of analysis is used in Panama Paper Analysis by the The International Consortium of Investigative Journalists (ICIJ) which released a dump of some of the information they received as part of the panama papers leak. The Panama Papers are 11.5 million leaked documents (or 2.6 terabytes of data) that were published beginning on April 3, 2016.


## Hypothesis 2 :The use of advanced analytics can provide a more comprehensive view of suspicious financial activities and uncover previously unknown patterns and relationships.
When it comes to analyzing graphs, algorithms explore the paths and distance between the vertices, the importance of the vertices, and connectedness of the vertices. The algorithms will often look at incoming edges, importance of neighboring vertices, and other indicators to help determine importance. However, In real-time fraud detection,for example, users can create a graph from transactions between entities as well as entities that share some information, including email addresses, passwords, addresses and more which makes it easier to reveal accounts with similar information and also to track which accounts are sending money to each other.Given the herculian task of analysing 559600 nodes and 6574881 edges and extracting structures from data, it would have taken days together to even understand how these nodes are related to each other using traditional dataframe analysis.


#### Challenges:
1. The size of the data files is too huge and so I have to zip them to upload to GitHub
2. Given the herculian task of plotting 559600 nodes and 6574881 edges, I have to Use multiprocessing to parallelize the computation for a few functions to run tasks in parallel to make the code run faster.

Finally, given additional time I would have reduced the respective nodes to respective unique instance of countries and see how would this work!! (both nodes and edges would require processing).


#### References:
* https://www.oracle.com/a/ocom/docs/graph-database-use-cases-ebook.pdf
* https://www.imf.org/external/np/mae/oshore/2000/eng/back.htm#II 
* https://networkx.org/documentation/stable/reference/generated/networkx.generators.ego.ego_graph.html
* https://networkx.org/documentation/stable/tutorial.html
* https://networkx.org/documentation/stable/reference/drawing.html
* https://networkx.org/documentation/stable/reference/classes/digraph.html#methods
* https://stackoverflow.com/questions/54969732/ive-an-undirected-networkx-graph-and-want-to-find-all-neighbors-of-node-a-that?rq=1
* https://www.geeksforgeeks.org/directed-graphs-multigraphs-and-visualization-in-networkx/
* https://stackoverflow.com/questions/54969732/ive-an-undirected-networkx-graph-and-want-to-find-all-neighbors-of-node-a-that?rq=1 
* https://www.kaggle.com/code/mayeesha/network-analysis-for-dummies-stackoverflow-data
* https://stackoverflow.com/questions/61929188/get-networkx-subgraph-containing-all-nodes-in-between
* https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html 
* https://medium.com/@oliviercruchant/python-networkx-graph-magic-260309cce484
* Complex Network Analysis in Python by Dmitry Zenoviev - book
* ChatGPT for error handling:
Query: Graph Visualization:
* Prompt : node_labels  returns nan
* Response : If the node_labels dictionary returns nan for some nodes, it's possible that the corresponding rows in the DataFrame don't have a value for the "name" column. You can check this by adding a print statement inside the loop that prints the row variable:

scss
Copy code
for _, row in df.iterrows():
    print(row)
    G.add_node(row['node_id'], node_type=node_type, **row.to_dict())


