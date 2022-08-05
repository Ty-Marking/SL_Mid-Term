import networkx
import pandas as pd
import numpy as np
from torch_geometric.data import Data

def import_raw(filepath):
    G = networkx.read_graphml(filepath)
    print(G)

    print("\n\nNodes:")
    nodes = list(G.nodes(data=True))
    nodes_dict = {i:nodes[i][1] for i in range(len(nodes))}
    nodes_df = pd.DataFrame.from_dict(nodes_dict, orient='index')
    print(nodes_df)

    #will need to process features but this is how to extract them into data
    node_features = nodes_df.columns
    x = nodes_df.loc[:,nodes_df.columns.isin(node_features)].values
    print(x)

    print("\n\nEdges:")


    edges = list(G.edges(data=True))
    edge_df=pd.DataFrame({'edge0':[edges[i][0] for i in range(len(edges))], 'edge1':[edges[i][1] for i in range(len(edges))]}, index=range(len(edges)))

    #add features
    edge_features = pd.DataFrame.from_dict({i:edges[i][2] for i in range(len(edges))}, orient='index')
    edge_df = edge_df.join(edge_features)
    print(edge_df)

    #will need to process features but this is how to extract them into data
    edge_features = ['labelE', 'VARIABLE']
    edge_index = np.asarray([edge_df['edge0'].values, edge_df['edge1'].values])
    edge_attr = np.asarray(edge_df.loc[:,edge_df.columns.isin(edge_features)].values)
    print(edge_index)
    print(edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == "__main__":
    print(import_raw("./data/export.graphml"))