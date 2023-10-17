import pickle
import networkx as nx

# graphs = pickle.load(open(f'mydata/tumor/knn_10/10', 'rb'))
graphs2 = pickle.load(open(f'mydata/tumor/feat', 'rb'))
# print(nx.adjacency_matrix(graphs[1]['reg001_A']))
print(graphs2[1]['reg001_A'][-10:])