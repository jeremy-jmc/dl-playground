# https://medium.com/analytics-vidhya/getting-the-intuition-of-graph-neural-networks-a30a2c34280d
# https://towardsdatascience.com/understanding-graph-convolutional-networks-for-node-classification-a2bfdb7aba7b
# https://graphneural.network/examples/
# https://towardsdatascience.com/graph-convolutional-networks-on-node-classification-2b6bbec1d042

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# * Create a simple graph
G = nx.Graph(name='G')
for i in range(6):
    G.add_node(i, name=i)

edges = [(0, 1), (0, 2), (1, 2), (0, 3), (3, 4), (3, 5), (4, 5)]
G.add_edges_from(edges)

print('Graph Info:\n', G)
print('Number of nodes', len(G.nodes))
print('Number of edges', len(G.edges))
print('Average degree', sum(dict(G.degree).values()) / len(G.nodes))

print('\nGraph Nodes: ', G.nodes.data())

nx.draw(G, with_labels=True, font_weight='bold')
plt.show()


# * Get the Adjacency Matrix (A) and Node Features Matrix (X) as numpy array
A = np.array(nx.attr_matrix(G, node_attr='name')[0])
X = np.array(nx.attr_matrix(G, node_attr='name')[1])
X = np.expand_dims(X, axis=1)

print('Shape of A: ', A.shape)
print('\nShape of X: ', X.shape)
print('\nAdjacency Matrix (A):\n', A)
print('\nNode Features Matrix (X):\n', X)

# * Inserting Adjacency Matrix (A) to Forward Pass Equation
# Dot product Adjacency Matrix (A) and Node Features (X)
# AX represents the sum of neighboring nodes features.
AX = np.dot(A, X)
print("Dot product of A and X (AX):\n", AX)
# The dot product of Adjacency Matrix and Node Features Matrix represents the sum of neighboring node features.

# * Inserting Self-Loops and Normalizing Adjacency Matrix
# Add Self Loops
G_self_loops = G.copy()

self_loops = []
for i in range(G.number_of_nodes()):
    self_loops.append((i, i))

G_self_loops.add_edges_from(self_loops)

# Check the edges of G_self_loops after adding the self loops
print('Edges of G with self-loops:\n', G_self_loops.edges)

# Get the Adjacency Matrix (A) and Node Features Matrix (X) of added self-lopps graph
A_hat = np.array(nx.attr_matrix(G_self_loops, node_attr='name')[0])
print('Adjacency Matrix of added self-loops G (A_hat):\n', A_hat)

# Calculate the dot product of A_hat and X (AX)
AX = np.dot(A_hat, X)
print('AX:\n', AX)


# * Normalizing Calculation
# In GCNs, we normalize our data by calculating the Degree Matrix (D) and performing dot product operation of the inverse of D with AX

# Get the Degree Matrix of the added self-loops graph
Deg_Mat = G_self_loops.degree()
print('Degree Matrix of added self-loops G (D): ', Deg_Mat)

# Convert the Degree Matrix to a N x N matrix where N is the number of nodes
D = np.diag([deg for (n, deg) in list(Deg_Mat)])
print('Degree Matrix of added self-loops G as numpy array (D):\n', D)

# Find the inverse of Degree Matrix (D)
D_inv = np.linalg.inv(D)
print('Inverse of D:\n', D_inv)

# Dot product of D and AX for normalization
DAX = np.dot(D_inv, AX)
print('DAX:\n', DAX)


# * Symmetrically-normalization: Kipf & Welling (ICLR 2017)
D_half_norm = fractional_matrix_power(D, -0.5)
DADX = D_half_norm.dot(A_hat).dot(D_half_norm).dot(X)
print('DADX:\n', DADX)

# In the paper, A* is referred to as renormalization trick.

# * Implementing GCN Layer
# Initialize the weights
np.random.seed(77777)
n_h = 4  # number of neurons in the hidden layer
n_y = 2  # number of neurons in the output layer
W0 = np.random.randn(X.shape[1], n_h) * 0.01
W1 = np.random.randn(n_h, n_y) * 0.01

# Implement ReLu as activation function


def relu(x):
    return np.maximum(0, x)

# Build GCN layer
# In this function, we implement numpy to simplify


def gcn(A, H, W):
    I = np.identity(A.shape[0])  # create Identity Matrix of A
    A_hat = A + I  # add self-loop to A
    D = np.diag(np.sum(A_hat, axis=0))  # create Degree Matrix of A
    D_half_norm = fractional_matrix_power(D, -0.5)  # calculate D to the power of -0.5
    print(f'{D_half_norm.shape=} {A_hat.shape=} {D_half_norm.shape=} {H.shape=} {W.shape=}')
    eq = D_half_norm.dot(A_hat).dot(D_half_norm).dot(H).dot(W)
    return relu(eq)


# Do forward propagation
H1 = gcn(A, X, W0)
H2 = gcn(A, H1, W1)
print('Features Representation from GCN output:\n', H2)
