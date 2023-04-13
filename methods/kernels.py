import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from utils import list_nodes_and_labels, product_graph


class BaseKernel:
    """
    Base kernel class.
    """

    def k_value(self, g1, g2):
        raise NotImplementedError

    def k_normalized(self, g1, g2):
        return self.k_value(g1, g2) / (np.sqrt(self.k_value(g2, g2) * self.k_value(g1, g1)))

    def kernel_matrix(self, X, Y, normalize=False):
        p, q = len(X), len(Y)
        k = np.zeros((p, q))
        for i in tqdm(range(p)):
            for j in range(q):
                k[i, j] = self.k_value(X[i], Y[j])
                if normalize:
                    # Cannot be done more efficiently if X and Y are distinct
                    k[i, j] /= np.sqrt(self.k_value(X[i], X[i]) * self.k_value(Y[j], Y[j]))
        return k


class UnlabeledShortestPathKernel(BaseKernel):
    """
    Computes shortest path kernel without relying on node/edges labels.
    """

    def __init__(self):
        super().__init__()

    def count_shortest_paths(self, g):
        fw = nx.floyd_warshall_numpy(g)
        fw = np.triu(fw, k=1)
        fw = np.where(fw == np.inf, 0, fw)
        fw = np.where(fw == np.nan, 0, fw)
        counts = np.bincount(fw.reshape(-1).astype(int))
        return counts

    def k_value(self, g1, g2):
        c1 = self.count_shortest_paths(g1)
        c2 = self.count_shortest_paths(g2)

        L = max(len(c1), len(c2))
        c1 = np.pad(c1[1:], (0, L - len(c1)), "constant")
        c2 = np.pad(c2[1:], (0, L - len(c2)), "constant")

        return np.sum(c1 * c2)

    def gram_matrix(self, graph_list, normalize=False):
        n = len(graph_list)
        k = np.zeros((n, n))
        for i in tqdm(range(n)):
            for j in range(i, n):
                k[i, j] = self.k_value(graph_list[i], graph_list[j])
                k[j, i] = k[i, j]

        if normalize:
            k_norm = np.zeros(k.shape)
            for i in range(n):
                for j in range(n):
                    k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])
        else:
            k_norm = k
        return k_norm


class NodeHistogramKernel(BaseKernel):
    """
    Computes node histogram kernel.
    """

    def __init__(self):
        super().__init__()

    def count_labels(self, g):
        labels = []
        node_data = g1.nodes(data=True)
        for l in range(1, len(g1.nodes) + 1):
            labels.append(node_data[l]["labels"][0])
        labels = np.array(labels)
        return np.bincount(labels.reshape(-1).astype(int))

    def process_labels(self, l1, l2):
        L = max(len(l1), len(l2))
        l1 = np.pad(l1, (0, L - len(l1)), "constant")
        l2 = np.pad(l2, (0, L - len(l2)), "constant")
        return l1, l2

    def k_value(self, g1, g2):
        l1 = self.count_labels(g1)
        l2 = self.count_labels(g2)
        l1, l2 = self.process_labels(l1, l2)
        return np.dot(l1, l2)

    def gram_matrix(self, graph_list, normalize=False):
        n = len(graph_list)
        k = np.zeros((n, n))
        for i in tqdm(range(n)):
            l1 = self.count_labels(graph_list[i])
            for j in range(i, n):
                l2 = self.count_labels(graph_list[j])
                k[i, j] = np.dot(*self.process_labels(l1, l2))
                k[j, i] = k[i, j]

        if normalize:
            k_norm = np.zeros(k.shape)
            for i in range(n):
                for j in range(n):
                    k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])
        else:
            k_norm = k
        return k_norm


class LabeledShortestPathKernel(BaseKernel):
    """
    Computes shortest path kernel taking into account node labels.
    """

    def __init__(self):
        super().__init__()

    def check_condition(self, length, set_label):
        return int(length and set_label)

    def k_value(self, g1, g2):
        # Node labels
        g1_labels = nx.get_node_attributes(g1, "labels")
        g2_labels = nx.get_node_attributes(g2, "labels")

        # Compute all pairs shortest paths in each graph
        sp_g1 = dict(nx.all_pairs_shortest_path_length(g1))
        sp_g2 = dict(nx.all_pairs_shortest_path_length(g2))

        k_val = 0
        for u1 in range(len(g1)):
            for v1 in range(u1 + 1, len(g1)):
                for u2 in range(len(g2)):
                    for v2 in range(u2 + 1, len(g2)):
                        l1 = sp_g1[u1].setdefault(v1, np.nan)
                        l2 = sp_g2[u2].setdefault(v2, np.nan)
                        set1 = set([g1_labels[u1][0], g1_labels[v1][0]])
                        set2 = set([g2_labels[u2][0], g2_labels[v2][0]])
                        k_val += self.check_condition((l1 == l2), (set1 == set2))

        return k_val

    def gram_matrix(self, graph_list, normalize=False):
        n = len(graph_list)
        k = np.zeros((n, n))
        for i in tqdm(range(n)):
            for j in range(i, n):
                k[i, j] = self.k_value(graph_list[i], graph_list[j])
                k[j, i] = k[i, j]

        if normalize:
            k_norm = np.zeros(k.shape)
            for i in range(n):
                for j in range(n):
                    k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])
        else:
            k_norm = k
        return k_norm


class EfficientLabeledShortestPathKernel(BaseKernel):
    """
    Computes shortest path kernel taking into account node labels.
    Relies mainly on numpy operations to speed the computation.
    """

    def __init__(self):
        super().__init__()

    def injection(self, p, q):
        return 50 * np.minimum(p, q) + np.maximum(p, q)

    def shortest_paths_array(self, g):
        idx = np.triu_indices(n=len(g), k=1)
        fw = nx.floyd_warshall_numpy(g)
        fw = np.triu(fw, k=1)
        fw = np.where(fw == np.inf, 0, fw)
        fw = np.where(fw == np.nan, 0, fw)
        return fw[idx]

    def labels_to_array(self, g):
        idx = np.triu_indices(n=len(g), k=1)
        labels = nx.get_node_attributes(g, "labels")
        labels = np.array([labels[i][0] for i in range(len(labels))])
        label_array = self.injection(*np.meshgrid(labels, labels, indexing="ij"))
        return label_array[idx]

    def k_value(self, g1, g2):
        # Array of shortest paths
        c1 = self.shortest_paths_array(g1)
        c2 = self.shortest_paths_array(g2)
        unique_lengths = np.intersect1d(c1[c1 != 0], c2[c2 != 0])

        # Array of edge "labels" ie the labels of its two nodes.
        labels1 = self.labels_to_array(g1)
        labels2 = self.labels_to_array(g2)

        # Unique labels
        uni_1 = np.unique(labels1[labels1 != 0], return_counts=False)
        uni_2 = np.unique(labels2[labels2 != 0], return_counts=False)

        k_val = 0
        for length in unique_lengths:
            mask1 = np.where(c1 == length, 1, 0)
            mask2 = np.where(c2 == length, 1, 0)
            for label in np.intersect1d(mask1 * labels1, mask2 * labels2):
                if label:
                    # We ignore 0s
                    bin1 = np.sum(np.where(mask1 * labels1 == label, 1, 0))
                    bin2 = np.sum(np.where(mask2 * labels2 == label, 1, 0))
                    k_val += bin1 * bin2

        return k_val

    def gram_matrix(self, graph_list, normalize=False):
        n = len(graph_list)
        k = np.zeros((n, n))
        for i in tqdm(range(n)):
            for j in range(i, n):
                k[i, j] = self.k_value(graph_list[i], graph_list[j])
                k[j, i] = k[i, j]

        if normalize:
            k_norm = np.zeros(k.shape)
            for i in range(n):
                for j in range(n):
                    k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])
        else:
            k_norm = k
        return k_norm


class RBF(BaseKernel):
    """
    RBF kernel using hand-crafted features.
    """

    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def kernel_matrix(self, X, Y):
        dists = cdist(X, Y, metric="sqeuclidean")
        K = np.exp(-0.5 * dists / self.sigma**2)
        return K

    def bincount(self, array, max_length=50):
        bcount = np.bincount(array)
        bcount = np.pad(bcount, (0, max(0, max_length - len(bcount))), "constant")
        return bcount

    def graph_diameter(self, g):
        fw = nx.floyd_warshall_numpy(g)
        fw = np.where(fw == np.inf, 0, fw)
        fw = np.where(fw == np.nan, 0, fw)
        return np.max(fw)

    def graph_features(self, g):
        # Node labels
        g_nodes = nx.get_node_attributes(g, "labels")
        g_nodes = [g_nodes[i][0] for i in range(len(g))]
        g_nodes = self.bincount(g_nodes, max_length=50)

        # Edge labels
        g_edges = nx.get_edge_attributes(g, "labels")
        g_edges = [g_edges[edge][0] for edge in g.edges]
        g_edges = self.bincount(g_edges, max_length=4)

        # Node degrees
        g_degrees = g.degree
        g_degrees = [g_degrees[i] for i in range(len(g))]
        g_degrees = self.bincount(g_degrees)[:5]

        # Graph diameters
        g_diam = self.graph_diameter(g)

        # Graph density
        g_dens = nx.density(g)
        return np.concatenate((g_nodes, g_edges, g_degrees, [g_diam], [g_dens]))

    def compute_features(self, list_graphs):
        fts = []
        for g in list_graphs:
            fts.append(self.graph_features(g).reshape(1, -1))
        X = np.concatenate(fts, axis=0)
        return X


class NthOrderWalkKernel(BaseKernel):
    """
    Computes N-th order Walk Kernel.
    """

    def __init__(self, n):
        super().__init__()
        self.n = n

    def k_value(self, g1, g2):
        prod = product_graph(g1, g2)

        if not len(prod):
            # Graphs do not share labels
            return 0

        A = nx.adjacency_matrix(prod)
        v = np.ones(len(prod))
        for _ in range(self.n):
            v = A @ v
        return np.sum(v)

    def gram_matrix(self, graph_list, normalize=False):
        n = len(graph_list)
        k = np.zeros((n, n))
        for i in tqdm(range(n)):
            for j in range(i, n):
                k[i, j] = self.k_value(graph_list[i], graph_list[j])
                k[j, i] = k[i, j]

        if normalize:
            k_norm = np.zeros(k.shape)
            for i in range(n):
                for j in range(n):
                    if not k[i, j]:
                        k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])
        else:
            k_norm = k
        return k_norm


class WeisfeilerLehmanKernel(BaseKernel):
    """
    Computes the Weisfeiler Lehman Subtree Kernel.
    """

    def __init__(self, method, depth, sigma=None):
        super().__init__()
        self.method = method
        self.depth = depth
        if self.method == "rbf":
            assert sigma is not None, "Sigma value required for RBF Kernel."
        self.sigma = sigma

    def compute_wl_features(self, g: nx.Graph) -> dict:
        """Computes a dictionary mapping each WL label to its count.

        Args:
            g (nx.Graph): Input graph.
            depth (int): Number of iterations of WL procedure.

        Returns:
            dict: Maps each label to its frequency in the graph.
        """

        N = len(g)
        _, labels = list_nodes_and_labels(g)

        # Initialize labels with present atoms
        node_labels = np.zeros((self.depth, N), dtype=object)
        count = np.bincount(labels)
        idx = count.nonzero()
        multisets = dict(zip(idx[0].astype(str), count[idx]))
        node_labels[0, :] = labels.astype(str)

        N_sets = len(multisets)

        # Iterate h times
        for i in range(1, self.depth):
            step_multisets = {}
            for u in range(N):
                # Node u label
                local_label = str(node_labels[i - 1, u])

                # Neighbours labels
                neighbour_labels = sorted([node_labels[i - 1, v] for v in g.neighbors(u)])

                # New label using local and sorted neighbouring labels
                new_label = local_label + "@" + "".join([str(v) for v in neighbour_labels])

                # Either creates entry at 1 or adds 1 to existing one
                step_multisets[new_label] = step_multisets.get(new_label, 0) + 1

                # Updates labels
                node_labels[i, u] = new_label

            if N_sets == len(step_multisets):
                # No additional labels found
                return multisets
            else:
                N_sets = len(step_multisets)
                multisets.update(step_multisets)
        return multisets

    def k_value(self, idx1, idx2):
        # Features for each graph
        f1 = self.X_wl[idx1]
        f2 = self.Y_wl[idx2]

        # Labels in common in both graphs
        common_labels = f1.keys() & f2.keys()

        if self.method == "linear":
            k = 0
            for label in common_labels:
                k += f1[label] * f2[label]

        elif self.method == "rbf":
            l2_diff = 0
            for label in common_labels:
                l2_diff += (f1[label] - f2[label]) ** 2

            # Takes into account non-common labels
            labels1, labels2 = set(f1.keys()), set(f2.keys())
            for label in labels1 - labels2:
                l2_diff += f1[label] ** 2
            for label in labels2 - labels1:
                l2_diff += f2[label] ** 2

            k = np.exp(-0.5 * l2_diff / self.sigma**2)

        return k

    def gram_matrix(self, graph_list, normalize=False):
        n = len(graph_list)

        # WL Features computation
        self.X_wl = {}
        self.Y_wl = {}
        for idx, g in enumerate(graph_list):
            feats = self.compute_wl_features(g)
            self.X_wl[idx] = feats
            self.Y_wl[idx] = feats

        k = np.zeros((n, n))
        for i in tqdm(range(n)):
            for j in range(i, n):
                k[i, j] = self.k_value(idx1=i, idx2=j)
                k[j, i] = k[i, j]

        if normalize:
            k_norm = np.zeros(k.shape)
            for i in range(n):
                for j in range(n):
                    if not k[i, j]:
                        k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])
        else:
            k_norm = k
        return k_norm

    def kernel_matrix(self, X, Y, normalize=False):
        # Creates the WL features for the graphs lists
        self.X_wl = {}
        self.Y_wl = {}
        for idx, g in enumerate(X):
            self.X_wl[idx] = self.compute_wl_features(g)
        for idx, g in enumerate(Y):
            self.Y_wl[idx] = self.compute_wl_features(g)

        p, q = len(X), len(Y)
        k = np.zeros((p, q))
        for i in tqdm(range(p)):
            for j in range(q):
                k[i, j] = self.k_value(idx1=i, idx2=j)
                if normalize:
                    # Cannot be done more efficiently if X and Y are distinct
                    k[i, j] /= np.sqrt(self.k_value(idx1=i, idx2=i) * self.k_value(idx1=j, idx2=j))
        return k
