import tensorflow as tf
import gin
import itertools
import networkx as nx
import numpy as np

def chinese_postman(adjacency_map):
    """ Implementation of Chinese postman algorithm.

    """

    n_atoms = tf.shape(adjacency_map)[0]

    # make adjacency map all ones and zeros
    adjacency_map = tf.where(
        tf.greater(
            adjacency_map,
            tf.constant(0, dtype=tf.float32)),
        tf.ones_like(
            adjacency_map),
        tf.zeros_like(
            adjacency_map))

    g = nx.from_numpy_matrix(adjacency_map.numpy())
    nx.set_edge_attributes(g, 'o', 'note')

    adjacency_map = tf.math.add(
        adjacency_map,
        tf.transpose(
            adjacency_map))

    v_degree = tf.reduce_sum(
        adjacency_map,
        axis=0)

    is_odd_node = tf.logical_not(
        tf.equal(
            tf.math.mod(
                v_degree,
                tf.constant(2, dtype=tf.float32)),
        tf.constant(0, dtype=tf.float32)))

    odd_nodes = tf.squeeze(
        tf.where(
            is_odd_node))

    n_odd_nodes = tf.shape(odd_nodes)[0]

    odd_node_pairs_ = tf.concat(
        [
            tf.expand_dims(
                tf.tile(
                    tf.expand_dims(
                        odd_nodes,
                        axis=0),
                    [
                        tf.shape(odd_nodes)[0],
                        1
                    ]),
                axis=2),
            tf.expand_dims(
                tf.tile(
                    tf.expand_dims(
                        odd_nodes,
                        axis=1),
                    [
                        1,
                        tf.shape(odd_nodes)[0],
                    ]),
                axis=2),


        ],
        axis=2)

    odd_node_pairs = tf.reshape(
        odd_node_pairs_,
        [-1, 2])

    odd_node_pairs = tf.boolean_mask(
        odd_node_pairs,
        tf.less(
            odd_node_pairs[:, 0],
            odd_node_pairs[:, 1]))

    odd_node_pairs_distances = tf.zeros(
        shape=(
            tf.shape(odd_node_pairs)[0],),
        dtype=tf.float32)

    path_count = adjacency_map

    idx = tf.constant(1, dtype=tf.float32)

    all_path_found = tf.constant(False)

    while all_path_found == tf.constant(False):
        odd_node_num_paths = tf.gather_nd(
            path_count,
            odd_node_pairs)

        odd_node_pairs_distances = tf.where(
            tf.logical_and(
                tf.equal(
                    odd_node_pairs_distances,
                    tf.constant(0, dtype=tf.float32)),
                tf.greater(
                    odd_node_num_paths,
                    tf.constant(0, dtype=tf.float32))),
            tf.multiply(
                idx,
                tf.ones_like(
                    odd_node_pairs_distances)),
            odd_node_pairs_distances)

        all_path_found = tf.reduce_all(
            tf.greater(
                odd_node_pairs_distances,
                tf.constant(0, dtype=tf.float32)))

        path_count = tf.matmul(
            adjacency_map,
            path_count)

        idx += tf.constant(1, dtype=tf.float32)

    odd_node_pairs_distances_matrix = tf.tensor_scatter_nd_update(
        tf.tensor_scatter_nd_update(
            tf.zeros((n_atoms, n_atoms), dtype=tf.float32),
            odd_node_pairs,
            odd_node_pairs_distances),
        tf.reverse(odd_node_pairs, axis=[1]),
        odd_node_pairs_distances)

    odd_nodes_possible_pairs = tf.reshape(
        tf.gather(
            odd_nodes,
            list(itertools.permutations(
                    list(range(int(n_odd_nodes)))))),
        [-1, n_odd_nodes / 2, 2])

    odd_nodes_possible_pairs = tf.boolean_mask(
        odd_nodes_possible_pairs,
        tf.reduce_all(
            tf.less(
                odd_nodes_possible_pairs[:, :, 0],
                odd_nodes_possible_pairs[:, :, 1]),
            axis=1))

    odd_nodes_pairs_distances = tf.reduce_sum(
        tf.gather_nd(
            odd_node_pairs_distances_matrix,
            odd_nodes_possible_pairs),
        axis=1)

    odd_nodes_orders = tf.argsort(odd_nodes_pairs_distances)
    _0, _1, count = tf.unique_with_counts(tf.sort(odd_nodes_pairs_distances))
    odd_nodes_orders = odd_nodes_orders[:count[0]]

    odd_nodes_chosen = tf.gather(
        odd_nodes_possible_pairs,
        odd_nodes_orders)

    euler_circuit_list_grand = []
    for idx0 in range(int(tf.shape(odd_nodes_chosen)[0])):
        euler_circuit_list = []
        g_ = nx.MultiGraph(g.copy())
        for idx1 in range(n_odd_nodes / 2):
            pair = odd_nodes_chosen[idx0, idx1, :].numpy()
            g_.add_edge(
                pair[0],
                pair[1],
                **{'note': 'a'})

        for source in range(int(n_atoms)):
            naive_circuit = nx.eulerian_circuit(g_, source=source)
            euler_circuit = []
            for edge in naive_circuit:
                edge_data = g_.get_edge_data(edge[0], edge[1])
                if edge_data[0]['note'] == 'o':
                    euler_circuit.append((edge[0], edge[1]))

                else:
                    aug_path = nx.shortest_path(g, edge[0], edge[1])
                    aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))
                    for edge_aug in aug_path_pairs:
                        euler_circuit.append((edge_aug[0], edge_aug[1]))

            euler_circuit_list.append(euler_circuit)

        euler_circuit_list = np.unique(np.array(euler_circuit_list)[:, :, 1], axis=0)
        euler_circuit_list_grand.append(euler_circuit_list)

    euler_circuit_list_grand = np.array(euler_circuit_list_grand)
    euler_circuit_list_grand = np.reshape(
        euler_circuit_list_grand,
        [-1, euler_circuit_list_grand.shape[-1]])
    euler_circuit_list_grand = np.unique(euler_circuit_list_grand, axis=0)
    return tf.convert_to_tensor(
        euler_circuit_list_grand,
        dtype=tf.int64)
