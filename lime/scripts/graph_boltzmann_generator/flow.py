import tensorflow as tf
import graph_conv
import gin
import is_new

class GraphFlow(tf.keras.Model):
    """ Graph flow model.
    """

    def __init__(
            self,
            dense_units=64,
            gru_units=128,
            graph_conv_units=64,
            flow_depth=4):

        super(GraphFlow, self).__init__()

        self.gru_xyz = tf.keras.layers.GRU(
            units=gru_units)
        self.gru_xyz_0 = tf.keras.layers.GRU(
            units=gru_units,
            return_sequences=True)
        self.gru_xyz_1 = tf.keras.layers.GRU(
            units=gru_units,
            return_sequences=True)

        self.gru_graph_forward = tf.keras.layers.GRU(
            units=gru_units,
            return_sequences=True,
            return_state=True)
        self.gru_graph_forward_0 = tf.keras.layers.GRU(
            units=gru_units,
            return_sequences=True)
        self.gru_graph_forward_1 = tf.keras.layers.GRU(
            units=gru_units,
            return_sequences=True)

        self.gru_graph_backward = tf.keras.layers.GRU(
            units=gru_units,
            return_sequences=True,
            return_state=True,
            go_backwards=True)
        self.gru_graph_backward_0 = tf.keras.layers.GRU(
            units=gru_units,
            return_sequences=True,
            go_backwards=True)
        self.gru_graph_backward_1 = tf.keras.layers.GRU(
            units=gru_units,
            return_sequences=True,
            go_backwards=True)


        self.graph_conv = graph_conv.GraphConv(
            units=graph_conv_units)

        self.d0 = tf.keras.layers.Dense(dense_units, activation='tanh')
        self.d1 = tf.keras.layers.Dense(dense_units, activation='tanh')
        self.d2 = tf.keras.layers.Dense(dense_units, activation='tanh')

        for idx in range(1, 4):
            setattr(
                self,
                'dw' + str(idx),
                tf.keras.layers.Dense(
                    flow_depth * idx ** 2))

            setattr(
                self,
                'db' + str(idx),
                tf.keras.layers.Dense(
                    flow_depth * idx))

        self.flow_depth = flow_depth

    def flow_nd(self, z_i, seq_xyz, h_graph, dimension=3):
        # (batch_size, d)
        h_xyz = self.gru_xyz(self.gru_xyz_1(self.gru_xyz_0(seq_xyz)))

        batch_size = tf.shape(h_xyz)[0]

        # (batch_size, d)
        h_path = self.d2(
                self.d1(
                    self.d0(
                            tf.concat(
                                [
                                    h_xyz,
                                    h_graph
                                ],
                                axis=1))))

        if dimension == 1:

            w = tf.reshape(
                self.dw1(
                    h_path),
                [batch_size, -1])

            b = tf.reshape(
                self.db1(
                    h_path),
                [batch_size, -1])

            d_log_det = tf.reduce_sum(
                w)

            z_i = tf.reduce_sum(w, axis=1)

            idx = 0
            def loop_body(idx, z_i, w=w, b=b):
                z_i = tf.math.add(
                    tf.math.multiply(
                        tf.math.exp(w[:, idx]),
                        z_i),
                    b[:, idx])
                return idx + 1, z_i

            _, z_i = tf.while_loop(
                lambda idx, z_i: tf.less(idx, self.flow_depth),
                loop_body,
                [idx, z_i],
                parallel_iterations=self.flow_depth)

        else:
            w = tf.reshape(
                getattr(self, 'dw' + str(dimension))(h_path),
                [batch_size, -1, dimension, dimension])

            b = tf.reshape(
                getattr(self, 'db' + str(dimension))(h_path),
                [batch_size, -1, dimension])

            # ldu decomposition
            l = tf.linalg.set_diag(
                    tf.linalg.band_part(
                        w,
                        -1, 0),
                    tf.ones_like(b))

            d = tf.linalg.set_diag(
                    tf.zeros_like(w),
                    tf.math.exp(
                        tf.linalg.diag_part(
                            w)))

            u = tf.linalg.set_diag(
                    tf.linalg.band_part(
                        w,
                        0, -1),
                    tf.ones_like(b))

            d_log_det = tf.reduce_sum(
                tf.linalg.diag_part(
                    w),
                axis=[1, 2])

            # (batch_size, flow_depth, dimension, dimension)
            w = tf.matmul(
                l,
                tf.matmul(
                    d,
                    u))

            idx = 0
            def loop_body(idx, z_i, w=w, b=b):
                # (batch_size, dimension)
                z_i = tf.math.add(
                    tf.einsum(
                        'ab, abd -> ad',
                        z_i, # (batch_size, 1, dimension)
                        w[:, idx, :, :]), # (batch_size, dimension, dimension)
                    b[:, idx, :]) # (batch_size, dimension)
                return idx + 1, z_i
            _, z_i = tf.while_loop(
                lambda idx, z_i: tf.less(idx, self.flow_depth),
                loop_body,
                [idx, z_i],
                parallel_iterations=self.flow_depth)


        return tf.squeeze(z_i), d_log_det

    # @tf.function
    def call(self, atoms, adjacency_map, walk, std=1e-3, batch_size=16):
        n_atoms = tf.shape(
            atoms,
            tf.int64)[0]

        batch_size = tf.convert_to_tensor(
            batch_size,
            tf.int64)


        log_det = tf.constant(0, dtype=tf.float32)
        d_log_det = tf.constant(0, dtype=tf.float32)

        # initialize output
        x = tf.zeros(
            shape=(
                batch_size,
                n_atoms,
                tf.constant(
                    3,
                    dtype=tf.int64)),
            dtype=tf.float32)



        h_graph = self.graph_conv(atoms, adjacency_map)

        h_graph_gru_forward, h_graph_gru_forward_state = self.gru_graph_forward(
            self.gru_graph_forward_1(self.gru_graph_forward_0(
            tf.gather(
                h_graph,
                walk))))

        h_graph_gru_backward, h_graph_gru_backward_state = self.gru_graph_backward(
            self.gru_graph_backward_1(self.gru_graph_backward_0(
            tf.gather(
                h_graph,
                walk))))

        # grab the graph latent code
        h_graph = tf.concat(
            [
                tf.gather(
                    h_graph,
                    walk),
                h_graph_gru_forward,
                h_graph_gru_backward,
                tf.tile(
                    tf.expand_dims(
                        h_graph_gru_forward_state,
                        axis=1),
                    [1, tf.shape(walk, tf.int64)[1], 1]),
                tf.tile(
                    tf.expand_dims(
                        h_graph_gru_backward_state,
                        axis=1),
                    [1, tf.shape(walk, tf.int64)[1], 1]),
            ],
            axis=2)

        seq_xyz = tf.tile(
            tf.constant(
                [[[0, 0, 0]]],
                dtype=tf.float32),
            [
                batch_size,
                tf.constant(1, dtype=tf.int64),
                tf.constant(1, dtype=tf.int64)
            ])

        # ~~~~~~~~~~~~~~~~~~~~~
        # handle the second idx
        # ~~~~~~~~~~~~~~~~~~~~~

        idx = walk[:, 1]

        z1, d_log_det = self.flow_nd(
            tf.random.normal(
                stddev=std,
                shape=(batch_size, )),
            seq_xyz,
            tf.gather_nd(
                h_graph,
                tf.stack(
                    [
                        tf.range(tf.shape(walk, tf.int64)[0]),
                        idx
                    ],
                    axis=1)),
            dimension=1)

        log_det += d_log_det

        xyz = tf.stack(
            [
                tf.zeros_like(z1),
                tf.zeros_like(z1),
                z1
            ],
            axis=1)

        x = tf.tensor_scatter_nd_update(
            x,
            tf.stack(
                    [
                        tf.range(tf.shape(walk, tf.int64)[0]),
                        idx
                    ],
                    axis=1),
            xyz)


        seq_xyz = tf.concat(
            [
                seq_xyz,
                tf.expand_dims(
                    xyz,
                    axis=1)
            ],
            axis=1)

        # ~~~~~~~~~~~~~~~~~~~~
        # handle the third idx
        # ~~~~~~~~~~~~~~~~~~~~
        idx = walk[:, 2]

        z2, d_log_det  = self.flow_nd(
            tf.random.normal(
                shape=(batch_size, 2),
                stddev=std),
            seq_xyz,
            tf.gather_nd(
                h_graph,
                tf.stack(
                    [
                        tf.range(tf.shape(walk, tf.int64)[0]),
                        idx
                    ],
                    axis=1)),
            dimension=2)


        log_det += d_log_det

        xyz = xyz + tf.concat(
            [
                tf.zeros(
                    shape=(batch_size, 1),
                ),
                z2
            ],
            axis=1)

        x = tf.tensor_scatter_nd_update(
            x,
            tf.stack(
                    [
                        tf.range(tf.shape(walk, tf.int64)[0]),
                        idx
                    ],
                    axis=1),
            xyz)

        seq_xyz = tf.concat(
            [
                seq_xyz,
                tf.expand_dims(
                    xyz,
                    axis=1)
            ],
            axis=1)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # handle the rest of the walk
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~

        walk_idx = tf.constant(
            3,
            dtype=tf.int64)

        is_new_ = is_new.is_new(walk)

        def loop_body(walk_idx, seq_xyz, h_graph, x, log_det, std=std, batch_size=batch_size):
            # (batch_size, )
            idx = tf.gather(
                walk,
                walk_idx,
                axis=1)

            is_new__ = is_new_[:, walk_idx]

            _xyz, _d_log_det = self.flow_nd(
                tf.random.normal(
                    shape=(batch_size, 3),
                    stddev=std
                ),
                seq_xyz,
                tf.gather_nd(
                    h_graph,
                    tf.stack(
                        [
                            tf.range(tf.shape(walk, tf.int64)[0]),
                            idx
                        ],
                        axis=1)))

            xyz = tf.where(
                tf.tile(
                    tf.expand_dims(
                        is_new__,
                        1),
                    [1, 3]),
                _xyz + seq_xyz[:, -1, :],
                tf.gather_nd(
                    x,
                    tf.stack(
                        [
                            tf.range(batch_size),
                            idx
                        ],
                        axis=1)))

            d_log_det = tf.where(
                is_new__,
                _d_log_det,
                tf.zeros_like(_d_log_det))

            log_det += d_log_det

            x_ = x

            x = tf.tensor_scatter_nd_update(
                x,
                tf.stack(
                        [
                            tf.range(tf.shape(walk, tf.int64)[0]),
                            idx
                        ],
                        axis=1),
                xyz)

            seq_xyz = tf.concat(
                [
                    seq_xyz,
                    tf.expand_dims(
                        xyz,
                        axis=1)
                ],
                axis=1)

            walk_idx = tf.math.add(
                walk_idx,
                tf.constant(1, dtype=tf.int64))

            return walk_idx, seq_xyz, h_graph, x, log_det

        walk_idx, seq_xyz, h_graph, x, log_det = tf.while_loop(
            lambda walk_idx, seq_xyz, h_graph, x, log_det: tf.less(
                walk_idx,
                tf.shape(walk, tf.int64)[1]),
            loop_body,
            [walk_idx, seq_xyz, h_graph, x, log_det],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, 3]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, 3]),
                tf.TensorShape([])
            ])

        # x, d_log_det = self.global_workup(x, h_graph)
        # log_det += d_log_det
        return x, log_det
