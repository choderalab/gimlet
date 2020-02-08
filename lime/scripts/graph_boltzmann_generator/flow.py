import tensorflow as tf
import graph_conv
import gin

class GraphFlow(tf.keras.Model):
    """ Graph flow model.
    """

    def __init__(
            self,
            dense_units=32,
            gru_units=32,
            graph_conv_units=32,
            flow_depth=4,
            whiten=True):

        super(GraphFlow, self).__init__()

        self.gru_xyz = tf.keras.layers.GRU(
            units=gru_units,
            return_sequences=True)
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
        self.whiten = whiten

        self.p_xy = tf.constant(
            [[0, 1, 0],
             [1, 0, 0],
             [0, 0, 1]],
            dtype=tf.int64)

        self.p_xz = tf.constant(
            [[0, 0, 1],
             [0, 1, 0],
             [1, 0, 0]],
            dtype=tf.int64)

        self.p_yz = tf.constant(
            [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]],
            dtype=tf.int64)

    @staticmethod
    def flow_zx(z_i, w, b):

        if tf.shape(tf.shape(z_i))[0] == 2:
            dimension = tf.shape(z_i)[-1]

        else:
            dimension = 1

        flow_depth = tf.shape(w)[1]

        if tf.equal(dimension, 1):
            d_log_det = tf.reduce_sum(
                tf.math.log(tf.math.abs(w) + 1e-1),
                axis=1)

            idx = 0
            def loop_body(idx, z_i, w=w, b=b):

                z_i = tf.math.add(
                    tf.math.multiply(
                        tf.math.abs(w[:, idx]) + 1e-1,
                        z_i),
                    b[:, idx])

                return idx + 1, z_i

            _, z_i = tf.while_loop(
                lambda idx, z_i: tf.less(idx, flow_depth),
                loop_body,
                [idx, z_i],
                parallel_iterations=flow_depth)

        else:
            # ldu decomposition
            l = tf.linalg.set_diag(
                    tf.linalg.band_part(
                        w,
                        -1, 0),
                    tf.ones_like(b))

            d = tf.linalg.set_diag(
                    tf.zeros_like(w),
                    tf.math.abs(
                        tf.linalg.diag_part(
                            w)) + 1e-1)

            u = tf.linalg.set_diag(
                    tf.linalg.band_part(
                        w,
                        0, -1),
                    tf.ones_like(b))

            d_log_det = tf.reduce_sum(
                tf.math.log(tf.math.abs(1e-1 +
                tf.linalg.diag_part(
                    w))),
                axis=[1, 2])

            # (batch_size, flow_depth, dimension, dimension)
            w = tf.matmul(
                l,
                tf.matmul(
                    d,
                    u))

            p_xy = tf.constant(
                [[0, 1, 0],
                 [1, 0, 0],
                 [0, 0, 1]],
                dtype=tf.float32)

            p_xz = tf.constant(
                [[0, 0, 1],
                 [0, 1, 0],
                 [1, 0, 0]],
                dtype=tf.float32)

            p_yz = tf.constant(
                [[0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 0]],
                dtype=tf.float32)

            idx = 0
            def loop_body(idx, z_i, w=w, b=b,
                    p_xy=p_xy, p_yz=p_yz, p_xz=p_xz):
                # (batch_size, dimension)
                z_i = tf.math.add(
                    tf.einsum(
                        'ab, abd -> ad',
                        z_i, # (batch_size, dimension)
                        w[:, idx, :, :]), # (batch_size, dimension, dimension)
                    b[:, idx, :]) # (batch_size, dimension)

                if dimension == 3:
                    p = [p_xy, p_yz, p_xz][idx % 3]

                    z_i = tf.einsum(
                        'ab, bd -> ad',
                        z_i,
                        p)

                return idx + 1, z_i

            _, z_i = tf.while_loop(
                lambda idx, z_i: tf.less(idx, flow_depth),
                loop_body,
                [idx, z_i])

        return z_i, d_log_det

    @staticmethod
    def flow_xz(x, w, b):

        dimension = tf.shape(tf.shape(x))[0]

        flow_depth = tf.shape(w)[1]

        z_i = x

        w = tf.reverse(w, axis=[1])
        b = tf.reverse(b, axis=[1])

        if dimension == 1:

            log_det = tf.reduce_sum(tf.math.log(tf.math.abs(w + 1e-1)), axis=1)

            idx = 0

            def loop_body(idx, z_i, w=w, b=b):
                z_i = tf.math.multiply(
                    tf.math.pow(tf.math.abs(w[:, idx] + 1e-1), -1.),
                    tf.math.subtract(
                        z_i,
                        b[:, idx]))

                return idx + 1, z_i

            _, z_i = tf.while_loop(
                lambda idx, z_i: tf.less(idx, flow_depth),
                loop_body,
                [idx, z_i],
                parallel_iterations=flow_depth)

        elif dimension == 2:

            # ldu decomposition
            l = tf.linalg.set_diag(
                    tf.linalg.band_part(
                        w,
                        -1, 0),
                    tf.ones_like(b))

            d = tf.linalg.set_diag(
                    tf.zeros_like(w),
                    tf.math.abs(1e-1 +
                        tf.linalg.diag_part(
                            w)))

            u = tf.linalg.set_diag(
                    tf.linalg.band_part(
                        w,
                        0, -1),
                    tf.ones_like(b))

            log_det = tf.reduce_sum(tf.math.log(tf.math.abs(1e-1 +
                tf.linalg.diag_part(
                    w))),
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
                z_i = tf.einsum(
                        'ab, abd -> ad',
                        tf.math.subtract(
                            z_i, # (batch_size, 1, dimension)
                            b[:, idx, :]),
                        tf.linalg.inv(w[:, idx, :, :])) # (batch_size, dimension, dimension)
                return idx + 1, z_i

            _, z_i = tf.while_loop(
                lambda idx, z_i: tf.less(idx, flow_depth),
                loop_body,
                [idx, z_i],
                parallel_iterations=flow_depth)

        else:

            # ldu decomposition
            l = tf.linalg.set_diag(
                    tf.linalg.band_part(
                        w,
                        -1, 0),
                    tf.ones_like(b))

            d = tf.linalg.set_diag(
                    tf.zeros_like(w),
                    tf.math.abs(1e-1 +
                        tf.linalg.diag_part(
                            w)))

            u = tf.linalg.set_diag(
                    tf.linalg.band_part(
                        w,
                        0, -1),
                    tf.ones_like(b))

            log_det = tf.reduce_sum(tf.math.log(tf.math.abs(1e-1 +
                tf.linalg.diag_part(
                    w))),
                axis=[1, 3])

            w = tf.matmul(
                l,
                tf.matmul(
                    d,
                    u))

            idx = 0
            def loop_body(idx, z_i, w=w, b=b):
                # (batch_size, n_walks, 3)
                z_i = tf.einsum(
                        'abc, abcd -> abd',
                        tf.math.subtract(
                            z_i, # (batch_size, 1, dimension)
                            b[:, idx, :, :]),
                        tf.linalg.inv(w[:, idx, :, :, :])) # (batch_size, dimension, dimension)
                return idx + 1, z_i

            _, z_i = tf.while_loop(
                lambda idx, z_i: tf.less(idx, flow_depth),
                loop_body,
                [idx, z_i],
                parallel_iterations=flow_depth)

        return z_i, log_det

    @staticmethod
    def whitening(seq_xyz):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # rearrange based on the first three atoms
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # to be
        # (0, 0, 0)
        # (0, 0, z1)
        # (0, y2, z2)

        # translation
        seq_xyz = tf.math.subtract(
            seq_xyz,
            tf.tile(
                tf.expand_dims(
                    seq_xyz[:, 0, :],
                    1),
                [1, tf.shape(seq_xyz)[1], 1]))

        # rotation
        # rotate along z so that y1 = 0
        cos = tf.math.divide_no_nan(
            seq_xyz[:, 1, 0],
            tf.norm(
                seq_xyz[:, 1, :2],
                axis=1))

        sin = tf.math.divide_no_nan(
            seq_xyz[:, 1, 1],
            tf.norm(
                seq_xyz[:, 1, :2],
                axis=1))

        # make sure `divide_no_nan` aren't triggered twice
        sin = tf.where(
            tf.equal(
                cos,
                tf.constant(0, dtype=tf.float32)),
            tf.constant(1, dtype=tf.float32),
            sin)

        r = tf.reshape(
            tf.stack(
                [
                    cos,
                    -sin,
                    tf.zeros_like(cos),
                    sin,
                    cos,
                    tf.zeros_like(cos),
                    tf.zeros_like(cos),
                    tf.zeros_like(cos),
                    tf.ones_like(cos)
                ],
                axis=1),
            [-1, 3, 3])

        seq_xyz = tf.einsum(
            'abc, aeb -> aec',
            r,
            seq_xyz)

        # rotate along y so that z1 = 0
        cos = tf.math.divide_no_nan(
            seq_xyz[:, 1, 2],
            tf.norm(
                tf.stack(
                    [
                        seq_xyz[:, 1, 0],
                        seq_xyz[:, 1, 2]
                    ],
                    axis=1),
                axis=1))

        sin = tf.math.divide_no_nan(
            seq_xyz[:, 1, 0],
            tf.norm(
                tf.stack(
                    [
                        seq_xyz[:, 1, 0],
                        seq_xyz[:, 1, 2]
                    ],
                    axis=1),
                axis=1))

        # make sure `divide_no_nan` aren't triggered twice
        sin = tf.where(
            tf.equal(
                cos,
                tf.constant(0, dtype=tf.float32)),
            tf.constant(1, dtype=tf.float32),
            sin)

        r = tf.reshape(
            tf.stack(
                [
                    cos,
                    tf.zeros_like(cos),
                    sin,
                    tf.zeros_like(cos),
                    tf.ones_like(cos),
                    tf.zeros_like(cos),
                    -sin,
                    tf.zeros_like(cos),
                    cos
                ],
                axis=1),
            [-1, 3, 3])

        seq_xyz = tf.einsum(
            'abc, aeb -> aec',
            r,
            seq_xyz)

        # rotate along z so that x2 = 0
        cos = tf.math.divide_no_nan(
            seq_xyz[:, 2, 0],
            tf.norm(
                seq_xyz[:, 2, :2],
                axis=1))

        sin = tf.math.divide_no_nan(
            seq_xyz[:, 2, 1],
            tf.norm(
                seq_xyz[:, 2, :2],
                axis=1))

        # make sure `divide_no_nan` aren't triggered twice
        sin = tf.where(
            tf.equal(
                cos,
                tf.constant(0, dtype=tf.float32)),
            tf.constant(1, dtype=tf.float32),
            sin)

        r = tf.reshape(
            tf.stack(
                [
                    sin,
                    cos,
                    tf.zeros_like(cos),
                    -cos,
                    sin,
                    tf.zeros_like(cos),
                    tf.zeros_like(cos),
                    tf.zeros_like(cos),
                    tf.ones_like(cos)
                ],
                axis=1),
            [-1, 3, 3])

        seq_xyz = tf.einsum(
            'abc, aeb -> aec',
            r,
            seq_xyz)

        # make sure what's close to zero is zero
        seq_xyz = tf.transpose(
            tf.tensor_scatter_nd_update(
                tf.transpose(
                    seq_xyz,
                    [1, 2, 0]),
                tf.constant(
                    [[0, 0],
                     [0, 1],
                     [0, 2],
                     [1, 0],
                     [1, 1],
                     [2, 0]],
                    dtype=tf.int64),
                tf.zeros(
                    shape=(
                        6,
                        tf.shape(seq_xyz)[0]),
                    dtype=tf.float32)),
            [2, 0, 1])

        return seq_xyz

    @staticmethod
    def is_new(walk):
        # (n_batch, n_atoms)
        is_virgin = tf.constant(
            True,
            shape=(
                tf.shape(walk)[0],
                tf.shape(tf.unique(walk[0])[0])[0]))

        # (n_batch, n_walks)
        is_new_ = tf.constant(
            False,
            shape=tf.shape(walk))

        for idx in range(tf.shape(walk)[1]):
            walk_row = walk[:, idx]

            walk_row_is_virgin = tf.gather_nd(
                is_virgin,
                tf.stack(
                    [
                        tf.range(
                            tf.shape(walk_row, tf.int64)[0]),
                        walk_row

                    ],
                    axis=1))

            virgin_idxs = tf.boolean_mask(
                tf.stack(
                    [
                        tf.range(
                            tf.shape(walk_row, tf.int64)[0]),
                        walk_row

                    ],
                    axis=1),
                walk_row_is_virgin)



            is_virgin = tf.tensor_scatter_nd_update(
                is_virgin,
                virgin_idxs,
                tf.constant(
                    False,
                    shape=(
                        tf.shape(virgin_idxs)[0],)))


            is_new_ = tf.transpose(
                tf.tensor_scatter_nd_update(
                    tf.transpose(is_new_),
                    [[idx]],
                    tf.expand_dims(walk_row_is_virgin, 0)))

        return is_new_

    @staticmethod
    def align_z(z_axis, xyz):

        cos_theta = tf.math.divide_no_nan(
            z_axis[:, 2],
            tf.linalg.norm(
                z_axis,
                axis=1))

        sin_theta = tf.math.divide_no_nan(
            tf.linalg.norm(
                z_axis[:, :2],
                axis=1),
            tf.linalg.norm(
                z_axis,
                axis=1))

        sin_theta = tf.where(
            tf.equal(
                cos_theta,
                tf.constant(0, dtype=tf.float32)),
            tf.constant(1, dtype=tf.float32),
            sin_theta)

        sin_phi = tf.math.divide_no_nan(
            z_axis[:, 1],
            tf.linalg.norm(
                z_axis[:, :2],
                axis=1))

        cos_phi = tf.math.divide_no_nan(
            z_axis[:, 2],
            tf.linalg.norm(
                z_axis[:, :2],
                axis=1))

        sin_phi = tf.where(
            tf.equal(
                cos_phi,
                tf.constant(0, dtype=tf.float32)),
            tf.constant(1, dtype=tf.float32),
            sin_phi)

        r = tf.reshape(
            tf.stack(
                [
                    cos_phi,
                    sin_phi * cos_theta,
                    sin_phi * sin_theta,
                    -sin_phi,
                    cos_phi * cos_theta,
                    cos_phi * sin_theta,
                    tf.zeros_like(cos_phi),
                    -sin_theta,
                    cos_theta
                ],
                axis=1),
            [-1, 3, 3])

        if tf.shape(tf.shape(xyz))[0] == 2:
            xyz = tf.einsum(
                'abc, ac -> ab',
                r,
                xyz)

        else:
            xyz = tf.einsum(
                'abc, aec -> aeb',
                r,
                xyz)

        return xyz

    # @staticmethod
    def align(self, seq_xyz, adjacency_map, walk, is_new_):
        batch_size = tf.shape(seq_xyz)[0]
        n_atoms = tf.shape(adjacency_map)[0]
        n_walk = tf.shape(seq_xyz)[1]

        xyz = tf.zeros(
            (batch_size, n_atoms, 3),
            dtype=tf.float32)

        adjacency_map_full = tf.math.add(
            adjacency_map,
            tf.transpose(adjacency_map))

        for idx in range(3):
            xyz = tf.tensor_scatter_nd_update(
                xyz,
                tf.stack(
                    [
                        tf.range(batch_size, dtype=tf.int64),
                        walk[:, idx]
                    ],
                    axis=1),
                seq_xyz[:, idx, :])

        def loop_body(idx, xyz,
                seq_xyz=seq_xyz,
                is_new_=is_new_,
                adjacency_map_full = adjacency_map_full,
                batch_size=batch_size):
            # grab the indices of three nodes along the route
            # (batch_size, )
            this_idx = walk[:, idx]
            parent_idx = walk[:, idx-1]

            d_xyz = seq_xyz[:, idx]

            # grab the indices of the neighbors of parent node
            parent_neighbor_idxs = tf.where(
                tf.greater(
                    tf.gather(
                        adjacency_map_full,
                        parent_idx),
                    tf.constant(0, dtype=tf.float32)))

            # grab the coordinates of the parent node
            parent_xyz = tf.gather_nd(
                xyz,
                tf.stack(
                    [
                        tf.range(batch_size, dtype=tf.int64),
                        parent_idx
                    ],
                    axis=1))

            # get the sum of the coordinates of parents' neighbor
            # (batch_size, 3)
            parent_neighbor_xyz_sum = tf.reduce_sum(
                tf.tensor_scatter_nd_update(
                    tf.zeros(
                        shape=(batch_size, n_atoms, 3),
                        dtype=tf.float32),
                    parent_neighbor_idxs,
                    tf.gather_nd(
                        xyz,
                        parent_neighbor_idxs)),
                axis=1)

            z_basis = tf.math.subtract(
                parent_xyz,
                parent_neighbor_xyz_sum)

            d_xyz = self.align_z(z_basis, d_xyz)

            # only update the new ones
            _xyz = tf.where(
                tf.tile(
                    tf.expand_dims(
                        is_new_[:, idx],
                        1),
                    [1, 3]),
                d_xyz + parent_xyz,
                tf.gather_nd(
                    xyz,
                    tf.stack(
                        [
                            tf.range(batch_size, dtype=tf.int64),
                            this_idx
                        ],
                        axis=1)))

            xyz = tf.tensor_scatter_nd_update(
                xyz,
                tf.stack(
                    [
                        tf.range(batch_size, dtype=tf.int64),
                        this_idx
                    ],
                    axis=1),
                _xyz)

            return idx+1, xyz

        idx = 3
        idx, xyz = tf.while_loop(
            lambda idx, xyz: tf.less(idx, n_walk),
            loop_body,
            [idx, xyz])

        # # (batch_size, n_walk)
        # d_log_det = tf.reduce_sum(
        #     tf.reshape(
        #         tf.boolean_mask(
        #                 tf.math.multiply(
        #                     tf.math.square(r),
        #                     tf.sin(phi)),
        #                 is_new_),
        #         [batch_size, -1]),
        #     axis=1)

        d_log_det = 0.

        return xyz, d_log_det

    def summarize_graph_state(self, atoms, adjacency_map, walk):
        batch_size = tf.shape(walk, tf.int64)[0]
        n_walk = tf.shape(walk, tf.int64)[1]

        h_graph = self.graph_conv(atoms, adjacency_map)

        adjacency_map_full = tf.math.add(
            adjacency_map,
            tf.transpose(adjacency_map))

        node_degree = tf.reduce_sum(
            adjacency_map_full,
            axis=0)

        node_degree_walk = tf.gather(
            node_degree,
            walk)

        node_degree_parent_walk = tf.concat(
            [
                tf.zeros(
                    shape=(
                        batch_size,
                        1),
                    dtype=tf.float32),
                node_degree_walk[:, 1:]
            ],
            axis=1)

        parent_neighbor_count = tf.zeros(
            (batch_size, n_walk),
            dtype=tf.float32)

        def loop_body(idx, parent_neighbor_count, walk=walk):
            parent_idx = walk[:, idx]
            this_idx = walk[:, idx+1]
            _parent_neighbor_count = tf.math.count_nonzero(
                tf.gather(
                    adjacency_map_full,
                    parent_idx),
                axis=1,
                dtype=tf.float32)
            parent_neighbor_count = tf.tensor_scatter_nd_update(
                parent_neighbor_count,
                tf.stack(
                    [
                        tf.range(batch_size),
                        this_idx
                    ],
                    axis=1),
                _parent_neighbor_count)
            return idx+1, parent_neighbor_count

        idx = 0

        _, parent_neighbor_count = tf.while_loop(
            lambda idx, parent_neighbor_count: tf.less(idx, n_walk - 1),
            loop_body,
            [idx, parent_neighbor_count])


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

        # # grab the graph latent code
        h_graph = tf.concat(
            [
                tf.expand_dims(node_degree_walk, 2),
                tf.expand_dims(node_degree_parent_walk, 2),
                tf.expand_dims(parent_neighbor_count, 2),
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

        # (n_batch, walk, d_hidden)
        return h_graph

    def summarize_geometry_state(self, seq_xyz):
        return self.gru_xyz(self.gru_xyz_1(self.gru_xyz_0(seq_xyz)))

    def get_flow_params(self, h_path, dimension=3):


        h_path_shape = tf.shape(h_path)

        if tf.shape(h_path_shape)[0] == 3:
            batch_size = h_path_shape[0]
            n_walk = h_path_shape[1]

            # NOTE: this is not efficient, but somehow the expression below
            # reshapes into the wrong result
            w = tf.stack(
                [
                    tf.reshape(
                        getattr(
                            self,
                            'dw' + str(dimension))(h_path[:, idx, :]),
                        [batch_size, -1, dimension, dimension])\
                            for idx in range(n_walk)
                ],
                axis=2)

            b = tf.stack(
                [
                    tf.reshape(
                        getattr(
                            self,
                            'db' + str(dimension))(h_path[:, idx, :]),
                        [batch_size, -1, dimension])\
                            for idx in range(n_walk)
                ],
                axis=2)

            # w = tf.reshape(
            #     w,
            #     [batch_size, -1, n_walk, dimension, dimension])
            #
            # b = tf.reshape(
            #     b,
            #     [batch_size, -1, n_walk, dimension])

        else:
            w = getattr(
                self,
                'dw' + str(dimension))(h_path)

            b = getattr(
                self,
                'db' + str(dimension))(h_path)

            batch_size = h_path_shape[0]

            if dimension == 1:
                w = tf.reshape(
                    w,
                    [batch_size, -1])

                b = tf.reshape(
                    b,
                    [batch_size, -1])

            else:
                w = tf.reshape(
                    w,
                    [batch_size, -1, dimension, dimension])

                b = tf.reshape(
                    b,
                    [batch_size, -1, dimension])

        return w, b

    # @tf.function
    def f_zx(self, z, atoms, adjacency_map, walk):

        # read the number of atoms
        n_atoms = tf.shape(
            atoms,
            tf.int64)[0]

        # the batch size is the first dimension of z
        batch_size = tf.shape(
            z,
            tf.int64)[0]

        # initialize log det
        log_det = tf.constant(0, dtype=tf.float32)

        # the latent representation of walks on the graph is independent
        # from its geometry
        h_graph = self.summarize_graph_state(
            atoms,
            adjacency_map,
            walk)

        seq_xyz = tf.tile(
            tf.constant(
                [[[0, 0, 0]]], # the first atom is placed at the center
                dtype=tf.float32),
            [
                batch_size,
                tf.constant(1, dtype=tf.int64),
                tf.constant(1, dtype=tf.int64)
            ])

        # ~~~~~~~~~~~~~~~~~~~~
        # handle the first idx
        # ~~~~~~~~~~~~~~~~~~~~
        h_path = self.d2(self.d1(self.d0(tf.concat(
            [
                self.summarize_geometry_state(seq_xyz)[:, -1, :],
                h_graph[:, 1, :]
            ],
            axis=1))))

        w, b = self.get_flow_params(h_path, dimension=1)

        z1, d_log_det = self.flow_zx(
            z[:, 0, 0],
            w,
            b)

        log_det += d_log_det

        xyz = tf.stack(
            [
                tf.zeros_like(z1),
                tf.zeros_like(z1),
                z1
            ],
            axis=1)

        seq_xyz = tf.concat(
            [
                seq_xyz,
                tf.expand_dims(
                    xyz,
                    axis=1)
            ],
            axis=1)

        # ~~~~~~~~~~~~~~~~~~~~~
        # handle the second idx
        # ~~~~~~~~~~~~~~~~~~~~~

        h_path = self.d2(self.d1(self.d0(tf.concat(
            [
                self.summarize_geometry_state(seq_xyz)[:, -1, :],
                h_graph[:, 2, :]
            ],
            axis=1))))

        w, b = self.get_flow_params(h_path, dimension=2)

        z2, d_log_det = self.flow_zx(
            z[:, 0, 1:],
            w,
            b)

        log_det += d_log_det

        xyz = tf.concat(
            [
                tf.zeros(
                    shape=(batch_size, 1),
                ),
                z2
            ],
            axis=1)

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

        # start to walk from the third index
        walk_idx = tf.constant(
            3,
            dtype=tf.int64)

        # extract the 1st entry of z
        z_idx = tf.constant(
            0,
            shape=(batch_size,),
            dtype=tf.int64)

        is_new_ = self.is_new(walk)

        def loop_body(walk_idx, seq_xyz, log_det, z_idx, h_graph=h_graph):

            # (batch_size, )
            idx = tf.gather(
                walk,
                walk_idx,
                axis=1)

            is_new__ = is_new_[:, walk_idx]


            z_idx = tf.where(
                is_new__,
                tf.math.add(
                    z_idx,
                    tf.constant(1, dtype=tf.int64)),
                z_idx)



            h_path = self.d2(self.d1(self.d0(tf.concat(
                [
                    self.summarize_geometry_state(seq_xyz)[:, -1, :],
                    h_graph[:, walk_idx, :]
                ],
                axis=1))))

            w, b = self.get_flow_params(h_path, dimension=3)

            _xyz, _d_log_det = self.flow_zx(
                tf.gather_nd(
                    z,
                    tf.stack(
                        [
                            tf.range(tf.shape(walk, tf.int64)[0]),
                            z_idx
                        ],
                        axis=1)),
                w,
                b)

            xyz = tf.where(
                tf.tile(
                    tf.expand_dims(
                        is_new__,
                        1),
                    [1, 3]),
                _xyz,
                seq_xyz[:, -1, :])

            d_log_det = tf.where(
                is_new__,
                _d_log_det,
                tf.zeros_like(_d_log_det))

            log_det += d_log_det

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

            return walk_idx, seq_xyz, log_det, z_idx

        walk_idx, seq_xyz, log_det, z_idx = tf.while_loop(
            lambda walk_idx, seq_xyz, log_det, z_idx: tf.less(
                walk_idx,
                tf.shape(walk, tf.int64)[1]),
            loop_body,
            [walk_idx, seq_xyz, log_det, z_idx],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, 3]),
                tf.TensorShape([]),
                tf.TensorShape([])
            ])

        #
        # x, d_log_det = self.align(
        #     seq_xyz, adjacency_map, walk, is_new_)
        #
        # log_det += d_log_det
        #
        seq_xyz_new = tf.reshape(
            tf.boolean_mask(
                seq_xyz,
                is_new_),
            [batch_size, -1, 3])

        walk_new = tf.reshape(
            tf.boolean_mask(
                walk,
                is_new_),
            [batch_size, -1])

        x = tf.tensor_scatter_nd_update(
            tf.zeros(
                (batch_size, n_atoms, 3),
                dtype=tf.float32),
            tf.reshape(
                tf.stack(
                    [
                        tf.tile(
                            tf.expand_dims(
                                tf.range(batch_size),
                                axis=1),
                            [1, n_atoms]),
                        walk_new
                    ],
                    axis=2),
                [-1, 2]),
            tf.reshape(
                seq_xyz_new,
                [-1, 3]))

        return x, log_det

    def f_xz(self, x, atoms, adjacency_map, walk):
        n_atoms = tf.shape(
            atoms,
            tf.int64)[0]

        batch_size = tf.shape(
            x,
            tf.int64)[0]

        log_det = tf.constant(0, dtype=tf.float32)

        h_graph = self.summarize_graph_state(atoms, adjacency_map, walk)

        # gather xyz sequence
        # (batch_size, n_walk, 3)
        seq_xyz = tf.gather_nd(
            x, # (batch_size, n_atoms, 3)
            tf.stack(
                [
                    tf.tile(
                        tf.expand_dims(
                            tf.range(
                                batch_size,
                                dtype=tf.int64),
                            axis=1),
                        [1, tf.shape(walk)[1]]),
                    walk # (batch_size, n_walk)
                ],
                axis=2))

        if self.whiten:
            seq_xyz = self.whitening(seq_xyz)

        # (batch_size, n_walk, d)
        h_xyz = self.summarize_geometry_state(seq_xyz)

        h_xyz = tf.concat(
            [
                tf.zeros(
                    shape=(
                        batch_size,
                        tf.constant(1, dtype=tf.int64),
                        tf.shape(h_xyz, tf.int64)[-1]),
                    dtype=tf.float32),
                h_xyz[:, :-1, :]
            ],
            axis=1)

        batch_size = tf.shape(h_xyz)[0]

        # (batch_size, n_walk, d)
        h_path = self.d2(self.d1(self.d0(tf.concat(
            [
                h_xyz,
                h_graph
            ],
            axis=2))))

        # ~~~~~~~~~~~~~~~~~~~~
        # handle the first idx
        # ~~~~~~~~~~~~~~~~~~~~

        w, b = self.get_flow_params(h_path[:, 1, :], dimension=1)

        z_i = seq_xyz[:, 1, -1]
        z_0_0, d_log_det = self.flow_xz(z_i, w, b)
        log_det += d_log_det

        # ~~~~~~~~~~~~~~~~~~~~~
        # handle the second idx
        # ~~~~~~~~~~~~~~~~~~~~~

        w, b = self.get_flow_params(h_path[:, 2, :], dimension=2)

        z_i = seq_xyz[:, 2, 1:]

        z_0_12, d_log_det = self.flow_xz(z_i, w, b)

        log_det += d_log_det


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # handle the rest of the indices
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # (batch_size, flow_depth, n_walks, 3, 3)
        w, b = self.get_flow_params(h_path[:, 3:, :], dimension=3)

        z_i = seq_xyz[:, 3:, :]
        z_rest, d_log_det = self.flow_xz(z_i, w, b)
        log_det += tf.reduce_sum(
            tf.reshape(
                tf.boolean_mask(
                    d_log_det,
                    self.is_new(walk)[:, 3:]),
                [batch_size, -1]),
            axis=1)

        # (batch, n_walk - 2)
        z = tf.concat(
                [
                    tf.expand_dims(
                        tf.concat(
                            [
                                tf.expand_dims(z_0_0, 1),
                                z_0_12
                            ],
                            axis=1),
                        axis=1),
                    tf.reshape(
                        tf.boolean_mask(
                            z_rest,
                            self.is_new(walk)[:, 3:]),
                        [batch_size, -1, 3])
                ],
                axis=1)

        return z, log_det
