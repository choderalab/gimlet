import gin
import tensorflow as tf
import math
import flow

CH = 0.1092

def add_hydrogen(atoms, adjacency_map, x):

    batch_size = tf.shape(x)[0]

    adjacency_map_full = tf.math.add(
        adjacency_map,
        tf.transpose(
            adjacency_map))

    n_atoms = tf.shape(atoms)[0]

    is_carbon = tf.equal(
        atoms,
        tf.constant(0, dtype=tf.int64))

    is_hydrogen = tf.equal(
        atoms,
        tf.constant(9, dtype=tf.int64))

    carbon_idxs = tf.where(is_carbon)[:, 0]

    n_carbon = tf.shape(carbon_idxs)[0]

    def loop_body(idx, x, adjacency_map=adjacency_map, n_atoms=n_atoms, batch_size=batch_size):
        carbon = carbon_idxs[idx]

        neighbor_hydrogens = tf.where(
            tf.logical_and(
                tf.greater(
                    tf.gather(
                        adjacency_map_full,
                        carbon),
                    tf.constant(0, dtype=tf.float32)),
                is_hydrogen))[:, 0]

        neighbor_carbons = tf.where(
            tf.logical_and(
                tf.greater(
                    tf.gather(
                        adjacency_map_full,
                        carbon),
                    tf.constant(0, dtype=tf.float32)),
                is_carbon))

        n_neighbor_carbons = tf.shape(neighbor_carbons)[0]
        n_neighbor_hydrogens = tf.shape(neighbor_hydrogens)[0]

        # (batch_size, 1, )
        carbon_xyz = tf.gather(
            x,
            carbon,
            axis=1)

        neighbor_carbon_xyz_sum = tf.reduce_sum(
            tf.transpose(
                tf.tensor_scatter_nd_update(
                    tf.zeros(
                        shape=(n_atoms, batch_size, 3),
                        dtype=tf.float32),
                    neighbor_carbons,
                    tf.gather_nd(
                        tf.transpose(
                            x,
                            [1, 0, 2]),
                            neighbor_carbons)),
                [1, 0, 2]),
            axis=1)

        z_basis = tf.math.subtract(
            carbon_xyz,
            neighbor_carbon_xyz_sum)

        z_basis, _ = tf.linalg.normalize(
            z_basis,
            axis=1)


        if tf.equal(n_neighbor_hydrogens, 1):

            d_xyz_h = z_basis * CH
            xyz_h = carbon_xyz + d_xyz_h

            x = tf.tensor_scatter_nd_update(
                x,
                neighbor_hydrogens,
                xyz_h)

        elif tf.equal(n_neighbor_hydrogens, 3):
            r = tf.tile(
                tf.constant([[CH, CH, CH]], dtype=tf.float32),
                [batch_size, 1])

            theta = tf.tile(
                tf.constant([[0.955, 0.955, 0.955]], dtype=tf.float32),
                [batch_size, 1])

            phi = 2.0 * math.pi * tf.random.uniform(shape=[batch_size, 1])

            phi = tf.concat([phi, phi + 2 * math.pi / 3., phi + 4 * math.pi / 3.], axis=1)


            # (batch_size, 3)
            x_ = r * tf.math.cos(theta) * tf.math.cos(phi)
            y_ = r * tf.math.cos(theta) * tf.math.sin(phi)
            z_ = r * tf.sin(theta)

            # (batch_size, 3, 3)
            d_xyz_h = tf.stack([x_, y_, z_], axis=2)


            # (batch_size, 3, 3)
            d_xyz_h = flow.GraphFlow.align_z(
                z_basis,
                d_xyz_h)

            xyz_h = tf.tile(tf.expand_dims(carbon_xyz, 1), [1, 3, 1]) + d_xyz_h

            x = tf.tensor_scatter_nd_update(
                tf.transpose(
                    x,
                    [1, 0, 2]),
                tf.expand_dims(neighbor_hydrogens, 1),
                tf.transpose(
                    xyz_h,
                    [1, 0, 2]))

            x = tf.transpose(
                x,
                [1, 0, 2])

        elif tf.equal(n_neighbor_hydrogens, 2):

            # (batch_size, 2, 3)
            neighbor_carbon_xyz = tf.gather(
                x,
                neighbor_carbons[:, 0],
                axis=1)

            z_basis = tf.squeeze(z_basis)

            x_basis = tf.linalg.cross(
                    neighbor_carbon_xyz[:, 0, :] - tf.squeeze(carbon_xyz),
                    neighbor_carbon_xyz[:, 1, :] - tf.squeeze(carbon_xyz))

            x_basis = tf.math.divide_no_nan(
                x_basis,
                tf.linalg.norm(x_basis, axis=0))

            x_ = tf.stack(
                [
                    CH * tf.math.cos(0.955),
                    -CH * tf.math.cos(0.955)
                ],
                axis=0)

            z_ = tf.stack(
                [
                    CH * tf.math.sin(0.955),
                    CH * tf.math.sin(0.955)
                ],
                axis=0)

            x_basis = tf.tile(tf.expand_dims(x_basis, 1), [1, 2, 1])
            z_basis = tf.tile(tf.expand_dims(z_basis, 1), [1, 2, 1])


            x_ = tf.tile(
                tf.expand_dims(tf.expand_dims(x_, 0), 2),
                [batch_size, 1, 3])

            z_ = tf.tile(
                tf.expand_dims(tf.expand_dims(z_, 0), 2),
                [batch_size, 1, 3])

            d_xyz_h = x_ * x_basis + z_ * z_basis

            xyz_h = tf.expand_dims(carbon_xyz, 1) + d_xyz_h

            x = tf.transpose(
                tf.tensor_scatter_nd_update(
                    tf.transpose(
                        x,
                        [1, 0, 2]),
                    tf.expand_dims(neighbor_hydrogens, 1),
                    tf.transpose(
                        xyz_h,
                        [1, 0, 2])),
                [1, 0, 2])

        return idx+1, x

    idx = 0

    _, x = tf.while_loop(
        lambda idx, x: tf.less(idx, n_carbon),
        loop_body,
        [idx, x])

    return x
