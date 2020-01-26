import gin
import flow
import tensorflow as tf
import numpy as np


chinese_postman_routes = tf.constant(
    [
        [0, 1, 4, 1, 5, 1, 2, 3, 2, 7, 2, 6],
        [0, 1, 4, 1, 5, 1, 2, 3, 2, 6, 2, 7],
        [0, 1, 4, 1, 5, 1, 2, 6, 2, 3, 2, 7],
        [0, 1, 4, 1, 5, 1, 2, 6, 2, 7, 2, 3],
        [0, 1, 4, 1, 5, 1, 2, 7, 2, 6, 2, 3],
        [0, 1, 4, 1, 5, 1, 2, 7, 2, 3, 2, 6],

        [0, 1, 5, 1, 4, 1, 2, 3, 2, 7, 2, 6],
        [0, 1, 5, 1, 4, 1, 2, 3, 2, 6, 2, 7],
        [0, 1, 5, 1, 4, 1, 2, 6, 2, 3, 2, 7],
        [0, 1, 5, 1, 4, 1, 2, 6, 2, 7, 2, 3],
        [0, 1, 5, 1, 4, 1, 2, 7, 2, 6, 2, 3],
        [0, 1, 5, 1, 4, 1, 2, 7, 2, 3, 2, 6],

        [5, 1, 0, 1, 4, 1, 2, 3, 2, 7, 2, 6],
        [5, 1, 0, 1, 4, 1, 2, 3, 2, 6, 2, 7],
        [5, 1, 0, 1, 4, 1, 2, 6, 2, 3, 2, 7],
        [5, 1, 0, 1, 4, 1, 2, 6, 2, 7, 2, 3],
        [5, 1, 0, 1, 4, 1, 2, 7, 2, 6, 2, 3],
        [5, 1, 0, 1, 4, 1, 2, 7, 2, 3, 2, 6],

        [5, 1, 4, 1, 0, 1, 2, 3, 2, 7, 2, 6],
        [5, 1, 4, 1, 0, 1, 2, 3, 2, 6, 2, 7],
        [5, 1, 4, 1, 0, 1, 2, 6, 2, 3, 2, 7],
        [5, 1, 4, 1, 0, 1, 2, 6, 2, 7, 2, 3],
        [5, 1, 4, 1, 0, 1, 2, 7, 2, 6, 2, 3],
        [5, 1, 4, 1, 0, 1, 2, 7, 2, 3, 2, 6],

        [4, 1, 5, 1, 0, 1, 2, 3, 2, 7, 2, 6],
        [4, 1, 5, 1, 0, 1, 2, 3, 2, 6, 2, 7],
        [4, 1, 5, 1, 0, 1, 2, 6, 2, 3, 2, 7],
        [4, 1, 5, 1, 0, 1, 2, 6, 2, 7, 2, 3],
        [4, 1, 5, 1, 0, 1, 2, 7, 2, 6, 2, 3],
        [4, 1, 5, 1, 0, 1, 2, 7, 2, 3, 2, 6],

        [4, 1, 0, 1, 5, 1, 2, 3, 2, 7, 2, 6],
        [4, 1, 0, 1, 5, 1, 2, 3, 2, 6, 2, 7],
        [4, 1, 0, 1, 5, 1, 2, 6, 2, 3, 2, 7],
        [4, 1, 0, 1, 5, 1, 2, 6, 2, 7, 2, 3],
        [4, 1, 0, 1, 5, 1, 2, 7, 2, 6, 2, 3],
        [4, 1, 0, 1, 5, 1, 2, 7, 2, 3, 2, 6],

    ],
    dtype=tf.int64)

mol = gin.i_o.from_smiles.to_mol('CC')
mol = gin.deterministic.hydrogen.add_hydrogen(mol)

graph_flow = flow.GraphFlow(flow_depth=4)
optimizer = tf.keras.optimizers.Adam(1e-3)
for dummy_idx in range(3000):
    with tf.GradientTape() as tape:

        x, log_det = graph_flow(
            mol[0],
            mol[1],

            tf.gather(
                chinese_postman_routes,
                tf.random.categorical(
                    tf.ones((1, 36), dtype=tf.float32),
                    32)[0]),
            std=1e-3,
            batch_size=32)

        bond_energy, angle_energy, one_four_energy, nonbonded_energy = gin.deterministic.mm.alkane_energy.alkane_energy(
            mol[0], mol[1], x)

        h = tf.reduce_sum(bond_energy) + tf.reduce_sum(angle_energy)
        ts = tf.reduce_sum(log_det)
        loss = h - ts

    print(h.numpy(), ts.numpy())
    variables = graph_flow.variables
    grad = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        zip(grad, variables))

    if dummy_idx // 100 == 0:
        np.save('x', x.numpy())
