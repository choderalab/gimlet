import gin
import flow
import tensorflow as tf
import numpy as np
import lime

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

graph_flow = flow.GraphFlow(flow_depth=8, whiten=True)
optimizer = tf.keras.optimizers.Adam(1e-2)
mol_optimizer = tf.keras.optimizers.Adam(10)
baoab = lime.nets.integrators.BAOAB(h=1e-4)

x = tf.Variable(
    tf.random.normal(
        shape=(32, 8, 3),
        dtype=tf.float32))

print('================================')
print('conformation initialization')
print('================================')

for _ in range(300):
    with tf.GradientTape() as tape:
        bond_energy, angle_energy, one_four_energy, nonbonded_energy = gin.deterministic.mm.alkane_energy.alkane_energy(
            mol[0], mol[1], x)

        energy = tf.reduce_sum(bond_energy) + tf.reduce_sum(angle_energy)

    print('E=', energy.numpy())
    grads = tape.gradient(energy, [x])
    mol_optimizer.apply_gradients(zip(grads, [x]))

for epoch_idx in range(10000):
    with tf.GradientTape(persistent=True) as tape:
        lamb = epoch_idx / 10000.

        z, log_det = graph_flow.f_xz(
            x,
            mol[0],
            mol[1],

            tf.gather(
                chinese_postman_routes,
                tf.random.categorical(
                    tf.ones((1, 36), dtype=tf.float32),
                    32)[0]))

        h_xz = tf.reduce_sum(tf.square(z))
        ts_xz = tf.reduce_sum(log_det)

        loss_xz = tf.reduce_sum(tf.square(z)) - tf.reduce_sum(log_det)

        z = tf.random.normal((32, 6, 3))

        x_, log_det = graph_flow.f_zx(
            z,
            mol[0],
            mol[1],
            tf.gather(
                chinese_postman_routes,
                tf.random.categorical(
                    tf.ones((1, 36), dtype=tf.float32),
                    32)[0]))

        bond_energy, angle_energy, one_four_energy, nonbonded_energy = gin.deterministic.mm.alkane_energy.alkane_energy(
            mol[0], mol[1], x_)

        h_zx = tf.reduce_sum(bond_energy) + tf.reduce_sum(angle_energy)
        ts_zx = tf.reduce_sum(log_det)
        loss_zx = tf.reduce_sum(h_zx) - tf.reduce_sum(ts_zx)

        bond_energy, angle_energy, one_four_energy, nonbonded_energy = gin.deterministic.mm.alkane_energy.alkane_energy(
            mol[0], mol[1], x)

        energy = tf.reduce_sum(bond_energy) + tf.reduce_sum(angle_energy)

        loss = loss_xz * (1 - lamb) + loss_zx * lamb

    print()
    print('H_XZ= ', h_xz.numpy(), ', TS_XZ= ', ts_xz.numpy())
    print('H_ZX= ', h_zx.numpy(), ', TS_ZX= ', ts_zx.numpy())
    print()

    grads = tape.gradient(loss, graph_flow.variables)
    optimizer.apply_gradients(zip(grads, graph_flow.variables))
    baoab.apply_gradients(
        zip(
            tape.gradient(energy, [x]),
            [x]))
