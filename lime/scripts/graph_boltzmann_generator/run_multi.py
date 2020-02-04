import gin
import flow
import tensorflow as tf
import numpy as np
import lime

import chinese_postman_routes


mols = [gin.i_o.from_smiles.to_mol(idx * 'C') for idx in range(2, 4)]
mols = [gin.deterministic.hydrogen.add_hydrogen(mol) for mol in mols]
_chinese_postman_routes = [chinese_postman_routes.chinese_postman(mol[1]) for mol in mols]
n_postmen = [tf.shape(route)[0] for route in _chinese_postman_routes]

graph_flow = flow.GraphFlow(flow_depth=3, whiten=True)
optimizer = tf.keras.optimizers.Adam(1e-3)

# x = tf.Variable(
#     tf.random.normal(
#         shape=(32, 8, 3),
#         dtype=tf.float32))
#
# print('================================')
# print('conformation initialization')
# print('================================')
#
# for _ in range(300):
#     with tf.GradientTape() as tape:
#         bond_energy, angle_energy, one_four_energy, nonbonded_energy = gin.deterministic.mm.alkane_energy.alkane_energy(
#             mol[0], mol[1], x)
#
#         energy = tf.reduce_sum(bond_energy) + tf.reduce_sum(angle_energy)
#
#     print('E=', energy.numpy())
#     grads = tape.gradient(energy, [x])
#     mol_optimizer.apply_gradients(zip(grads, [x]))

for epoch_idx in range(10000):
    with tf.GradientTape(persistent=True) as tape:
        # lamb = epoch_idx / 10000.
        #
        # z, log_det = graph_flow.f_xz(
        #     x,
        #     mol[0],
        #     mol[1],
        #
        #     tf.gather(
        #         chinese_postman_routes,
        #         tf.random.categorical(
        #             tf.ones((1, 36), dtype=tf.float32),
        #             32)[0]))
        #
        # h_xz = tf.reduce_sum(tf.square(z))
        # ts_xz = tf.reduce_sum(log_det)

        # loss_xz = tf.reduce_sum(tf.square(z)) - tf.reduce_sum(log_det)
        #
        loss = 0.
        for idx in range(2):
            mol = mols[idx]

            z = tf.random.normal((8, 2 * (idx+2) + 2, 3))

            x_, log_det = graph_flow.f_zx(
                z,
                mol[0],
                mol[1],
                tf.gather(
                    _chinese_postman_routes[idx],
                    tf.random.categorical(
                        tf.ones((1, n_postmen[idx]), dtype=tf.float32),
                        8)[0]))

            bond_energy, angle_energy, one_four_energy, nonbonded_energy = gin.deterministic.mm.alkane_energy.alkane_energy(
                mol[0], mol[1], x_)

            h_zx = tf.reduce_sum(bond_energy) + tf.reduce_sum(angle_energy) # + tf.reduce_sum(one_four_energy)
            ts_zx = tf.reduce_sum(log_det)
            # # loss_zx = tf.reduce_sum(h_zx) - tf.reduce_sum(ts_zx)
            #
            # bond_energy, angle_energy, one_four_energy, nonbonded_energy = gin.deterministic.mm.alkane_energy.alkane_energy(
            #     mol[0], mol[1], x)
            #
            # energy = tf.reduce_sum(bond_energy) + tf.reduce_sum(angle_energy)

            loss += h_zx - ts_zx

        print(loss)
    grads = tape.gradient(loss, graph_flow.variables)
    optimizer.apply_gradients(zip(grads, graph_flow.variables))
    # baoab.apply_gradients(
    #     zip(
    #         tape.gradient(energy, [x]),
    #         [x]))

    if epoch_idx % 100 == 0:
        graph_flow.save_weights('graph_flow.h5')
