import gin
import tensorflow as tf



mol = gin.i_o.from_smiles.to_mol('CC')
mol = gin.deterministic.hydrogen.add_hydrogen(mol)
x = tf.Variable(tf.random.normal((256, 8, 3), dtype=tf.float32))
opt = tf.keras.optimizers.Adam(10)

for dummy_idx in range(500):
    with tf.GradientTape() as tape:
        bond_energy, angle_energy, one_four_energy, nonbonded_energy = gin.deterministic.mm.alkane_energy.alkane_energy(
            mol[0], mol[1], x)

        loss = tf.reduce_sum(bond_energy) + tf.reduce_sum(angle_energy)

    print(loss)
    grads = tape.gradient(loss, [x])
    opt.apply_gradients(zip(grads, [x]))


import numpy as np
np.save('x', x.numpy())
