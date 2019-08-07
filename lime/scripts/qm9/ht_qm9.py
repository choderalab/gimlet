# =============================================================================
# imports
# =============================================================================
import os
import sys
import tensorflow as tf
import gin
import lime
import pandas as pd
import numpy as np


mols_ds = gin.i_o.from_sdf.to_ds(
    '/Users/yuanqingwang/Downloads/qm9/gdb9.sdf',
    has_charge=False)

attr_ds = tf.data.Dataset.from_tensor_slices(
    pd.read_csv(
        '/Users/yuanqingwang/Downloads/qm9/gdb9.sdf.csv'
        ).values[:, 1:].astype(np.float32))

mols_ds = mols_ds.map(
    lambda atoms, adjacency_map, coordinates, charges:\
        (tf.cast(atoms, tf.float32), adjacency_map, coordinates))

ds = tf.data.Dataset.zip((mols_ds, attr_ds))

ds = ds.map(
    lambda mol, attr:\
        (
            tf.concat(
                [
                    tf.expand_dims(mol[0], 1),
                    mol[2]
                ],
                axis=1),
            mol[1],
            attr
        ))

for x in ds:
    print(x)
    break

ds = gin.probabilistic.gn.GraphNet.batch(
    ds, 256, attr_dimension=19, feature_dimension=4, atom_dtype=tf.float32)


for x in ds:
    print(x)
    break
