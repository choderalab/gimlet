import gin
import tensorflow as tf
import time
import os

ds = gin.i_o.from_sdf.to_ds('/Users/yuanqingwang/Downloads/mols.sdf', True)
ds = ds.map(lambda atoms, adjacency_map, coordinates, charges:\
    (atoms, adjacency_map, charges))


# ds = ds.take(256)

ds = gin.probabilistic.gn.GraphNet.batch(ds, 64, per_atom_attr=True).cache(
    '/Users/yuanqingwang/Downloads/temp')

time0 = time.time()
for x in ds.skip(20):
    print(x)

time1 = time.time()

print(time1 - time0)
