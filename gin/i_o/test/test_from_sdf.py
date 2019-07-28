import gin
ds = gin.i_o.from_sdf.to_ds('data/mols.sdf', True)
ds = ds.map(lambda atoms, adjacency_map, coordinates, charges:\
    (atoms, adjacency_map, charges))
ds = gin.probabilistic.gn.GraphNet.batch(ds, 256, per_atom_attr=True)

for x in ds:
    print(x)
