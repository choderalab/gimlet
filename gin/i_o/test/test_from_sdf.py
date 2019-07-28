import gin
ds = gin.i_o.from_sdf.to_ds('data/mols.sdf', True)

for x in ds:
    print(x)
