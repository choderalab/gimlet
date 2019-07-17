import gin
ds = gin.i_o.from_sdf.to_ds('data/molecule_1.sdf', True)

ds = list(ds)
print(ds)
