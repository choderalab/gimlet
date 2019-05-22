import gin
ds = gin.i_o.from_sdf.read_sdf('data/caffeine.sdf')

for mol in ds:
    print(mol)
