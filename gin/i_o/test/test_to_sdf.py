import gin

mols = list(
    gin.i_o.from_sdf.to_ds('data/caffeine.sdf'))
print(mols)
gin.i_o.to_sdf.write_sdf(mols, 'caffeine_out.sdf')
