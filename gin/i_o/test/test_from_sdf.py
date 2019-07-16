import gin
ds = gin.i_o.from_sdf.to_ds('data/caffeine.sdf')

ds = list(ds)
print(ds)
