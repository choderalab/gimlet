import pytest
import gin
import numpy as np
import numpy.testing as npt
import tensorflow as tf
import pandas as pd


# read data
df = pd.read_csv('data/delaney-processed.csv')
x_array = df[['smiles']].values.flatten()
y_array = df[['measured log solubility in mols per litre']].values.flatten()
y_array = (y_array - np.mean(y_array) / np.std(y_array))

ds = gin.i_o.from_smiles.smiles_to_mols_with_attributes(x_array, y_array)

ds = gin.probabilistic.gn.GraphNet.batch(ds, 256)
