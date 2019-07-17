import gin
import tonic
import pandas as pd
import pytest
import numpy as np
import numpy.testing as npt
import tensorflow as tf
import tonic.for_biologists.architectures.vcharge


def test_mini_fit():
    ds = gin.i_o.from_sdf.to_ds('data/mols.sdf', True)

    ds = ds.map(lambda atoms, adjacency_map, coordinates, charges:
        (
            tf.py_function(
                lambda atoms, adjacency_map: \
                    tf.reshape(
                        tonic.for_biologists.architectures.vcharge\
                        .VChargeTyping(
                        [atoms, adjacency_map]).get_assignment(),
                    [-1]),
                [atoms, adjacency_map],
                tf.int64),
            adjacency_map,
            coordinates,
            charges
        ))

    charge_model = tonic.for_biologists.architectures.vcharge.VCharge()
    tonic.for_biologists.architectures.vcharge.train(
        ds,
        charge_model)





test_mini_fit()



'''
df = pd.read_csv('data/SAMPL.csv')
df = df[~df['smiles'].str.contains('B')]
# df = df[~df['smiles'].str.contains('P')]
df = df[~df['smiles'].str.contains('\+')]
df = df[~df['smiles'].str.contains('\-')]
smiles_array = df[['smiles']].values.flatten()

@pytest.mark.parametrize('smiles', smiles_array)
def test_VCharge_typing_mutually_exclusive(smiles):
    mol = gin.i_o.from_smiles.smiles_to_mol(smiles)
    mol = gin.deterministic.hydrogen.add_hydrogen(mol)
    typing = tonic.for_biologists.architectures.vcharge.VChargeTyping(mol)
    gaff_typing = [
        tf.expand_dims(
            getattr(typing, 'is_' + str(idx)).__call__(),
            0) \
        for idx in range(1, 40)]

    gaff_typing = tf.concat(
        gaff_typing,
        axis=0)


    print(tf.where(
        tf.logical_not(
            tf.equal(
                tf.math.count_nonzero(
                    gaff_typing,
                    axis=0),
                tf.constant(1, dtype=tf.int64)))))

    print(tf.boolean_mask(
        gaff_typing,
        tf.logical_not(
            tf.equal(
                tf.math.count_nonzero(
                    gaff_typing,
                    axis=0),
                tf.constant(1, dtype=tf.int64))),
            axis=1))

    tf.debugging.Assert(
        tf.reduce_all(
            tf.equal(
                tf.math.count_nonzero(
                    gaff_typing,
                    axis=0),
                tf.constant(1, dtype=tf.int64))),
        data=[smiles])
'''
