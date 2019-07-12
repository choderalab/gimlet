import gin
import pandas as pd
import pytest
import numpy as np
import numpy.testing as npt
import tensorflow as tf

df = pd.read_csv('data/SAMPL.csv')
df = df[~df['smiles'].str.contains('B')]
df = df[~df['smiles'].str.contains('P')]
df = df[~df['smiles'].str.contains('\+')]
df = df[~df['smiles'].str.contains('\-')]
smiles_array = df[['smiles']].values.flatten()

'''
# NOTE: we test everything twice, before and after hydrogenation
# @pytest.mark.parametrize('smiles', smiles_array)
def test_hybridization_mutually_exclusive(smiles):
    mol = gin.i_o.from_smiles.smiles_to_mol(smiles)
    typing = gin.deterministic.typing.Typing(mol)
    hybridization = tf.concat(
        [
            tf.expand_dims(
                typing.is_sp1,
                0),
            tf.expand_dims(
                typing.is_sp2,
                0),
            tf.expand_dims(
                typing.is_sp3,
                0)
        ],
        axis=0)

    tf.debugging.Assert(
        tf.reduce_all(
            tf.equal(
                tf.math.count_nonzero(
                    hybridization,
                    axis=0),
                tf.constant(1, dtype=tf.int64))),
        [smiles, hybridization])

    mol = gin.deterministic.hydrogen.add_hydrogen(mol)

    typing = gin.deterministic.typing.Typing(mol)
    hybridization = tf.concat(
        [
            tf.expand_dims(
                typing.is_sp1,
                0),
            tf.expand_dims(
                typing.is_sp2,
                0),
            tf.expand_dims(
                typing.is_sp3,
                0)
        ],
        axis=0)

    tf.debugging.Assert(
        tf.reduce_all(
            tf.equal(
                tf.math.count_nonzero(
                    hybridization,
                    axis=0),
                tf.constant(1, dtype=tf.int64))),
        [smiles, hybridization])



# @pytest.mark.parametrize('smiles', smiles_array)
def test_element_mutually_exclusive(smiles):
    mol = gin.i_o.from_smiles.smiles_to_mol(smiles)
    typing = gin.deterministic.typing.Typing(mol)
    elements = tf.concat(
        [
            tf.expand_dims(
                typing.is_carbon,
                0),
            tf.expand_dims(
                typing.is_nitrogen,
                0),
            tf.expand_dims(
                typing.is_oxygen,
                0),
            tf.expand_dims(
                typing.is_sulfur,
                0),
            tf.expand_dims(
                typing.is_chlorine,
                0),
            tf.expand_dims(
                typing.is_flourine,
                0),
            tf.expand_dims(
                typing.is_iodine,
                0)
        ],
        axis=0)

    tf.debugging.Assert(
        tf.reduce_all(
            tf.equal(
                tf.math.count_nonzero(
                    elements,
                    axis=0),
                tf.constant(1, dtype=tf.int64))),
        [smiles, elements])

@pytest.mark.parametrize('smiles', smiles_array)
def test_number_of_connection__mutually_exclusive(smiles):
    mol = gin.i_o.from_smiles.smiles_to_mol(smiles)
    typing = gin.deterministic.typing.Typing(mol)
    connection_cases = tf.concat(
        [
            tf.expand_dims(
                typing.is_connected_to_1,
                0),
            tf.expand_dims(
                typing.is_connected_to_2,
                0),
            tf.expand_dims(
                typing.is_connected_to_3,
                0),
            tf.expand_dims(
                typing.is_connected_to_4,
                0)
        ],
        axis=0)

    tf.debugging.Assert(
        tf.reduce_all(
            tf.equal(
                tf.math.count_nonzero(
                    connection_cases,
                    axis=0),
                tf.constant(1, dtype=tf.int64))),
        [smiles, connection_cases])

    mol = gin.deterministic.hydrogen.add_hydrogen(mol)

    typing = gin.deterministic.typing.Typing(mol)
    connection_cases = tf.concat(
        [
            tf.expand_dims(
                typing.is_connected_to_1,
                0),
            tf.expand_dims(
                typing.is_connected_to_2,
                0),
            tf.expand_dims(
                typing.is_connected_to_3,
                0),
            tf.expand_dims(
                typing.is_connected_to_4,
                0)
        ],
        axis=0)

    tf.debugging.Assert(
        tf.reduce_all(
            tf.equal(
                tf.math.count_nonzero(
                    connection_cases,
                    axis=0),
                tf.constant(1, dtype=tf.int64))),
        [smiles, connection_cases])

    connection_cases_heavy = tf.concat(
        [
            tf.expand_dims(
                typing.is_connected_to_1_heavy,
                0),
            tf.expand_dims(
                typing.is_connected_to_2_heavy,
                0),
            tf.expand_dims(
                typing.is_connected_to_3_heavy,
                0),
            tf.expand_dims(
                typing.is_connected_to_4_heavy,
                0)
        ],
        axis=0)

    tf.debugging.Assert(
        tf.reduce_all(
            tf.equal(
                tf.math.count_nonzero(
                    connection_cases_heavy,
                    axis=0),
                tf.constant(1, dtype=tf.int64))),
        [smiles, connection_cases_heavy])

'''
@pytest.mark.parametrize('smiles', smiles_array)
def test_GAFF_typing_mutually_exclusive(smiles):
    mol = gin.i_o.from_smiles.smiles_to_mol(smiles)
    typing = gin.deterministic.typing.TypingGAFF(mol)
    gaff_typing = [
        tf.expand_dims(
            getattr(typing, 'is_' + str(idx)).__call__(),
            0) \
        for idx in range(1, 36)]

    gaff_typing = tf.concat(
        gaff_typing,
        axis=0)

    tf.debugging.Assert(
        tf.reduce_all(
            tf.equal(
                tf.math.count_nonzero(
                    gaff_typing,
                    axis=0),
                tf.constant(1, dtype=tf.int64))),
        [
            smiles,
            gaff_typing
        ])
