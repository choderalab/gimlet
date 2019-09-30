# =============================================================================
# imports
# =============================================================================
from sklearn import metrics
import os
import sys
import tensorflow as tf
import gin
import lime
import pandas as pd
import numpy as np
import qcportal as ptl
client = ptl.FractalClient()

TRANSLATION = {
    6: 0,
    7: 1,
    8: 2,
    16: 3,
    15: 4,
    9: 5,
    17: 6,
    35: 7,
    53: 8,
    1: 9
}


ds_qc = client.get_collection("OptimizationDataset", "OpenFF Full Optimization Benchmark 1")
# ds_name = tf.data.Dataset.from_tensor_slices(list(ds_qc.data.records))

def data_generator():
    for record_name in list(ds_qc.data.records):
        r = ds_qc.get_record(record_name, specification='default')
        if r is not None:
            traj = r.get_trajectory()
            if traj is not None:
                for snapshot in traj:
                    energy = tf.convert_to_tensor(
                        snapshot.properties.scf_total_energy,
                        dtype=tf.float32)

                    mol = snapshot.get_molecule()

                    atoms = tf.convert_to_tensor(
                        [TRANSLATION[atomic_number] for atomic_number in mol.atomic_numbers],
                        dtype=tf.int64)

                    adjacency_map = tf.tensor_scatter_nd_update(
                        tf.zeros(
                            (
                                tf.shape(atoms, tf.int64)[0],
                                tf.shape(atoms, tf.int64)[0]
                            ),
                            dtype=tf.float32),
                        tf.convert_to_tensor(
                            np.array(mol.connectivity)[:, :2],
                            dtype=tf.int64),
                        tf.convert_to_tensor(
                            np.array(mol.connectivity)[:, 2],
                            dtype=tf.float32))

                    features = gin.probabilistic.featurization.featurize_atoms(
                        atoms, adjacency_map)

                    xyz = tf.convert_to_tensor(
                        mol.geometry,
                        dtype=tf.float32)

                    jacobian = tf.convert_to_tensor(
                        snapshot.return_result,
                        dtype=tf.float32)

                    atoms = tf.concat(
                        [
                            features,
                            xyz,
                            jacobian
                        ],
                    axis=1)

                    yield(atoms, adjacency_map, energy)

ds = tf.data.Dataset.from_generator(
    data_generator,
    (tf.float32, tf.float32, tf.float32))

ds = gin.probabilistic.gn.GraphNet.batch(
    ds, 256, feature_dimension=18, atom_dtype=tf.float32).shuffle(
        100000,
        seed=2666).cache(
            str(os.getcwd()) + '/temp')

print(list(ds) is not None)
