# =============================================================================
# imports
# =============================================================================
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(3)
from sklearn import metrics
import gin
import lime
import pandas as pd
import numpy as np
import qcportal as ptl
client = ptl.FractalClient()
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
FF = ForceField('test_forcefields/smirnoff99Frosst.offxml')
import cmiles
from simtk import openmm
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

HARTREE_TO_KJ_PER_MOL = 2625.5
BOHR_TO_NM = 0.0529177
HARTREE_PER_BOHR_TO_KJ_PER_MOL_PER_NM = 49614.77

ds_qc = client.get_collection("OptimizationDataset", "OpenFF Full Optimization Benchmark 1")

def data_generator():
    for record_name in random.sample(list(ds_qc.data.records), 10):
        try:
            print(record_name, flush=True)
            r = ds_qc.get_record(record_name, specification='default')
            if r is not None:
                traj = r.get_trajectory()
                if traj is not None:
                    for snapshot in traj:

                        mol = snapshot.get_molecule()
                        # mol = snapshot.get_molecule().dict(encoding='json')
                        
                        xyz = tf.convert_to_tensor(
                            mol.geometry * BOHR_TO_NM,
                            dtype=tf.float32)

                        qm_force = tf.convert_to_tensor(
                            snapshot.return_result\
                            * HARTREE_PER_BOHR_TO_KJ_PER_MOL_PER_NM,
                            dtype=tf.float32)

                        mol = cmiles.utils.load_molecule(mol.dict(encoding='json'))

                        top = Topology.from_molecules(Molecule.from_openeye(mol))
                        sys = FF.create_openmm_system(top)

                        yield(
                            xyz,
                            qm_force,
                            sys)
       
        except:
            pass

traj = tf.ones(
    shape=(1, 6),
    dtype=tf.float32)

for xyz, qm_force, sys in data_generator():
    
    context = openmm.Context(sys, openmm.VerletIntegrator(0.001))
    context.setPositions(xyz * 1.0)

    force = sys.getForce(2)
    force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    force.updateParametersInContext(context)
    # print(context.getState(getEnergy=True, groups=1<<2).getPotentialEnergy())
    
    traj = tf.concat(
        [
            traj,
            tf.concat(
                [
                    context.getState(
                        getVelocities=True,
                        getForces=True).getForces(asNumpy=True)._value,
                    qm_force
                ],
                axis=1)
        ],
        axis=0)


traj = traj[1:].numpy()
np.save('traj', traj)


plt.style.use('ggplot')
mm = traj[:, :3]
qm = traj[:, 3:]

plt.figure()
plt.scatter(np.linalg.norm(mm, axis=1), np.linalg.norm(qm, axis=1), s=1)
plt.xlabel(r'$\rho_\mathtt{mm}$')
plt.ylabel(r'$\rho_\mathtt{qm}$')
plt.savefig('rho.jpg')

plt.figure()
plt.scatter(
        np.arctan2(mm[:, 0], mm[:, 1]), 
        np.arctan2(qm[:, 0], qm[:, 1]),
        s=1)

plt.xlabel(r'$\theta_\mathtt{mm}$')
plt.ylabel(r'$\theta_\mathtt{qm}$')
plt.savefig('theta.jpg')

plt.figure()
plt.scatter(
    np.arctan2(mm[:, 0], mm[:, 2]),
    np.arctan2(qm[:, 0], qm[:, 2]),
    s = 1)

plt.xlabel(r'$\phi_\mathtt{mm}$')
plt.ylabel(r'$\phi_\mathtt{qm}$')
plt.savefig('phi.jpg')



