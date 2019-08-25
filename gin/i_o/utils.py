"""
MIT License

Copyright (c) 2019 Chodera lab // Memorial Sloan Kettering Cancer Center,
Weill Cornell Medical College, Nicea Research, and Authors

Authors:
Yuanqing Wang
Chaya Stern

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os
import tensorflow as tf

def oemol_to_dict(oemol, read_wbo=False):
    """
    Return list of elements, partial charges and connectivity with WBOs for the bonds

    Parameters
    ----------
    oemol : oechem.OEMol
        Molecule must have partial charges and Wiberg Bond Orders precalculated.

    Returns
    -------
    mol_dict: dict
        dictionary of atomic symbols, partial charges and connectivity with Wiberg Bond Orders

    """
    from openeye import oechem

    atomic_symbols = [oechem.OEGetAtomicSymbol(atom.GetAtomicNum()) for atom in oemol.GetAtoms()]
    partial_charges = [atom.GetPartialCharge() for atom in oemol.GetAtoms()]

    # Build connectivity with WBOs
    connectivity = []
    for bond in oemol.GetBonds():
        a1 = bond.GetBgn().GetIdx()
        a2 = bond.GetEnd().GetIdx()
        if read_wbo==True:
            if not 'WibergBondOrder' in bond.GetData():
                raise RuntimeError('Molecule does not have Wiberg Bond Orders')
            wbo = bond.GetData('WibergBondOrder')
            connectivity.append([a1, a2, wbo])
        else:
            bond_order = bond.GetOrder()
            connectivity.append([a1, a2, bond_order])

    mol_dict = {'atomic_symbols': atomic_symbols,
                'partial_charges': partial_charges,
                'connectivity': connectivity}

    return mol_dict

def file_to_oemols(filename):
    """Create OEMol from file. If more than one mol in file, return list of OEMols.

    Parameters
    ----------
    filename: str
        absolute path to oeb file

    Returns
    -------
    mollist: list
        list of OEMol for multiple molecules. OEMol if file only has one molecule.
    """
    from openeye import oechem

    if not os.path.exists(filename):
        raise Exception("File {} not found".format(filename))

    ifs = oechem.oemolistream(filename)
    mollist = []

    molecule = oechem.OEMol()
    while oechem.OEReadMolecule(ifs, molecule):
        molecule_copy = oechem.OEMol(molecule)
        oechem.OEPerceiveChiral(molecule_copy)
        oechem.OE3DToAtomStereo(molecule_copy)
        oechem.OE3DToBondStereo(molecule_copy)
        mollist.append(molecule_copy)
    ifs.close()

    if len(mollist) is 1:
        mollist = mollist[0]
    return mollist



def conjugate_average(atoms, adjacency_map):
    n_atoms = tf.shape(atoms, tf.int64)[0]

    adjacency_map_full = tf.transpose(adjacency_map) + adjacency_map

    # =================
    # conjugate systems
    # =================
    # search start with atoms connected to double bonds
    sp2_idxs = tf.boolean_mask(
        tf.range(n_atoms,
            dtype=tf.int64),
        tf.reduce_any(
            tf.equal(
                adjacency_map_full,
                tf.constant(2.0, dtype=tf.float32)),
            axis=1))

    # NOTE:
    # right now,
    # we only allow carbon, nitrogen, or oxygen
    # as part of our conjugate system
    sp2_idxs = tf.boolean_mask(
        sp2_idxs,
        tf.logical_or(

            tf.equal( # carbon
                tf.gather(atoms, sp2_idxs),
                tf.constant(
                    0,
                    dtype=tf.int64)),

            tf.logical_or(
                tf.equal( # nitrogen
                    tf.gather(atoms, sp2_idxs),
                    tf.constant(
                        1,
                        dtype=tf.int64)),

                tf.equal( # oxygen
                    tf.gather(atoms, sp2_idxs),
                    tf.constant(
                        2,
                        dtype=tf.int64)))))

    # gather only sp2 atoms
    sp2_adjacency_map = tf.gather(
        tf.gather(
            adjacency_map_full,
            sp2_idxs,
            axis=0),
        sp2_idxs,
        axis=1)

    # init conjugate_systems
    conjugate_systems = tf.expand_dims(
        tf.ones_like(sp2_idxs) \
            * tf.constant(-1, dtype=tf.int64),
        0)

    # init visited flags
    visited = tf.tile(
        tf.expand_dims(
            tf.constant(False),
            0),
        tf.expand_dims(
            tf.shape(sp2_idxs)[0],
            0))

    def loop_body_inner(queue, visited, conjugate_system,
            sp2_adjacency_map=sp2_adjacency_map):
        # dequeue
        idx = queue[-1]
        queue = queue[:-1]

        # flag the position of self
        is_self = tf.equal(
            tf.range(
                tf.cast(
                    tf.shape(sp2_adjacency_map)[0],
                    tf.int64),
                dtype=tf.int64),
            idx)

        # flag the neighbors
        is_neighbors = tf.greater(
            sp2_adjacency_map[idx, :],
            tf.constant(0, dtype=tf.float32))

        # flag the neighbors that are not visited
        is_unvisited_neighbors = tf.logical_and(
            is_neighbors,
            tf.logical_not(
                visited))

        # check if self is visited
        self_is_unvisited = tf.logical_not(visited[idx])

        # put self in the conjugate system if self is unvisited
        conjugate_system = tf.cond(
            self_is_unvisited,

            # if self.is_visited
            lambda: tf.concat( # put self into the conjugate_system
                [
                    conjugate_system,
                    tf.expand_dims(
                        idx,
                        0)
                ],
                axis=0),

            # else:
            lambda: conjugate_system)

        # get the states of the neighbors
        neighbors_unvisited = tf.boolean_mask(
            tf.range(
                tf.cast(
                    tf.shape(sp2_adjacency_map)[0],
                    tf.int64),
                dtype=tf.int64),
            is_unvisited_neighbors)

        # append the undiscovered neighbors to the conjugate system
        conjugate_system = tf.concat(
            [
                conjugate_system,
                neighbors_unvisited
            ],
            axis=0)

        # enqueue
        queue = tf.concat(
            [
                queue,
                neighbors_unvisited
            ],
            axis=0)

        # change the visited flag
        visited = tf.where(
            tf.logical_or(
                is_unvisited_neighbors,
                is_self),

            # if sp2_atom is neighbor and is unvisited:
            tf.tile(
                tf.expand_dims(
                    tf.constant(True),
                    0),
                tf.expand_dims(
                    tf.shape(is_unvisited_neighbors)[0],
                    0)),

            # else:
            visited)

        return queue, visited, conjugate_system

    def loop_body_outer(
            conjugate_systems,
            visited,
            sp2_adjacency_map=sp2_adjacency_map):

        # start with the first element that's not visited
        queue = tf.expand_dims(
            tf.boolean_mask(
                tf.range(
                    tf.cast(
                        tf.shape(sp2_adjacency_map)[0],
                        tf.int64),
                    dtype=tf.int64),
                tf.logical_not(visited))[0],
            0)

        # init conjugate system
        conjugate_system = tf.constant([], dtype=tf.int64)

        queue, visited, conjugate_system = tf.while_loop(
            # while len(queue) > 0:
            lambda queue, visited, conjugate_system: tf.greater(
                tf.shape(queue)[0],
                tf.constant(0, tf.int32)),

            # execute inner loop body
            loop_body_inner,

            # loop var
            [queue, visited, conjugate_system],

            shape_invariants=[
                tf.TensorShape([None, ]),
                tf.TensorShape(visited.get_shape()),
                tf.TensorShape([None, ])])

        conjugate_system = tf.cond(
            # check if there are at least three atoms in the conjugate system
            tf.greater_equal(
                tf.shape(conjugate_system)[0],
                tf.constant(3, dtype=tf.int32)),

            # if there is more than three atoms in the conjugate system:
            lambda: conjugate_system,

            # else:
            lambda: tf.constant([], dtype=tf.int64))

        # pad it with -1
        conjugate_system = tf.concat(
            [
                conjugate_system,
                tf.ones(tf.shape(sp2_idxs)[0] \
                    - tf.shape(conjugate_system)[0],
                    dtype=tf.int64) \
                    * tf.constant(-1, dtype=tf.int64)
            ],
            axis=0)

        # put conjugate system into the list of systems
        conjugate_systems = tf.concat(
            [
                conjugate_systems,
                tf.expand_dims(
                    conjugate_system,
                    0)
            ],
            axis=0)

        return conjugate_systems, visited

    conjugate_systems, visited = tf.while_loop(
        # while not all are visited
        lambda conjugate_systems, visited: tf.reduce_any(
            tf.logical_not(
                visited)),

        # loop_body
        loop_body_outer,

        # loop var
        [conjugate_systems, visited],

        shape_invariants=[
            tf.TensorShape([None, sp2_idxs.shape[0]]),
            tf.TensorShape((visited.get_shape()))])


    # get rid of the empty entries
    conjugate_systems = tf.boolean_mask(
        conjugate_systems,
        tf.reduce_any(
            tf.greater(
                conjugate_systems,
                tf.constant(-1, dtype=tf.int64)),
            axis=1))


    # while loop
    idx = tf.constant(0, dtype=tf.int32)
    bond_idxs_to_update = tf.constant([[-1, -1]], dtype=tf.int64)
    bond_orders_to_update = tf.constant([-1], dtype=tf.float32)

    # update the bond order according to conjugate_systems
    def loop_body(idx,
            bond_idxs_to_update,
            bond_orders_to_update,
            conjugate_systems=conjugate_systems,
            sp2_idxs=sp2_idxs,
            sp2_adjacency_map=sp2_adjacency_map):

        # get the conjugate system and get rid of the padding
        conjugate_system = conjugate_systems[idx, :]
        conjugate_system = tf.boolean_mask(
            conjugate_system,
            tf.greater(
                conjugate_system,
                tf.constant(-1, dtype=tf.int64)))

        # get the bonds in the system
        system_idxs = tf.gather(
            sp2_idxs,
            conjugate_system)

        # get the full adjacency map
        system_adjacency_map_full = tf.gather(
            tf.gather(
                sp2_adjacency_map,
                conjugate_system,
                axis=0),
            conjugate_system,
            axis=1),

        # flatten it to calculate average bond
        system_adjacency_map_flatten = tf.reshape(
            tf.linalg.band_part(
                system_adjacency_map_full,
                0, -1),
            [-1])

        # get the average bond order and count
        system_total_bond_order = tf.reduce_sum(system_adjacency_map_flatten)
        system_total_bond_count = tf.reduce_sum(
            tf.boolean_mask(
                tf.ones_like(
                    system_adjacency_map_flatten,
                    dtype=tf.float32),
                tf.greater(
                    system_adjacency_map_flatten,
                    tf.constant(0, dtype=tf.float32))))

        system_mean_bond_order = tf.math.divide(
            system_total_bond_order,
            system_total_bond_count)

        # dirty stuff to get the bond indices to update
        all_idxs_x, all_idxs_y = tf.meshgrid(
            tf.range(n_atoms, dtype=tf.int64),
            tf.range(n_atoms, dtype=tf.int64))

        all_idxs_stack = tf.stack(
            [
                all_idxs_y,
                all_idxs_x
            ],
            axis=2)

        system_idxs_2d = tf.gather(
            tf.gather(
                all_idxs_stack,
                system_idxs,
                axis=0),
            system_idxs,
            axis=1)

        system_idxs_2d = tf.boolean_mask(
            system_idxs_2d,
            tf.greater(
                tf.gather_nd(
                    adjacency_map,
                    system_idxs_2d),
                tf.constant(0, dtype=tf.float32)))

        bond_idxs_to_update = tf.concat(
            [
                bond_idxs_to_update,
                system_idxs_2d
            ],
            axis=0)

        bond_orders_to_update = tf.concat(
            [
                bond_orders_to_update,
                tf.ones(
                    (tf.cast(system_total_bond_count, tf.int32), ),
                    dtype=tf.float32) \
                    * system_mean_bond_order
            ],
            axis=0)

        return idx + 1, bond_idxs_to_update, bond_orders_to_update

    _, bond_idxs_to_update, bond_orders_to_update = tf.while_loop(
        # condition
        lambda idx, _1, _2: tf.less(
            idx,
            tf.shape(conjugate_systems)[0]),

        # loop body
        loop_body,

        # var
        [idx, bond_idxs_to_update, bond_orders_to_update],

        shape_invariants = [
            idx.get_shape(),
            tf.TensorShape([None, 2]),
            tf.TensorShape([None, ])])

    # get rid of the placeholder
    bond_idxs_to_update = bond_idxs_to_update[1:]
    bond_orders_to_update = bond_orders_to_update[1:]

    # scatter update
    adjacency_map = tf.tensor_scatter_nd_update(
        adjacency_map,
        bond_idxs_to_update,
        bond_orders_to_update)

    return adjacency_map
