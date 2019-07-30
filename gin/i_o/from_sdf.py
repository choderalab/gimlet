"""
MIT License

Copyright (c) 2019 Chodera lab // Memorial Sloan Kettering Cancer Center,
Weill Cornell Medical College, Nicea Research, and Authors

Authors:
Yuanqing Wang

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

# =============================================================================
# imports
# =============================================================================
import tensorflow as tf
# tf.enable_eager_execution()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
N_CPUS = multiprocessing.cpu_count()

# =============================================================================
# imports
# =============================================================================
TRANSLATION = {
    b'C': 0,
    b'N': 1,
    b'O': 2,
    b'S': 3,
    b'P': 4,
    b'F': 5,
    b'Cl': 6,
    b'Br': 7,
    b'I': 8,
    b'H': 9
}

# =============================================================================
# utility functions
# =============================================================================
def to_ds(file_path, has_charge=False):
    """

    Organic atoms:
    [C, N, O, S, P, F, Cl, Br, I, H]

    Corresponding indices:
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    # read the file
    text = tf.io.read_file(
        file_path)

    lines = tf.strings.split(
        tf.expand_dims(text, 0),
        '\n').values

    lines = tf.boolean_mask(
        lines,

        tf.logical_not(
            tf.strings.regex_full_match(
                lines,
                '.*RAD.*')))

    # get the starts and the ends
    starts = tf.strings.regex_full_match(
        lines,
        '.*V2000.*')

    ends = tf.strings.regex_full_match(
        lines,
        '.*END.*')

    starts_idxs = tf.boolean_mask(
        tf.range(
            tf.cast(
                starts.shape[0],
                tf.int64),
            dtype=tf.int64),
        starts)

    ends_idxs = tf.boolean_mask(
        tf.range(
            tf.cast(
                ends.shape[0],
                tf.int64),
            dtype=tf.int64),
        ends) - tf.constant(1, dtype=tf.int64)

    mol_chunks = tf.concat(
        [
            tf.expand_dims(
                starts_idxs,
                1),
            tf.expand_dims(
                ends_idxs,
                1)
        ],
        axis=1)

    def read_one_mol(idx, has_charge=has_charge, lines=lines):
        mol_chunk = mol_chunks[idx]
        start = mol_chunk[0]
        end = mol_chunk[1]

        # process the head line to get the number of the atoms and bonds
        head = lines[start]
        head = tf.strings.split(tf.expand_dims(head, 0), ' ').values
        head = tf.boolean_mask(
            head,
            tf.logical_not(
                tf.equal(
                    head, '')))

        n_atoms = tf.strings.to_number(
            head[0],
            tf.int64)

        n_bonds = tf.strings.to_number(
            head[1],
            tf.int64)

        # get the lines for atoms and bonds
        atoms_lines = tf.slice(
            lines,
            tf.expand_dims(start+1, 0),
            tf.expand_dims(n_atoms, 0))

        bonds_lines = tf.slice(
            lines,
            tf.expand_dims(start+n_atoms+1, 0),
            tf.expand_dims(n_bonds, 0))

        charges = tf.zeros((n_atoms, ), dtype=tf.float32)

        if has_charge:
            # check if there is any total charge annotated in the system
            has_total_charge = tf.reduce_any(
                tf.strings.regex_full_match(
                    tf.slice(
                        lines,
                        tf.expand_dims(start+1, 0),
                        tf.expand_dims(end-start, 0)),
                    '.*CHG.*'))

            if has_total_charge:
                charge_lines = tf.slice(
                    lines,
                    tf.expand_dims(start+n_atoms+n_bonds+2, 0),
                    tf.expand_dims(2*n_atoms, 0))

            else:
                charge_lines = tf.slice(
                    lines,
                    tf.expand_dims(start+n_atoms+n_bonds+1, 0),
                    tf.expand_dims(2*n_atoms, 0))

            charge_lines = tf.gather(
                charge_lines,
                tf.add(
                    tf.math.multiply(
                        tf.constant(2, dtype=tf.int64),
                        tf.range(
                            n_atoms,
                            dtype=tf.int64)),
                    tf.constant(1, dtype=tf.int64)))

            charges = tf.strings.to_number(
                tf.reshape(
                    charge_lines,
                    [-1]),
                tf.float32)

        # process atom lines
        atoms_lines = tf.strings.split(atoms_lines, ' ').values
        atoms_lines = tf.reshape(
            tf.boolean_mask(
                atoms_lines,
                tf.logical_not(
                    tf.equal(
                        atoms_lines,
                        ''))),
            [n_atoms, -1])

        coordinates = tf.strings.to_number(
            atoms_lines[:, :3],
            tf.float32)


        atoms = tf.cast(
            tf.map_fn(
                lambda x: TRANSLATION[x.numpy()],
                atoms_lines[:, 3],
                tf.int32),
            tf.int64)

        # process bond lines
        bonds_lines = tf.strings.split(bonds_lines, ' ').values
        bonds_lines = tf.reshape(
            tf.boolean_mask(
                bonds_lines,
                tf.logical_not(
                    tf.equal(
                        bonds_lines,
                        ''))),
            [n_bonds, -1])

        bond_idxs = tf.strings.to_number(
            bonds_lines[:, :2],
            tf.int64) - 1

        bond_orders = tf.strings.to_number(
            bonds_lines[:, 2],
            tf.float32)

        adjacency_map = tf.zeros(
            shape=(n_atoms, n_atoms),
            dtype=tf.float32)

        adjacency_map = tf.tensor_scatter_nd_update(
            adjacency_map,
            bond_idxs,
            bond_orders)

        # adjacency_map = adjacency_map.read_value()

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

        return atoms, adjacency_map, coordinates, charges

    ds = tf.data.Dataset.from_tensor_slices(
        tf.range(
            tf.cast(
                mol_chunks.shape[0],
                tf.int64),
            dtype=tf.int64))

    ds = ds.map(lambda x: tf.py_function(
        read_one_mol,
        [x],
        [tf.int64, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=4*N_CPUS)

    return ds
