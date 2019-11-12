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

# =============================================================================
# utility functions
# =============================================================================
class TypingBase(object):
    """ Wrapper class for atom typing in General Amber Force Field.

    Organic atoms:
    [C, N, O, S, P, F, Cl, Br, I, H]

    Corresponding indices:
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Methods
    -------
    is_sp2 :
        determine whether an atom is sp2.

    is_{type} :
        whether the idx'th atom in the molecule is of that type or not

    """
    def __init__(self, mol):
        # get the adjacency_map
        adjacency_map = mol[1]

        # use the full adjacency map for atom typing
        self.adjacency_map_full = adjacency_map \
            + tf.transpose(adjacency_map)

        self.atoms = mol[0]

        # self.n_atoms = int(self.atoms.shape[0])

    def _is_carbon(self):
        return tf.equal(
            self.atoms,
            tf.constant(0, dtype=tf.int64))

    @property
    def is_carbon(self):
        if not hasattr(self, '__is_carbon'):
            self.__is_carbon = self._is_carbon()

        return self.__is_carbon

    def _is_nitrogen(self):
        return tf.equal(
            self.atoms,
            tf.constant(1, dtype=tf.int64))

    @property
    def is_nitrogen(self):
        if not hasattr(self, '__is_nitrogen'):
            self.__is_nitrogen = self._is_nitrogen()

        return self. __is_nitrogen

    def _is_oxygen(self):
        return tf.equal(
            self.atoms,
            tf.constant(2, dtype=tf.int64))

    @property
    def is_oxygen(self):
        if not hasattr(self, '__is_oxygen'):
            self.__is_oxygen = self._is_oxygen()

        return self. __is_oxygen

    def _is_sulfur(self):
        return tf.equal(
            self.atoms,
            tf.constant(3, dtype=tf.int64))

    @property
    def is_sulfur(self):
        if not hasattr(self, '__is_sulfur'):
            self.__is_sulfur = self._is_sulfur()

        return self.__is_sulfur

    def _is_phosphorus(self):
        return tf.equal(
            self.atoms,
            tf.constant(4, dtype=tf.int64))

    @property
    def is_phosphorus(self):
        if not hasattr(self, '__is_phosphorus'):
            self.__is_phosphorus = self._is_phosphorus()

        return self.__is_phosphorus

    def _is_fluorine(self):
        return tf.equal(
            self.atoms,
            tf.constant(5, dtype=tf.int64))

    @property
    def is_fluorine(self):
        if not hasattr(self, '__is_fluorine'):
            self.__is_fluorine = self._is_fluorine()

        return self.__is_fluorine

    def _is_chlorine(self):
        return tf.equal(
            self.atoms,
            tf.constant(6, dtype=tf.int64))

    @property
    def is_chlorine(self):
        if not hasattr(self, '__is_chlorine'):
            self.__is_chlorine = self._is_chlorine()

        return self.__is_chlorine

    def _is_bromine(self):
        return tf.equal(
            self.atoms,
            tf.constant(7, dtype=tf.int64))

    @property
    def is_bromine(self):
        if not hasattr(self, '__is_chlorine'):
            self.__is_bromine = self._is_bromine()

        return self.__is_bromine

    def _is_iodine(self):
        return tf.equal(
            self.atoms,
            tf.constant(8, dtype=tf.int64))
    @property
    def is_iodine(self):
        if not hasattr(self, '__is_iodine'):
            self.__is_iodine = self._is_iodine()

        return self.__is_iodine

    def _is_hydrogen(self):
        return tf.equal(
            self.atoms,
            tf.constant(9, dtype=tf.int64)
        )

    @property
    def is_hydrogen(self):
        if not hasattr(self, '__is_hydrogen'):
            self.__is_hydrogen = self._is_hydrogen()

        return self.__is_hydrogen

    def _is_heavy(self):
        return tf.logical_not(
            tf.equal(
                self.atoms,
                tf.constant(9, dtype=tf.int64)))

    @property
    def is_heavy(self):
        if not hasattr(self, '__is_heavy'):
            self.__is_heavy = self._is_heavy()

        return self.__is_heavy

    def _is_sp1(self):
        return tf.logical_or(
            tf.reduce_any(
                tf.greater(
                    self.adjacency_map_full,
                    tf.constant(2, dtype=tf.float32)),
                axis=0),
            tf.greater_equal(
                tf.math.count_nonzero(
                    tf.greater_equal(
                        self.adjacency_map_full,
                        tf.constant(2, dtype=tf.float32)),
                    axis=0),
                tf.constant(2, dtype=tf.int64)))

    @property
    def is_sp1(self):
        if not hasattr(self, '__is_sp1'):
            self.__is_sp1 = self._is_sp1()

        return self.__is_sp1

    def _is_sp2(self):
        return tf.logical_and(
            tf.reduce_any(
                    tf.logical_and(
                        tf.greater(
                            self.adjacency_map_full,
                            tf.constant(1, dtype=tf.float32)),
                        tf.less(
                            self.adjacency_map_full,
                            tf.constant(3, dtype=tf.float32)
                        )),
                    axis=1),
            tf.logical_not(
                self.is_sp1))

    @property
    def is_sp2(self):
        if not hasattr(self, '__is_sp2'):
            self.__is_sp2 = self._is_sp2()

        return self.__is_sp2

    def _is_sp3(self):
        return tf.reduce_all(
                tf.less_equal(
                    self.adjacency_map_full,
                    tf.constant(1, dtype=tf.float32)),
                axis=0)
    @property
    def is_sp3(self):
        if not hasattr(self, '__is_sp3'):
            self.__is_sp3 = self._is_sp3()

        return self.__is_sp3

    def _is_connected_to_oxygen(self):
        is_oxygen = self.is_oxygen
        oxygen_connection_idxs = tf.boolean_mask(
            self.adjacency_map_full,
            is_oxygen)

        return tf.reduce_any(
            tf.greater(
                oxygen_connection_idxs,
                tf.constant(0, dtype=tf.float32)),
            axis=0)

    @property
    def is_connected_to_oxygen(self):
        if not hasattr(self, '__is_connected_to_oxygen'):
            self.__is_connected_to_oxygen = self._is_connected_to_oxygen()

        return self.__is_connected_to_oxygen

    def _is_connected_to_sulfur(self):
        is_sulfur = self.is_sulfur
        sulfur_connection_idxs = tf.boolean_mask(
            self.adjacency_map_full,
            is_sulfur)
        return tf.reduce_any(
            tf.greater(
                sulfur_connection_idxs,
                tf.constant(0, dtype=tf.float32)),
            axis=0)

    @property
    def is_connected_to_sulfur(self):
        if not hasattr(self, '__is_connected_to_sulfur'):
            self.__is_connected_to_sulfur = self._is_connected_to_sulfur()

        return self.__is_connected_to_sulfur

    def _is_connected_to_carbon(self):
        carbon_connection_idxs = tf.boolean_mask(
            self.adjacency_map_full,
            self.is_carbon)
        return tf.reduce_any(
            tf.greater(
                carbon_connection_idxs,
                tf.constant(0, dtype=tf.float32)),
            axis=0)

    @property
    def is_connected_to_carbon(self):
        if not hasattr(self, '__is_connected_to_carbon'):
            self.__is_connected_to_carbon \
                = self._is_connected_to_carbon()

        return self.__is_connected_to_carbon

    def _is_connected_to_sp1_carbon(self):
        is_sp1_carbon = tf.logical_and(
            self.is_carbon,
            self.is_sp1)
        sp1_carbon_connection_idxs = tf.boolean_mask(
            self.adjacency_map_full,
            is_sp1_carbon)
        return tf.reduce_any(
            tf.greater(
                sp1_carbon_connection_idxs,
                tf.constant(0, dtype=tf.float32)),
            axis=0)

    @property
    def is_connected_to_sp1_carbon(self):
        if not hasattr(self, '__is_connected_to_sp1_carbon'):
            self.__is_connected_to_sp1_carbon \
                = self._is_connected_to_sp1_carbon()

        return self.__is_connected_to_sp1_carbon

    def _is_connected_to_sp2_carbon(self):
        is_sp2_carbon = tf.logical_and(
            self.is_carbon,
            self.is_sp2)
        sp2_carbon_connection_idxs = tf.boolean_mask(
            self.adjacency_map_full,
            is_sp2_carbon)
        return tf.reduce_any(
            tf.greater(
                sp2_carbon_connection_idxs,
                tf.constant(0, dtype=tf.float32)),
            axis=0)

    @property
    def is_connected_to_sp2_carbon(self):
        if not hasattr(self, '__is_connected_to_sp2_carbon'):
            self.__is_connected_to_sp2_carbon \
                = self._is_connected_to_sp2_carbon()

        return self.__is_connected_to_sp2_carbon

    def _is_connected_to_sp3_carbon(self):
        is_sp3_carbon = tf.logical_and(
            self.is_carbon,
            self.is_sp3)
        sp3_carbon_connection_idxs = tf.boolean_mask(
            self.adjacency_map_full,
            is_sp3_carbon)
        return tf.reduce_any(
            tf.greater(
                sp3_carbon_connection_idxs,
                tf.constant(0, dtype=tf.float32)),
            axis=0)

    @property
    def is_connected_to_sp3_carbon(self):
        if not hasattr(self, '__is_connected_to_sp3_carbon'):
            self.__is_connected_to_sp3_carbon \
                = self._is_connected_to_sp3_carbon()

        return self.__is_connected_to_sp3_carbon

    def _is_connected_to_nitrogen(self):
        nitrogen_connection_idxs = tf.boolean_mask(
            self.adjacency_map_full,
            self.is_nitrogen)
        return tf.reduce_any(
            tf.greater(
                nitrogen_connection_idxs,
                tf.constant(0, dtype=tf.float32)),
            axis=0)

    @property
    def is_connected_to_nitrogen(self):
        if not hasattr(self, '__is_connected_to_nitrogen'):
            self.__is_connected_to_nitrogen \
                = self._is_connected_to_nitrogen()

        return self.__is_connected_to_nitrogen

    def _is_connected_to_phosphorus(self):
        phosphorus_connection_idxs = tf.boolean_mask(
            self.adjacency_map_full,
            self.is_phosphorus)
        return tf.reduce_any(
            tf.greater(
                phosphorus_connection_idxs,
                tf.constant(0, dtype=tf.float32)),
            axis=0)

    @property
    def is_connected_to_phosphorus(self):
        if not hasattr(self, '__is_connected_to_phosphorus'):
            self.__is_connected_to_phosphorus \
                = self._is_connected_to_phosphorus()

        return self.__is_connected_to_phosphorus

    def _has_1_hydrogen(self):
        # calculate the number of hydrogens added to heavy atoms
        # the heavy atoms with one hydrogen
        return tf.reduce_any(
            tf.reshape(
                tf.concat(
                    [
                        # sp3 carbon
                        tf.logical_and(
                            self.is_carbon,
                            tf.logical_and(
                                self.is_sp3,
                                self.is_connected_to_3_heavy)),

                        # sp2 carbon
                        tf.logical_and(
                            self.is_carbon,
                            tf.logical_and(
                                self.is_sp2,
                                self.is_connected_to_2_heavy)),

                        # sp1 carbon
                        tf.logical_and(
                            self.is_carbon,
                            tf.logical_and(
                                self.is_sp1,
                                self.is_connected_to_1_heavy)),

                        # sp3 nitrogen or phosphorus
                        tf.logical_and(
                            tf.logical_or(
                                self.is_nitrogen,
                                self.is_phosphorus),
                            tf.logical_and(
                                self.is_sp3,
                                self.is_connected_to_2_heavy)),

                        # sp2 nitrogen or phosphorus
                        tf.logical_and(
                            tf.logical_or(
                                self.is_nitrogen,
                                self.is_phosphorus),
                            tf.logical_and(
                                self.is_sp2,
                                self.is_connected_to_1_heavy)),

                        # sp3 oxygen or sulfur
                        tf.logical_and(
                            tf.logical_or(
                                self.is_oxygen,
                                self.is_sulfur),
                            self.is_sp3)

                    ],
                    axis=0),
                [-1, tf.shape(self.atoms, tf.int64)[0]]),
            axis=0)

    @property
    def has_1_hydrogen(self):
        if not hasattr(self, '__has_1_hydrogen'):
            self.__has_1_hydrogen \
                = self._has_1_hydrogen()

        return self.__has_1_hydrogen

    def _has_2_hydrogen(self):
        return tf.reduce_any(
            tf.reshape(
                tf.concat(
                    [
                        # sp3 carbon
                        tf.logical_and(
                            self.is_carbon,
                            tf.logical_and(
                                self.is_sp3,
                                self.is_connected_to_2_heavy)),

                        # sp2 carbon
                        tf.logical_and(
                            self.is_carbon,
                            tf.logical_and(
                                self.is_sp2,
                                self.is_connected_to_1_heavy)),

                        # sp3 nitrogen or phosphorus
                        tf.logical_and(
                            tf.logical_or(
                                self.is_nitrogen,
                                self.is_phosphorus),
                            tf.logical_and(
                                self.is_sp3,
                                self.is_connected_to_1_heavy))

                    ],
                    axis=0),
                [-1, tf.shape(self.atoms)[0]]),
            axis=0)

    @property
    def has_2_hydrogen(self):
        if not hasattr(self, '__has_2_hydrogen'):
            self.__has_2_hydrogen \
                = self._has_2_hydrogen()

        return self.__has_2_hydrogen

    def _has_3_hydrogen(self):
        return tf.logical_and(
            self.is_carbon,
            tf.logical_and(
                self.is_sp3,
                self.is_connected_to_1_heavy))

    @property
    def has_3_hydrogen(self):
        if not hasattr(self, '__has_3_hydrogen'):
            self.__has_3_hydrogen \
                = self._has_3_hydrogen()

        return self.__has_3_hydrogen

    def _is_connected_to_1_heavy(self):
        return tf.equal(
            tf.math.count_nonzero(
                tf.boolean_mask(
                    self.adjacency_map_full,
                    self.is_heavy,
                    axis=1),
                axis=1),
            tf.constant(1, dtype=tf.int64))

    @property
    def is_connected_to_1_heavy(self):
        if not hasattr(self, '__is_connected_to_1_heavy'):
            self.__is_connected_to_1_heavy \
                = self._is_connected_to_1_heavy()

        return self.__is_connected_to_1_heavy

    def _is_connected_to_1(self):
        return tf.equal(
            tf.math.count_nonzero(
                self.adjacency_map_full,
                axis=0),
            tf.constant(1, dtype=tf.int64))

    @property
    def is_connected_to_1(self):
        if not hasattr(self, '__is_connected_to_1'):
            self.__is_connected_to_1 \
                = self._is_connected_to_1()

        return self.__is_connected_to_1

    def _is_connected_to_2_heavy(self):
        return tf.equal(
            tf.math.count_nonzero(
                tf.boolean_mask(
                    self.adjacency_map_full,
                    self.is_heavy,
                    axis=1),
                axis=1),
            tf.constant(2, dtype=tf.int64))

    @property
    def is_connected_to_2_heavy(self):
        if not hasattr(self, '__is_connected_to_2_heavy'):
            self.__is_connected_to_2_heavy \
                = self._is_connected_to_2_heavy()

        return self.__is_connected_to_2_heavy

    def _is_connected_to_2(self):
        return tf.equal(
            tf.math.count_nonzero(
                self.adjacency_map_full,
                axis=0),
            tf.constant(2, dtype=tf.int64))

    @property
    def is_connected_to_2(self):
        if not hasattr(self, '__is_connected_to_2'):
            self.__is_connected_to_2 \
                = self._is_connected_to_2()

        return self.__is_connected_to_2

    def _is_connected_to_3_heavy(self):
        return tf.equal(
            tf.math.count_nonzero(
                tf.boolean_mask(
                    self.adjacency_map_full,
                    self.is_heavy,
                    axis=1),
                axis=1),
            tf.constant(3, dtype=tf.int64))

    @property
    def is_connected_to_3_heavy(self):
        if not hasattr(self, '__is_connected_to_3_heavy'):
            self.__is_connected_to_3_heavy \
                = self._is_connected_to_3_heavy()

        return self.__is_connected_to_3_heavy

    def _is_connected_to_3(self):
        return tf.equal(
            tf.math.count_nonzero(
                self.adjacency_map_full,
                axis=0),
            tf.constant(3, dtype=tf.int64))

    @property
    def is_connected_to_3(self):
        if not hasattr(self, '__is_connected_to_3'):
            self.__is_connected_to_3 \
                = self._is_connected_to_3()

        return self.__is_connected_to_3

    def _is_connected_to_4_heavy(self):
        return tf.equal(
            tf.math.count_nonzero(
                tf.boolean_mask(
                    self.adjacency_map_full,
                    self.is_heavy,
                    axis=1),
                axis=1),
            tf.constant(4, dtype=tf.int64))

    @property
    def is_connected_to_4_heavy(self):
        if not hasattr(self, '__is_connected_to_4_heavy'):
            self.__is_connected_to_4_heavy \
                = self._is_connected_to_4_heavy()

        return self.__is_connected_to_4_heavy

    def _is_connected_to_4(self):
        return tf.equal(
            tf.math.count_nonzero(
                self.adjacency_map_full,
                axis=0),
            tf.constant(4, dtype=tf.int64))

    @property
    def is_connected_to_4(self):
        if not hasattr(self, '__is_connected_to_4'):
            self.__is_connected_to_4 \
                = self._is_connected_to_4()

        return self.__is_connected_to_4

    # @tf.function
    def _is_in_ring(self):
        """ Determine whether an atom in a molecule is in a ring or not.
        """
        # TODO: currently this is kinda slow since we do this for each
        #       atom. Think about another way to do this!

        # inner loop body
        def loop_body_inner(
                root,
                visited,
                parents,
                queue,
                is_in_ring_,
                ring_detected,
                adjacency_map_full=self.adjacency_map_full):

            # dequeue
            idx = queue[-1]
            queue = queue[:-1]

            # flag the position of self
            is_self = tf.equal(
                tf.range(
                    tf.cast(
                        self.adjacency_map_full.shape[0],
                        tf.int64),
                    dtype=tf.int64),
                idx)

            is_root = tf.equal(
                tf.range(
                    tf.cast(
                        self.adjacency_map_full.shape[0],
                        tf.int64),
                    dtype=tf.int64),
                root)

            # flag the neighbors
            is_neighbors = tf.greater(
                self.adjacency_map_full[idx, :],
                tf.constant(0, dtype=tf.float32))

            # check whether root is in the neighbors
            root_is_neighbor = is_neighbors[root]

            # check whether the parent of self is root
            parent_is_root = tf.equal(
                root,
                parents[idx])

            # check whether a ring is detected
            ring_detected = tf.logical_and(
                root_is_neighbor,
                tf.logical_not(
                    parent_is_root))

            # put self and root in the ring
            is_in_ring_ = tf.cond(
                ring_detected,

                # if finished:
                lambda: tf.where(
                    tf.logical_or(
                        is_self,
                        is_root),

                    tf.tile(
                        tf.expand_dims(
                            tf.constant(True),
                            0),

                        tf.expand_dims(
                            tf.shape(is_in_ring_)[0],
                            0)),

                    is_in_ring_),

                lambda: is_in_ring_)

            # flag the neighbors that are not visited
            is_unvisited_neighbors = tf.logical_and(
                is_neighbors,
                tf.logical_not(
                    visited))

            # get the states of the neighbors
            neighbors_unvisited = tf.boolean_mask(
                tf.range(
                    tf.cast(
                        self.adjacency_map_full.shape[0],
                        tf.int64),
                    dtype=tf.int64),
                is_unvisited_neighbors)

            # enqueue
            queue = tf.concat(
                [
                    queue,
                    neighbors_unvisited
                ],
                axis=0)

            # put self as the parent for the neighbors that are not visited
            parents = tf.where(
                is_neighbors,

                tf.ones_like(parents) * idx,

                parents)

            visited = tf.where(
                tf.logical_or(
                    is_self,
                    is_unvisited_neighbors),

                    tf.tile(
                        tf.expand_dims(
                            tf.constant(True),
                            0),

                        tf.expand_dims(
                            tf.shape(is_unvisited_neighbors)[0],
                            0)),

                    visited)

            return root, visited, parents, queue, is_in_ring_, ring_detected

        # define outer loop body
        def loop_body_outer(
                root,
                is_in_ring_,
                adjacency_map_full=self.adjacency_map_full):

            # init visited flags as all False
            visited = tf.tile(
                tf.expand_dims(
                    False,
                    0),
                tf.expand_dims(
                    self.adjacency_map_full.shape[0],
                    0))

            # put the root in the queue
            queue = tf.expand_dims(
                root,
                0)

            # init parents
            parents = tf.ones(
                (self.adjacency_map_full.shape[0]),
                dtype=tf.int64) \
                * -1

            ring_detected = tf.constant(False)

            # execute inner loop
            root, visited, parents, queue, is_in_ring_, ring_detected\
                = tf.while_loop(
                    # condition
                    lambda root, visited, parents, queue,
                        is_in_ring_, ring_detected:\
                            tf.greater(
                                tf.shape(queue)[0],
                                0),

                    # loop body
                    loop_body_inner,

                    # loop var
                    [root, visited, parents, queue, is_in_ring_, ring_detected],

                    shape_invariants=[
                        root.get_shape(),
                        visited.get_shape(),
                        parents.get_shape(),
                        tf.TensorShape([None, ]),
                        is_in_ring_.get_shape(),
                        ring_detected.get_shape()
                    ])

            # increment
            return root + 1, is_in_ring_

        # execute the outer loop
        root = tf.constant(0, dtype=tf.int64)
        is_in_ring_ = tf.tile(
            tf.expand_dims(
                tf.constant(False),
                0),

            tf.expand_dims(
                tf.shape(self.adjacency_map_full)[0],
                0))

        _, is_in_ring_ = tf.while_loop(
            # condition
            lambda root, _: tf.less(
                root,
                tf.cast(
                    self.adjacency_map_full.shape[0],
                    tf.int64)),

            # body
            loop_body_outer,

            # var
            [root, is_in_ring_],

            shape_invariants=[
                root.get_shape(),
                is_in_ring_.get_shape()],

            )# parallel_iterations=self.n_atoms)

        return is_in_ring_

    @property
    def is_in_ring(self):
        if not hasattr(self, '__is_in_ring'):
            self.__is_in_ring = self._is_in_ring()

        return self.__is_in_ring

    # @tf.function
    def _is_in_conjugate_system(self):
        """ Determine whether an atom in a molecule is in a conjugated
        system or not.

        """
        # TODO: when compiling this into a graph, it throws error
        sp2_idxs = tf.boolean_mask(
            tf.range(
                tf.cast(
                    tf.shape(self.adjacency_map_full)[0],
                    tf.int64),
                dtype=tf.int64),
            self.is_sp2)

        # gather only sp2 atoms
        sp2_adjacency_map = tf.gather(
            tf.gather(
                self.adjacency_map_full,
                sp2_idxs,
                axis=0),
            sp2_idxs,
            axis=1)

        self.sp2_adjacency_map = sp2_adjacency_map

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
                        tf.shape(self.sp2_adjacency_map)[0],
                        tf.int64),
                    dtype=tf.int64),
                idx)

            # flag the neighbors
            is_neighbors = tf.greater(
                self.sp2_adjacency_map[idx, :],
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
                        tf.shape(self.sp2_adjacency_map)[0],
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
                            tf.shape(self.sp2_adjacency_map)[0],
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
            lambda conjugate_systems, visited: loop_body_outer(
                conjugate_systems, visited,
                sp2_adjacency_map=sp2_adjacency_map),

            # loop var
            [conjugate_systems, visited],

            shape_invariants=[
                tf.TensorShape([None, sp2_idxs.shape[0]]),
                tf.TensorShape((visited.get_shape()))])


        # get all the indices
        idxs_in_conjugate_systems = tf.cast(
            tf.unique(
                tf.reshape(
                    conjugate_systems,
                    [-1]))[0],
            tf.int64)

        idxs_in_conjugate_systems = tf.boolean_mask(
            idxs_in_conjugate_systems,
            tf.logical_not(
                tf.equal(
                    idxs_in_conjugate_systems,
                    tf.constant(-1, dtype=tf.int64))),)

        idxs_in_conjugate_systems = tf.gather(
            sp2_idxs,
            idxs_in_conjugate_systems)

        # get the flags of whether an atom is in conjugate system
        is_in_conjugate_system_ = tf.reduce_any(
            tf.equal(
                tf.range(
                    tf.cast(
                        tf.shape(self.adjacency_map_full)[0],
                        tf.int64),
                    dtype=tf.int64),

                tf.tile(
                    tf.expand_dims(
                        idxs_in_conjugate_systems,
                        1),
                    [
                        1,
                        tf.shape(self.adjacency_map_full)[0]
                    ])),
            axis=0)

        return is_in_conjugate_system_

    @property
    def is_in_conjugate_system(self):
        if not hasattr(self, '__is_in_conjugate_system'):
            self.__is_in_conjugate_system = self._is_in_conjugate_system()

        return self.__is_in_conjugate_system

    # @tf.function
    def _is_aromatic(self):
        """ Determine whether the atoms are in an aromatic system.

        Now we achieve this by searching through the ring systems in
        conjugate systems.
        """
        # TODO: not sure whether this defines aromatic system well

        # get the flags of whether atoms are in conjugate systems
        is_in_conjugate_system_ = self.is_in_conjugate_system

        conjugate_system_idxs = tf.boolean_mask(
            tf.range(
                tf.shape(is_in_conjugate_system_, tf.int64)[0],
                dtype=tf.int64),
            is_in_conjugate_system_)

        # get the adjacency map for the conjugate system only
        adjacency_map_conjugate_system = tf.boolean_mask(
            tf.boolean_mask(
                self.adjacency_map_full,
                is_in_conjugate_system_,
                axis=0),
            is_in_conjugate_system_,
            axis=1)

        self.adjacency_map_conjugate_system = adjacency_map_conjugate_system

        # TODO: currently this is kinda slow since we do this for each
        #       atom. Think about another way to do this!

        # inner loop body
        def loop_body_inner(
                root,
                visited,
                parents,
                queue,
                is_in_ring_,
                ring_detected,
                adjacency_map_conjugate_system=adjacency_map_conjugate_system):

            # dequeue
            idx = queue[-1]
            queue = queue[:-1]

            # flag the position of self
            is_self = tf.equal(
                tf.range(
                    tf.cast(
                        tf.shape(self.adjacency_map_conjugate_system)[0],
                        tf.int64),
                    dtype=tf.int64),
                idx)

            is_root = tf.equal(
                tf.range(
                    tf.cast(
                        tf.shape(self.adjacency_map_conjugate_system)[0],
                        tf.int64),
                    dtype=tf.int64),
                root)

            # flag the neighbors
            is_neighbors = tf.greater(
                self.adjacency_map_conjugate_system[idx, :],
                tf.constant(0, dtype=tf.float32))

            # check whether root is in the neighbors
            root_is_neighbor = is_neighbors[root]

            # check whether the parent of self is root
            parent_is_root = tf.equal(
                root,
                parents[idx])

            # check whether a ring is detected
            ring_detected = tf.logical_and(
                root_is_neighbor,
                tf.logical_not(
                    parent_is_root))

            # put self and root in the ring
            is_in_ring_ = tf.cond(
                ring_detected,

                # if finished:
                lambda: tf.where(
                    tf.logical_or(
                        is_self,
                        is_root),

                    tf.tile(
                        tf.expand_dims(
                            tf.constant(True),
                            0),

                        tf.expand_dims(
                            tf.shape(is_in_ring_)[0],
                            0)),

                    is_in_ring_),

                lambda: is_in_ring_)

            # flag the neighbors that are not visited
            is_unvisited_neighbors = tf.logical_and(
                is_neighbors,
                tf.logical_not(
                    visited))

            # get the states of the neighbors
            neighbors_unvisited = tf.boolean_mask(
                tf.range(
                    tf.cast(
                        tf.shape(self.adjacency_map_conjugate_system)[0],
                        tf.int64),
                    dtype=tf.int64),
                is_unvisited_neighbors)

            # enqueue
            queue = tf.concat(
                [
                    queue,
                    neighbors_unvisited
                ],
                axis=0)

            # put self as the parent for the neighbors that are not visited
            parents = tf.where(
                is_neighbors,

                tf.ones_like(parents) * idx,

                parents)

            visited = tf.where(
                tf.logical_or(
                    is_self,
                    is_unvisited_neighbors),

                    tf.tile(
                        tf.expand_dims(
                            tf.constant(True),
                            0),

                        tf.expand_dims(
                            tf.shape(is_unvisited_neighbors)[0],
                            0)),

                    visited)

            return root, visited, parents, queue, is_in_ring_, ring_detected

        # define outer loop body
        def loop_body_outer(
                root,
                is_in_ring_,
                adjacency_map_conjugate_system=adjacency_map_conjugate_system):

            # init visited flags as all False
            visited = tf.tile(
                tf.expand_dims(
                    False,
                    0),
                tf.expand_dims(
                    tf.shape(self.adjacency_map_conjugate_system)[0],
                    0))

            # put the root in the queue
            queue = tf.expand_dims(
                root,
                0)

            # init parents
            parents = tf.ones(
                (tf.shape(self.adjacency_map_conjugate_system)[0], ),
                dtype=tf.int64) \
                * -1

            ring_detected = tf.constant(False)

            # execute inner loop
            root, visited, parents, queue, is_in_ring_, ring_detected\
                = tf.while_loop(
                    # condition
                    lambda root, visited, parents, queue,
                        is_in_ring_, ring_detected:\
                            tf.greater(
                                tf.shape(queue)[0],
                                0),

                    # loop body
                    loop_body_inner,

                    # loop var
                    [root, visited, parents, queue, is_in_ring_, ring_detected],

                    shape_invariants=[
                        root.get_shape(),
                        visited.get_shape(),
                        parents.get_shape(),
                        tf.TensorShape([None, ]),
                        is_in_ring_.get_shape(),
                        ring_detected.get_shape()
                    ])

            # increment
            return root + 1, is_in_ring_

        # execute the outer loop
        root = tf.constant(0, dtype=tf.int64)
        is_in_ring_ = tf.tile(
            tf.expand_dims(
                tf.constant(False),
                0),

            tf.expand_dims(
                tf.shape(self.adjacency_map_conjugate_system)[0],
                0))

        _, is_in_ring_ = tf.while_loop(
            # condition
            lambda root, _: tf.less(
                root,
                tf.cast(
                    tf.shape(self.adjacency_map_conjugate_system)[0],
                    tf.int64)),

            # body
            loop_body_outer,

            # var
            [root, is_in_ring_],

            shape_invariants=[
                root.get_shape(),
                is_in_ring_.get_shape()],

            ) # parallel_iterations=self.n_atoms)

        # get the aromatic indices
        aromatic_idxs = tf.gather(
            tf.range(
                tf.cast(
                    tf.shape(self.adjacency_map_full)[0],
                    tf.int64),
                dtype=tf.int64),
            tf.boolean_mask(
                tf.range(
                    tf.cast(
                        tf.shape(self.adjacency_map_conjugate_system)[0],
                        tf.int64),
                    dtype=tf.int64),
                is_in_ring_))

        # switch back to the full indexing of the atoms
        aromatic_idxs = tf.gather(
            conjugate_system_idxs,
            aromatic_idxs)

        # turn the indices
        is_aromatic_ = tf.reduce_any(
            tf.equal(
                tf.range(
                    tf.cast(
                        tf.shape(self.adjacency_map_full)[0],
                        tf.int64),
                    dtype=tf.int64),

                tf.tile(
                    tf.expand_dims(
                        aromatic_idxs,
                        1),
                    [
                        1,
                        tf.shape(self.adjacency_map_full)[0]
                    ])),
            axis=0)

        return is_aromatic_

    @property
    def is_aromatic(self):
        if not hasattr(self, '__is_aromatic'):
            self.__is_aromatic = self._is_aromatic()

        return self.__is_aromatic

    def _is_connected_to_aromatic(self):
        return tf.greater(
            tf.math.count_nonzero(
                tf.boolean_mask(
                    self.adjacency_map_full,
                    self.is_aromatic,
                    axis=1),
                axis=1),
            tf.constant(0, dtype=tf.int64))

    @property
    def is_connected_to_aromatic(self):
        if not hasattr(self, '__is_connected_to_aromatic'):
            self.__is_connected_to_aromatic \
                = self._is_connected_to_aromatic()

        return self.__is_connected_to_aromatic

    def get_assignment(self):
        """ Assign the typing in the forcefield in to atoms.

        """
        pass


class Typing(TypingBase):
    def __init__(self, mol):
        super(Typing, self).__init__(mol)

class TypingGAFF(TypingBase):
    """ Produce typing for GAFF.

    Based on "Development and testing of a general amber force field."
    doi: 10.1002/jcc.20035


    """
    def __init__(self, mol):
        super(TypingGAFF, self).__init__(mol)

    def is_1(self):
        """ c
        sp2 carbon in C=O, C=S
        """
        return tf.logical_and(
            self.is_carbon,
            tf.logical_and(
                self.is_sp2,
                tf.logical_or(
                    self.is_connected_to_oxygen,
                    self.is_connected_to_sulfur)))

    def is_2(self):
        """ c1
        sp1 carbon
        """
        return tf.logical_and(
            self.is_carbon,
            self.is_sp1)

    def is_3(self):
        """ c2
        sp2 carbon, aliphatic
        """
        return tf.logical_and(
            self.is_carbon,
            tf.logical_and(
                self.is_sp2,
                tf.logical_not(
                    self.is_aromatic)))

    def is_4(self):
        """ c3
        sp3 carbon
        """
        return tf.logical_and(
            self.is_carbon,
            self.is_sp3)

    def is_5(self):
        """ ca
        sp2 carbon, aromatic
        """
        return tf.logical_and(
            self.is_carbon,
            self.is_aromatic)

    def is_6(self):
        """ n
        sp2 nitrogen in amides
        """
        is_carbonyl_carbon = tf.logical_and(
            self.is_carbon,
            tf.logical_and(
                self.is_sp2,
                self.is_connected_to_oxygen))

        carbonyl_carbon_idxs = tf.boolean_mask(
            tf.range(
                tf.cast(
                    tf.shape(self.adjacency_map_full)[0],
                    tf.int64),
                dtype=tf.int64),
            is_carbonyl_carbon)

        return tf.logical_and(
            self.is_nitrogen,
            tf.reduce_any(
                tf.greater(
                    tf.gather(
                        self.adjacency_map_full,
                        carbonyl_carbon_idxs),
                    tf.constant(0, dtype=tf.float32)),
                axis=0))

    def is_7(self):
        """ n1
        sp1 nitrogen
        """
        return tf.logical_and(
            self.is_nitrogen,
            self.is_sp1)

    def is_8(self):
        """ n2
        sp2 nitrogen with 2 subst., real double bonds
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                    self.is_connected_to_2_heavy,
                    tf.reduce_any(
                        tf.greater(
                            self.adjacency_map_full,
                            tf.constant(1, dtype=tf.float32)),
                        axis=0)))

    def is_9(self):
        """ n3
        sp3 nitrogen with 3 subst.
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_sp3,
                self.is_connected_to_3_heavy))

    def is_10(self):
        """ n4
        sp3 nitrogen with 4 subst.
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_sp3,
                self.is_connected_to_4_heavy))

    def is_11(self):
        """ na
        sp2 nitrogen with 3 subst.
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_sp2,
                self.is_connected_to_3_heavy))

    def is_12(self):
        """ nh
        amine nitrogen connected to aromatic rings
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_connected_to_1_heavy,
                self.is_connected_to_aromatic))

    def is_13(self):
        """ no
        nitrogen in nitro groups
        """
        # TODO: might not be the right way to do this
        return tf.logical_and(
            self.is_nitrogen,
            self.is_connected_to_oxygen)

    def is_14(self):
        """ o
        sp2 oxygen in C=O, COO-
        """
        return tf.logical_and(
            self.is_oxygen,
            tf.logical_and(
                self.is_connected_to_sp2_carbon,
                self.is_sp2))

    def is_15(self):
        """ oh
        sp3 oxygen in hydroxyl groups
        """
        return tf.logical_and(
            tf.logical_and(
                self.is_oxygen,
                self.is_sp3),
            self.is_connected_to_1_heavy)

    def is_16(self):
        """ os
        sp3 oxygen in ethers and esters
        """
        return tf.logical_and(
            tf.logical_and(
                self.is_oxygen,
                self.is_sp3),
            self.is_connected_to_2_heavy)

    def is_17(self):
        """ s2
        sp2 sulfur (p=S, C=S, etc.)
        """
        return tf.logical_and(
            self.is_sulfur,
            self.is_sp2)

    def is_18(self):
        """ sh
        sp3 sulfur in thiol groups
        """
        return tf.logical_and(
            tf.logical_and(
                self.is_sulfur,
                self.is_sp3),
            self.is_connected_to_1_heavy)

    def is_19(self):
        """ ss
        sp3 sulfur in -SR and -SS
        """
        return tf.logical_and(
            tf.logical_and(
                self.is_sulfur,
                self.is_sp3),
            self.is_connected_to_2_heavy)

    def is_20(self):
        """ s4
        hypervalent sulfur, 3 subst.
        """
        return tf.logical_and(
            self.is_sulfur,
            self.is_connected_to_3_heavy)

    def is_21(self):
        """ s4
        hypervalent sulfur, 4 subst.
        """
        return tf.logical_and(
            self.is_sulfur,
            self.is_connected_to_4_heavy)

    def is_22(self):
        """ p2
        sp2 phosphorus (C=P, etc.)
        """
        return tf.logical_and(
            self.is_phosphorus,
            self.is_sp2)

    def is_23(self):
        """ p3
        sp3 phosphorus, 3 subst.
        """
        return tf.logical_and(
            tf.logical_and(
                self.is_phosphorus,
                self.is_sp3),
            self.is_connected_to_3_heavy)

    def is_24(self):
        """ p4
        phosphorus, 3 subst.
        """
        return tf.logical_and(
            tf.logical_and(
                self.is_phosphorus,
                tf.logical_not(
                    self.is_sp3)),
            self.is_connected_to_3_heavy)

    def is_25(self):
        """ p5
        hypervalent phosphorus, 4 subst.
        """
        return tf.logical_and(
            tf.logical_and(
                self.is_phosphorus,
                tf.logical_not(
                    self.is_sp3)),
            self.is_connected_to_4_heavy)

    def is_26(self):
        """ hc
        hydrogen on aliphatic carbon.
        """
        return tf.logical_and(
            self.is_hydrogen,
            tf.logical_and(
                self.is_connected_to_carbon,
                tf.logical_not(
                    self.is_connected_to_aromatic)))

    def is_27(self):
        """ ha
        hydrogen on aromatic carbon
        """
        return tf.logical_and(
            self.is_hydrogen,
            tf.logical_and(
                self.is_connected_to_carbon,
                self.is_connected_to_aromatic))

    def is_28(self):
        """ hn
        hydrogen on nitrogen.
        """
        return tf.logical_and(
            self.is_hydrogen,
            self.is_connected_to_nitrogen)

    def is_29(self):
        """ ho
        hydrogen on oxygen
        """
        return tf.logical_and(
            self.is_hydrogen,
            self.is_connected_to_oxygen)

    def is_30(self):
        """ hs
        hydrogen on sulfur
        """
        return tf.logical_and(
            self.is_hydrogen,
            self.is_connected_to_sulfur)

    def is_31(self):
        """ hp
        hydrogen on phosphorus
        """
        return tf.logical_and(
            self.is_hydrogen,
            self.is_connected_to_phosphorus)

    def is_32(self):
        """ f
        any fluorine
        """
        return self.is_fluorine

    def is_33(self):
        """ cl
        any chlorine
        """
        return self.is_chlorine

    def is_34(self):
        """ br
        any bromine
        """
        return self.is_bromine

    def is_35(self):
        """ i
        any iodine
        """
        return self.is_iodine

    def get_assignment(self):
        fn_array = [getattr(self, 'is_' + str(idx)) for idx in range(1, 36)]

        # use paralleled while loop for this assignment
        idx = tf.constant(1, dtype=tf.int64)
        assignment = tf.constant(-1, dtype=tf.int64) \
            * tf.ones_like(
                self.atoms,
                dtype=tf.int64)

        def loop_body(idx, assignment):
            # get the function
            fn = fn_array[idx - 1]
            is_this_type = fn()
            assignment = tf.where(
                is_this_type,

                # when it is of this type
                idx * tf.ones(
                    (tf.shape(assignment)[0], ),
                    dtype=tf.int64),

                # when it is not
                assignment)

            return idx + 1, assignment

        _, assignment = tf.while_loop(
            lambda idx, _: tf.less(idx, 36),

            loop_body,

            [idx, assignment])

        self.assignment = assignment
        return self.assignment
