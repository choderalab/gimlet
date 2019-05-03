"""
MIT License

Copyright (c) 2019 Chodera lab // Memorial Sloan Kettering Cancer Center
and Authors

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
tf.enable_eager_execution

import gin.molecule

class MoleculesTrainable(Object):
    """ A wrapper class for a list of molecules.

    """
    def __init__(self):
        pass

    def from_list_of_smiles(list_of_smiles, batch_size=-1):
        """ Parsing from a list of smiles string.

        Parameters
        ----------
        list_of_smiles : list

        """
        from gin.i_o import from_smiles

        # init a empty dataset
        ds = None

        # decide batch_size
        if batch_size == -1:
            batch_size = len(ds_smiles)

        # calculate the number of batches
        n_batches = int(len(ds_smiles) // batch_size) + 1

        for idx_batch in n_batches: # loop through batches
            # process on batch
            batch = list_of_smiles[idx * batch_size : (idx + 1) * batch_size]

            # init atoms and edges
            atoms_idxs = tf.constant([], dtype=tf.int64)
            atoms_types = tf.constant([], dtype=tf.int64)
            bonds = tf.constant([], dtype=tf.int64)
            globals = tf.constant([], dtype=tf.int64)

            # get the molecules
            molecules = tf.map_fn(
                from_smiles.smiles_to_organic_topological_molecule,
                batch)

            def loop_body(idx, atoms, bonds, molecules=molecules):
                # get that specific molecule
                molecule = molecules[idx]

                #
