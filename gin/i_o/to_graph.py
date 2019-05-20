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
# utility functions
# =============================================================================
def get_unweighted_adjacency_matrix(molecule, laplacian=False):
    """ Calculate unweighted adjacency matrix from the list of atoms and
    bonds.

    For a graph,
    $\mathcal{G} = (\mathcal{V}, \mathcal{\Epsilon}, \mathcal{A}),$

    the unweighted adjacency matrix is:

    $$
    A_{ij} = \begin{cases}
        1, e_{ij} \in \mathcal{Epsilon}\\
        0, \mathtt{otherwise}
    $$

    Parameters
    ----------
    molecule : object
        the Molecule object

    Returns
    -------
    adjacency_matrix : tf.Tensor, shape = (n_atoms, n_atoms)
        the adjacency matrix where the values are one where there is a bond,
        and zero where there isn't.

        if `laplacian` is set to `True`, return the transformed matrix where
        $$
        L = D - A
        $$
    """
    # initialize and query
    n_atoms = molecule.atoms.shape[0]
    n_bonds = molecule.bonds.shape[0]
    adjacency_matrix = tf.Variable(
        tf.zeros((n_atoms, n_atoms), tf.int32))

    # for loop
    idx = tf.constant(0)
    def loop_body(idx):
        # read the bond
        bond = molecule.bonds[idx]

        # put 1 at two positions
        adjacency_matrix[bond[0], bond[1]].assign(
            tf.constant(1, dtype=tf.int32))
        adjacency_matrix[bond[1], bond[0]].assign(
            tf.constant(1, dtype=tf.int32))

        return idx + 1

    tf.while_loop(
        tf.less(idx, n_bonds), # condition
        loop_body, # body
        [idx]) # var

    # Laplacian transformation
    if laplacian == True:
        d = tf.diag(
            tf.reduce_sum(adjacency_matrix, axis=0))
        adjacency_matrix = d - adjacency_matrix

    return adjacency_matrix
