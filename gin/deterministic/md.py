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
tf.enable_eager_execution()
import gin
import gin.deterministic.forcefield

# =============================================================================
# utility functions
# =============================================================================
def topological_molecule_to_3d_conformation_distance_geometry(
        mol,
        forcefiled='gaff'):
    """ Generate the 3d conformation of a small molecule based on its
    adjacency map using distance geometry.


    """


# =============================================================================
# module classes
# =============================================================================
class MoleculeMDSystem(gin.molecule.Molecule):
    """
    A single molecule system that could be calculated in MD calculation.
    """
    def __init__(self):
        super(MoleculeMDSystem, self).__init__()

    def get_3d_conformation(
            self,
            forcefield=gin.forcefields.gaff,
            n_conformations=1):
        """ Generate 3D conformation of a small molecule based on
        its topology.

        Parameters
        ----------

        """
        pass
