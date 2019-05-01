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

# =============================================================================
# utility classes
# =============================================================================
class MoleculeGraphsTrainable(tf.keras.Model):
    """ A group of molecules with trainable features.


    """

    def __init__(
            self,
            molecules,
            phi_e,
            rho_e_v,
            phi_v,
            rho_e_u,
            rho_v_u,
            phi_u):
        super(MoleculeGraphsTrainable, self).__init__()
        self.molecules = molecules
        self.molecule_y = molecule_y
        self.atoms_y = atomic_y
        self.phi_e = phi_e
        self.rho_e_v = rho_e_v
        self.phi_v = phi_v
        self.rho_e_u = rho_e_u
        self.phi_u = phi_u

    def propagate(self):
        """ Propagate 
        """
