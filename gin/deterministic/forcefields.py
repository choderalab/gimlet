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
import xml.etree.ElementTree as ET
import os

# =============================================================================
# CONSTANTS
# =============================================================================

GAFF_TYPING_TRANSLATION_DICT = {
    1 : 'c',
    2 : 'c1',
    3 : 'c2',
    4 : 'c3',
    5 : 'ca',
    6 : 'n',
    7 : 'n1',
    8 : 'n2',
    9 : 'n3',
    10 : 'n4',
    11 : 'na',
    12 : 'nh',
    13 : 'no',
    14 : 'o',
    15 : 'oh',
    16 : 'os',
    17 : 's2',
    18 : 'sh',
    19 : 'ss',
    20 : 's4',
    21 : 's6',
    22 : 'p2',
    23 : 'p3',
    24 : 'p4',
    25 : 'p5',
    26 : 'hc',
    27 : 'ha',
    28 : 'hn',
    29 : 'ho',
    30 : 'hs',
    31 : 'hp',
    32 : 'f',
    33 : 'cl',
    34 : 'br',
    35 : 'i'
}

GAFF_XML_PATH = os.path.join(os.path.dirname(__file__),'data/gaff.xml')
GAFF2_XML_PATH = os.path.join(os.path.dirname(__file__),'data/gaff2.xml')

# =============================================================================
# module classes
# =============================================================================
class ForceFieldBase(object):
    """ Base class for forcefield.
    """
    def __init__(self):
        pass

    def load_typing_translation_dict(self, typing_translation_dict):
        self.typing_translation_dict = typing_translation_dict

    def load_param_xml(self, xml_path):
        """ Load a parameter xml file.

        Parameters
        ----------
        xml_path : str
            path where the xml file for the forcefield lives
        """
        # read the xml file
        f_handle = open(xml_path, 'r')
        tree = ET.parse(f_handle)
        root = tree.getroot()
        self.root = root

    def get_bond(self, atom1_type, atom2_type):
        length = 0.
        k = 0.

        # get the strings for atom0 and atom1
        atom1_str = self.typing_translation_dict[atom1_type]
        atom2_str = self.typing_translation_dict[atom2_type]

        # get the entry for the bond
        bond_entry = self.root.find(
            './/HarmonicBondForce/Bond'
            '[@type1=\"%s\"][@type2=\"%s\"]'\
            % (
                atom1_str,
                atom2_str,
            ))

        if type(bond_entry) == type(None):

            bond_entry = self.root.find(
                './/HarmonicBondForce/Bond'
                '[@type1=\"%s\"][@type2=\"%s\"]'\
                % (
                    atom2_str,
                    atom1_str,
                ))


        # get length and k
        length = float(bond_entry.get('length'))
        k = float(bond_entry.get('k'))

        return length, k

    def get_angle(self, atom1_type, atom2_type, atom3_type):
        # get the string representation of atom0, 1, and 2
        # NOTE: atom1 is the atom in the center
        angle = .0
        k = .0

        # TODO: solve the inconsistency between atom types
        #       to get rid of try-except

        try:
            atom1_str = self.typing_translation_dict[atom1_type]
            atom2_str = self.typing_translation_dict[atom2_type]
            atom3_str = self.typing_translation_dict[atom3_type]

            # get the entry for the bond
            bond_entry = self.root.find(
                './/HarmonicAngleForce/Angle'
                '[@type1=\'%s\'][@type2=\'%s\'][@type3=\'%s\']'\
                % (
                    atom1_str,
                    atom2_str,
                    atom3_str,
                ))

            if type(bond_entry) == type(None):
                bond_entry = self.root.find(
                    './/HarmonicAngleForce/Angle'
                    '[@type1=\'%s\'][@type2=\'%s\'][@type3=\'%s\']'\
                    % (
                        atom3_str,
                        atom2_str,
                        atom1_str,
                    ))

            # get length and k
            angle = float(bond_entry.get('angle'))
            k = float(bond_entry.get('k'))

        except:
            pass

        return angle, k

    def get_proper(
            self,
            atom1_type,
            atom2_type,
            atom3_type,
            atom4_type):

        # get the string representation of atom2 and 3
        # NOTE: here only atom 2 and 3 matter here
        atom1_str = self.typing_translation_dict[atom1_type]
        atom2_str = self.typing_translation_dict[atom2_type]
        atom3_str = self.typing_translation_dict[atom3_type]
        atom4_str = self.typing_translation_dict[atom4_type]

        periodicity1 = .0
        phase1 = .0
        k1 = .0

        periodicity2 = .0
        phase2 = .0
        k2 = .0

        periodicity3 = .0
        phase3 = .0
        k3 = .0

        # get the entry of the proper torsion
        # first search for the entry
        proper_entry = self.root.find(
            './/PeriodicTorsionForce/Proper'
            '[@type1=\'%s\'][@type2=\'%s\'][@type3=\'%s\'][@type4=\'%s\']'\
            %(
                atom1_str,
                atom2_str,
                atom3_str,
                atom4_str,
            ))

        if type(proper_entry) == type(None):
            proper_entry = self.root.find(
                './/PeriodicTorsionForce/Proper'
                '[@type1=\'%s\'][@type2=\'%s\'][@type3=\'%s\'][@type4=\'%s\']'\
                %(
                    atom4_str,
                    atom3_str,
                    atom2_str,
                    atom1_str,
                ))

        if type(proper_entry) == type(None):
            # try only matching two atoms
            proper_entry = self.root.find(
                './/PeriodicTorsionForce/Proper'
                '[@type2=\'%s\'][@type3=\'%s\']'
                %(
                    atom2_str,
                    atom3_str,
                ))

        if type(proper_entry) == type(None):
            # try only matching two atoms
            proper_entry = self.root.find(
                './/PeriodicTorsionForce/Proper'
                '[@type3=\'%s\'][@type2=\'%s\']'
                %(
                    atom2_str,
                    atom3_str,
                ))
        try:
            periodicity1 = float(proper_entry.get('periodicity1'))
            phase1 = float(proper_entry.get('phase1'))
            k1 = float(proper_entry.get('k1'))
        except:
            pass

        try:
            periodicity2 = float(proper_entry.get('periodicity2'))
            phase2 = float(proper_entry.get('phase2'))
            k2 = float(proper_entry.get('k2'))
        except:
            pass

        try:
            periodicity3 = float(proper_entry.get('periodicity3'))
            phase3 = float(proper_entry.get('phase3'))
            k3 = float(proper_entry.get('k3'))
        except:
            pass

        return (
            periodicity1, phase1, k1,
            periodicity2, phase2, k2,
            periodicity3, phase3, k3)

    def get_improper(
            self,
            atom1_type,
            atom2_type,
            atom3_type,
            atom4_type):

        # get the string representation of atom2 and 3
        # NOTE: here only atom 2 and 3 matter here
        atom1_str = self.typing_translation_dict[atom1_type]
        atom2_str = self.typing_translation_dict[atom2_type]
        atom3_str = self.typing_translation_dict[atom3_type]
        atom4_str = self.typing_translation_dict[atom4_type]

        periodicity1 = .0
        phase1 = .0
        k1 = .0

        periodicity2 = .0
        phase2 = .0
        k2 = .0

        periodicity3 = .0
        phase3 = .0
        k3 = .0

        # get the entry of the proper torsion
        # first search for the entry
        improper_entry = self.root.find(
            './/PeriodicTorsionForce/Improper'
            '[@type1=\'%s\'][@type2=\'%s\'][@type3=\'%s\'][@type4=\'%s\']'\
            %(
                atom1_str,
                atom2_str,
                atom3_str,
                atom4_str,
            ))

        if type(improper_entry) == type(None):
            improper_entry = self.root.find(
                './/PeriodicTorsionForce/Improper'
                '[@type1=\'%s\'][@type2=\'%s\'][@type3=\'%s\'][@type4=\'%s\']'\
                %(
                    atom4_str,
                    atom3_str,
                    atom2_str,
                    atom1_str,
                ))

        if type(improper_entry) == type(None):
            # try only matching two atoms
            improper_entry = self.root.find(
                './/PeriodicTorsionForce/Improper'
                '[@type2=\'%s\'][@type3=\'%s\']'
                %(
                    atom2_str,
                    atom3_str,
                ))

        if type(improper_entry) == type(None):
            # try only matching two atoms
            improper_entry = self.root.find(
                './/PeriodicTorsionForce/Improper'
                '[@type3=\'%s\'][@type2=\'%s\']'
                %(
                    atom2_str,
                    atom3_str,
                ))

        try:
            periodicity1 = float(improper_entry.get('periodicity1'))
            phase1 = float(improper_entry.get('phase1'))
            k1 = float(improper_entry.get('k1'))
        except:
            pass

        try:
            periodicity2 = float(improper_entry.get('periodicity2'))
            phase2 = float(improper_entry.get('phase2'))
            k2 = float(improper_entry.get('k2'))
        except:
            pass

        try:
            periodicity3 = float(improper_entry.get('periodicity3'))
            phase3 = float(improper_entry.get('phase3'))
            k3 = float(improper_entry.get('k3'))
        except:
            pass

        return (
            periodicity1, phase1, k1,
            periodicity2, phase2, k2,
            periodicity3, phase3, k3)

    def get_nonbonded(self, atom_type):
        """ Get the nonbonded entry.
        """
        sigma = .0
        epsilon = .0

        atom_str = self.typing_translation_dict[atom_type]
        nonbonded_entry = self.root.find(
            './/NonbondedForce/Atom[@type=\'%s\']' % atom_str)

        try:
            sigma = float(nonbonded_entry.get('sigma'))
            epsilon = float(nonbonded_entry.get('epsilon'))
        except:
            pass

        return sigma, epsilon

    def get_onefour_scaling(self):
        """ Get the scaling constant for one-four interactions.

        """
        coulomb14scale = .0
        lj14scale = .0

        try:
            nonbonded_entry = self.root.find('.//NonbondedForce')
            coulomb14scale = float(nonbonded_entry.get('coulomb14scale'))
            lj14scale = float(nonbonded_entry.get('lj14scale'))
        except:
            pass

        return coulomb14scale, lj14scale

class GAFF(ForceFieldBase):
    def __init__(self):
        super(GAFF, self).__init__()
        self.load_typing_translation_dict(GAFF_TYPING_TRANSLATION_DICT)
        self.load_param_xml(GAFF_XML_PATH)

gaff = GAFF()
