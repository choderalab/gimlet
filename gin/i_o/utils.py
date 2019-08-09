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

def oemol_to_dict(oemol):
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
        if not 'WibergBondOrder' in bond.GetData():
            raise RuntimeError('Molecule does not have Wiberg Bond Orders')
        wbo = bond.GetData('WibergBondOrder')
        connectivity.append([a1, a2, wbo])

    mol_dict = {'atomic_symbols': atomic_symbols,
                'partial_charges': partial_charges,
                'connectivity': connectivity}

    return mol_dict

def file_to_oemols(filename):
    """Create OEMol from file. If more than one mol in file, return list of OEMols.

    Parameters
    ----------
    filename: str
        absolute path to
    title: str, title
        title for molecule. If None, IUPAC name will be given as title.

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