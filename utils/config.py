"""
Configuration file for setting Global variables
"""

"""
Extraction of compounds from the PubChem Database
"""
MAX_PUBCHEM_CID = 160000000
MAX_N_ATOMS = 60
# H, C, N, O, F, P, S, Cl, Br
PERMITTED_ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17, 35}

"""
SCF calculations
"""
# ORCA input file header for Geometry Optimisation
DFT_OPT_HEADER = "!PM3 def2-SVP OPT"
# ORCA input file header for NMR shielding constants calculations
DFT_NMR_HEADER = "!HF-3c NMR"
# ORCA path
ORCA_PATH = "/Users/ORCA"

"""
Molecular graph parameters
"""
# Fixed Dimension of Adjacency Matrix - maximal permitted number of atoms in the molecule stripped of hydrogens
DIMENSION = 54
# 0 - no bond 1,2,3 and 4 - aromatic
NUM_BOND_TYPES = 5
# Shielding constant range definition
SHIELDING_MAX = 1000
SHIELDING_MIN = -1000
