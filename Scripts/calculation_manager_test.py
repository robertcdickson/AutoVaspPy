from ASE_testing import VaspCalculations
from ase.io import read

# hubbard
hub = {'Mn': {'L': 2,
              'U': 3.9,
              'J': 0},
       'Fe': {'L': 2,
              'U': 5.3,
              'J': 0}}

# MnFe2O4_structure = read("./Cifs/MnFe2O4-Normal.cif")
MnFe2O4_structure = read("./relax/POSCAR")
MnFe2O4_calculation = VaspCalculations(MnFe2O4_structure)

MnFe2O4_calculation.calc_manager(calc_seq=["relax", "relax-mag"], hubbard_params=hub)
