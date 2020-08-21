from ASE_testing import VaspCalculations
from ase.io import read

# MnFe2O4_structure = read("./Cifs/MnFe2O4-Normal.cif")
MnFe2O4_structure = read("./relax/POSCAR")
MnFe2O4_calculation = VaspCalculations(MnFe2O4_structure)

MnFe2O4_calculation.calc_manager(calc_seq=["relax", "relax-mag"])
