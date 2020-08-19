from ASE_testing import VaspCalculations
from ase.io import read

MnFe2O4_structure = read("../Cifs/MnFe2O4-Normal.cif")
MnFe2O4_Normal = VaspCalculations(MnFe2O4_structure)

MnFe2O4_Normal.single_vasp_calc(path_name="./scf")
