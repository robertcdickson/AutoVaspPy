from ASE_testing import VaspCalculations
from ase.io import read

MnFeOCl_structure = read("../Cifs/Mn7FeCl3O10.cif")
MnFeOCl = VaspCalculations(MnFeOCl_structure)

MnFeOCl.calc_manager(calc_seq=["relax"])