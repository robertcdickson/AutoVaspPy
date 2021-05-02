from AutoVaspPy import VaspCalculations
from ase.io import read

MnFeOCl_structure = read("../Cifs/Mn7FeCl3O10.cif")
MnFeOCl = VaspCalculations(MnFeOCl_structure)

MnFeOCl.calculation_manager(calculation_sequence=["relax"])