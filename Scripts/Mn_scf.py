from AutoVaspPy import VaspCalculations
from ase.io import read
import os
import sys

cwd = os.getcwd()

MnFeOCl_structure = read("../Cifs/MnFe2O4-Inverse.cif")
MnFeOCl = VaspCalculations(MnFeOCl_structure, write_file="test.out")

MnFeOCl.calculation_manager(calculation_sequence=["relax"], add_settings_dict={"relax": {}})

