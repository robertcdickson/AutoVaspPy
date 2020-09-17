from ASE_testing import VaspCalculations
from ase.io import read
import os
import sys

cwd = os.getcwd()

print(os.environ["PATH"])
with open("./run_vasp.py", "w") as wf:
    wf.write("import os\n")
    wf.write("exitcode = os.system('mpirun vasp_std')")

MnFeOCl_structure = read("../Cifs/MnFe2O4-Inverse.cif")
MnFeOCl = VaspCalculations(MnFeOCl_structure, write_file="test.out")

MnFeOCl.calc_manager(calc_seq=["relax"], add_settings_dict={"relax": {}})

