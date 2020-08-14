from ase.calculators.vasp import Vasp
from ase.io import read
import os
import shutil
import time


"""def vasp_calculation(vasp_output_file, main_settings, additional_settings):

    with open(vasp_output_file, "w") as vasp_out:

        # defining vasp settings
        vasp_settings = main_settings.copy()
        vasp_settings.update(additional_settings)

        if os.path.isfile("OSZICAR"):
            with open("OSZICAR", mode="r") as oszicar_file:
                shutil.copyfileobj(oszicar_file, vasp_out)"""


structure = read("./Cifs/new_cif.cif")
calc = Vasp(prec='Accurate',
            xc='PBE',
            lreal=False)
structure.set_calculator(calc)
x = structure.get_potential_energy()

print(structure)

