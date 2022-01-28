from AutoVaspPy import VaspCalculations
from ase.io import read
import os

os.chdir("other_tests/NiFe2O4")

# hubbard
hub = {'Ni': {'L': 2,
              'U': 6.2,  # need to recalculate this value
              'J': 0},
       'Fe': {'L': 2,
              'U': 5.0,
              'J': 0}}

settings = {
    "relax-mag": {"algo": "normal"},
    "scf-mag":
        {"write_safe_file": True},
    "bands-mag":
        {"read_safe_file": True},
    "eps-mag":
        {"write_safe_file": True, "read_safe_file": True}
}

mags = "4*5, 2*-5, 8*0"

# MnFe2O4_structure = read("./scf-mag/POSCAR")
for i in range(1):
    if not os.path.exists(f"./test{i}"):
        os.mkdir(f"./test{i}")
    os.chdir(f"./test{i}")
    init_structure = read("../POSCAR")
    calculations = VaspCalculations(init_structure).calculation_manager(calculation_sequence=["relax-mag"],
                                                                        additional_settings=settings, nkpts=10,
                                                                        magnetic_moments=mags,
                                                                        hubbard_parameters=hub)
    os.chdir("../")
"""calculations = VaspCalculations(init_structure).get_band_path()"""

# MnFe2O4_calculation.calc_manager(calculation_sequence=["bands-mag"])
