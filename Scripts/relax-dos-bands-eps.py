from autochempy import VaspCalculations
from ase.io import read
import os

os.chdir("./tests/NiFe2O4")

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
        {"write_safe_file": True, "icharg": 1, "kpts": [1, 1, 1], "nedos": 2500},
    "bands-mag":
        {"read_safe_file": True},
    "eps-mag":
        {"write_safe_file": True, "read_safe_file": True, "kpts": [1, 1, 1], "nedos": 2500}
}
mags = "4*5, 2*-5, 8*0"
# MnFe2O4_structure = read("./scf-mag/POSCAR")
for i in range(1):
    if not os.path.exists(f"./test{i}"):
        os.mkdir(f"./test{i}")
    os.chdir(f"./test{i}")
    init_structure = read("../POSCAR")
    calculations = VaspCalculations(init_structure).calc_manager(calc_seq=["relax-mag"],
                                                             add_settings_dict=settings, nkpts=10,
                                                             magnetic_moments=mags,
                                                             hubbard_params=hub)
    os.chdir("../")
"""calculations = VaspCalculations(init_structure).get_band_path()"""

# MnFe2O4_calculation.calc_manager(calc_seq=["bands-mag"])
