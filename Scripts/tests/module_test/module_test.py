from ASE_testing import VaspCalculations
from ase.io import read

MnFe2O4_structure = read("../../../Cifs/MnFe2O4-Normal.cif")
MnFe2O4_Normal = VaspCalculations(MnFe2O4_structure)

"""for atom in MnFe2O4_structure:
    print(atom)
    atom.add("magmom = 6")"""

# single scf
# MnFe2O4_Normal.single_vasp_calc(path_name="./scf")

# single bands non-magnetic
# MnFe2O4_Normal.single_vasp_calc(calculation_type="bands", path_name="./bands", use_safe_file=False, nkpts=50)

# hubbard parameters
hub = {
    "ldau_luj": {'Mn': {'L': 2, 'U': 3.9, 'J': 0},
                 'Fe': {'L': 2, 'U': 5.3, 'J': 0}}
}

# magnetic and hubbard calculation
"""MnFe2O4_Normal.single_vasp_calc(calculation_type="scf", add_settings=hubbard,
                                path_name="./scf-mag", use_safe_file=True,
                                mags=[6, 6, 6, 6, -6, -6, 0, 0, 0, 0, 0, 0, 0, 0])"""

# loop to test multiple hubbard values
"""for i in range(3, 7):
    for j in range(3, 7):
        hubbard = {
            "ldau_luj": {'Mn': {'L': 2, 'U': i, 'J': 0},
                         'Fe': {'L': 2, 'U': j, 'J': 0}}
        }
        MnFe2O4_Normal.single_vasp_calc(calculation_type="bands", add_settings=hubbard,
                                        path_name=f"./hub_Mn-{i}_Fe-{j}", use_safe_file=True,
                                        mags=[6, 6, 6, 6, -6, -6, 0, 0, 0, 0, 0, 0, 0, 0])"""

MnFe2O4_Normal.get_band_path(40)
