from pymatgen.io.vasp.outputs import BSVasprun
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure
import os

# vaspout = BSVasprun("./hubbard/hub_Mn-5_Fe-5/vasprun.xml")
vaspout = BSVasprun("../calc_seq_test/bands-mag/vasprun.xml")
# 50 k-points = 0.3863000000000012 eV
# 100 k-points = 0.38480000000000114 eV
# 8x8x8 50 k-points = 0.39029999999999987 eV
# 50 k-points with LASPH = 0.3932000000000002 eV
bandstr = vaspout.get_band_structure(line_mode=True)

print(bandstr.get_band_gap())
# print(bandstr.get_direct_band_gap_dict())

plt = BSPlotter(bandstr).get_plot(ylim=[-10, 5])
plt.show()


# get band gap for all directories

def get_band_gap(directory):
    vasp_out = BSVasprun(directory + "/vasprun.xml")
    band_str = vasp_out.get_band_structure(line_mode=True)
    band_gap = band_str.get_band_gap()
    print(band_gap)
    return band_gap


get_band_gap("../calc_seq_test/bands-mag")
# for directory in os.getcwd():
#    get_band_gap(directory)
