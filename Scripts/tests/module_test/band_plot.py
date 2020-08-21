from pymatgen.io.vasp.outputs import BSVasprun
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure
import os

vaspout = BSVasprun("./hubbard/hub_Mn-6_Fe-3/vasprun.xml")
bandstr = vaspout.get_band_structure(line_mode=True)

print(bandstr.get_band_gap())
print(bandstr.get_direct_band_gap_dict())

plt = BSPlotter(bandstr).get_plot(ylim=[-10, 5])
plt.show()


# get band gap for all directories

def get_band_gap(directory):
    vasp_out = Vasprun(directory + "/vasprun.xml")
    band_str = vasp_out.get_band_structure(line_mode=True)
    band_gap = band_str.get_band_gap()
    print(band_gap)
    return band_gap

# for directory in os.getcwd():
#    get_band_gap(directory)
