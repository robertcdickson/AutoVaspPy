from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.electronic_structure.bandstructure import BandStructure

vaspout = Vasprun("./bands/vasprun.xml")
bandstr = vaspout.get_band_structure(line_mode=True)

print(bandstr.get_band_gap())

plt = BSPlotter(bandstr).get_plot(ylim=[-10, 5])
plt.show()
