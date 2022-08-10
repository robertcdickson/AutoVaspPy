import os
import matplotlib.font_manager as font_manager
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.io.vasp.inputs import Kpoints


def absorbance_plot(
        files,
        plot_labels,
        plot_colors=None,
        fill=True,
        dark_theme=True,
):
    """
    Plot the absorption spectrum from vasprun.xml files


    Args:
        files: list
            List of vasprun.xml files to read and plot
        plot_labels: list
            Labels for each plot
        plot_colors: list
            Color of each plot
        save_file: str
            File to save figure to
        fill: bool
            If True, fill between plot and x axis
        xlims: list
            List of minimum and maximum x values
        ylims: list
            List of minimum and maximum y values
        dark_theme: bool
            If True, A dark theme is used

    Returns:

    """
    import theme_colors

    if not plot_colors:
        plot_colors = theme_colors.colours

    # font
    font_path = "/usr/share/fonts/avenir_ff/AvenirLTStd-Black.ttf"
    prop = font_manager.FontProperties(fname=font_path)

    # padding for axes
    rcParams["xtick.major.pad"] = "10"
    rcParams["ytick.major.pad"] = "10"
    rcParams["font.family"] = prop.get_name()

    if dark_theme:
        rcParams["axes.edgecolor"] = "white"
        rcParams["text.color"] = "white"
        rcParams["axes.labelcolor"] = "white"
        rcParams["xtick.color"] = "white"
        rcParams["ytick.color"] = "white"
    else:
        rcParams["axes.edgecolor"] = "black"
        rcParams["text.color"] = "black"
        rcParams["axes.labelcolor"] = "black"
        rcParams["xtick.color"] = "black"
        rcParams["ytick.color"] = "black"

    def plot(file_name, colour_name, label_name):
        """
        file_name: str
            Vasprun.xml files to read and plot
        colour_name: tuple
            Color for plot
        label_name: str
            label for legend
        """

        # read vasp run
        vaspout = Vasprun(file_name)

        # get dielectric function data
        de_data = vaspout.dielectric

        # energy for x axis
        energy = np.array(de_data[0])

        # change from eV to nm
        x_axis_nm = 1239.84193 / energy

        def get_dielectrics(vasprun):
            real_avg = np.array(
                [
                    sum(vasprun.dielectric[1][i][0:3]) / 3
                    for i in range(len(vasprun.dielectric_data["density"][0]))
                ]
            )
            imag_avg = np.array(
                [
                    sum(vasprun.dielectric[2][i][0:3]) / 3
                    for i in range(len(vasprun.dielectric_data["density"][0]))
                ]
            )
            real_avg = real_avg
            imag_avg = imag_avg
            ex = real_avg + imag_avg * 1j
            return ex

        eps = get_dielectrics(vaspout)
        norm_epsilon = np.sqrt(eps.real ** 2 + eps.imag ** 2)
        extinction_coefficient = np.sqrt((-eps.real + norm_epsilon) / 2.0)
        # refractive_index = np.sqrt((eps.real + norm_epsilon) / 2.0)
        # reflectivity = ((refractive_index - 1.) ** 2 + extinction_coefficient ** 2) / (
        #         (refractive_index + 1.) ** 2 + extinction_coefficient ** 2)
        absorption_coefficient = 4 * np.pi * extinction_coefficient / x_axis_nm

        # plot absorption and fill between
        if not colour_name:
            plt.plot(
                x_axis_nm,
                absorption_coefficient,
                color=colour_name,
                label=label_name,
                zorder=2,
            )
            if fill:
                plt.fill_between(
                    x_axis_nm, absorption_coefficient, alpha=0.25, zorder=0
                )
        else:
            plt.plot(
                x_axis_nm,
                absorption_coefficient,
                color=colour_name,
                label=label_name,
                zorder=2,
            )
            if fill:
                plt.fill_between(
                    x_axis_nm,
                    absorption_coefficient,
                    color=colour_name,
                    alpha=0.25,
                    zorder=0,
                )

        # set face colour
        fig = plt.gcf()
        fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
        if dark_theme:
            fig.set_facecolor((38 / 255, 38 / 255, 38 / 255))
        else:
            fig.set_facecolor((1, 1, 1))

        # change scientific notation text size
        ax = plt.gca()

        if dark_theme:
            ax.set_facecolor((38 / 255, 38 / 255, 38 / 255))
        else:
            ax.set_facecolor((1, 1, 1))

        ax.yaxis.get_offset_text().set_fontsize(16)
        if dark_theme:
            ax.yaxis.get_offset_text().set_color("w")
        else:
            ax.yaxis.get_offset_text().set_color("black")

        # change axes tick labels size
        ax.tick_params(axis="x", labelsize=22)
        ax.tick_params(axis="y", labelsize=22)

        # set axes limits
        plt.xlim(200, 900)
        plt.ylim(0, 0.1)

        # set axes labels
        if dark_theme:
            plt.xlabel("Wavelength / nm", fontsize=28, c="w")
            plt.ylabel("a.u.", fontsize=28, c="w")
        else:
            plt.xlabel("Wavelength / nm", fontsize=28, c="black")
            plt.ylabel("a.u.", fontsize=28, c="black")

        # legend and tight layout
        plt.legend(fontsize=18, fancybox=True, framealpha=0)

    if type(files) == list:
        for f, col, l in zip(files, plot_colors, plot_labels):
            plot(f, col, l)
    elif type(files) == str:
        plot(files, plot_colors, plot_labels)



def element_dos_plot(
        dosrun,
        colours,
        process=False,
        total_dos=False,
        xlim=None,
        ylim=None,
        save_file=None,
        dark_theme=True,
        separate_spins=True,
):
    """
    Plot elemental density of states from vasprun.xml files

    Args:
        dosrun: str
            vasprun.xml file to load
        colours: dict
            dict of colors of plot
        process: bool
            If True, process colours before
        total_dos: bool
            If True, plot total density of states
        xlim: list
            Defines the min and max of the x axis
        ylim: list
             Defines the min and max of the y axis
        save_file: str
            Name of file to save plot to
        dark_theme: bool
            If True, plot with a dark theme
        separate_spins: bool
            If True, plot DOS with two spin channels

    Returns:
        None

    """
    if ylim is None:
        ylim = []
    if xlim is None:
        xlim = []
    fontpath = "/usr/share/fonts/avenir_ff/AvenirLTStd-Black.ttf"
    prop = font_manager.FontProperties(fname=fontpath)

    # padding for axes
    rcParams["xtick.major.pad"] = "10"
    rcParams["ytick.major.pad"] = "10"
    rcParams["font.family"] = prop.get_name()

    if dark_theme:
        rcParams["axes.edgecolor"] = "white"
        rcParams["text.color"] = "white"
        rcParams["axes.labelcolor"] = "white"
        rcParams["xtick.color"] = "white"
        rcParams["ytick.color"] = "white"
    else:
        rcParams["axes.edgecolor"] = "black"
        rcParams["text.color"] = "black"
        rcParams["axes.labelcolor"] = "black"
        rcParams["xtick.color"] = "black"
        rcParams["ytick.color"] = "black"

    # process colours given
    if process:
        for key, value in zip(colours.keys(), colours.values()):
            colours[key] = [x / 255 for x in colours[key]]

    # load file
    run = Vasprun(dosrun, separate_spins=separate_spins)

    # print bands
    print(run.eigenvalue_band_properties[0])

    # get the total dos
    dos = run.complete_dos

    # get dos of each unique element in system
    pdos = dos.get_element_dos()
    # plotter object
    plotter = DosPlotter()

    # add the total dos
    if total_dos:
        plotter.add_dos("Total", dos)
        # make plot
        pdos_plot = plotter.get_plot()
    else:
        # add all element dos
        plotter.add_dos_dict(pdos)

        # make plot
        pdos_plot = plotter.get_plot()

    # set face colour
    fig = pdos_plot.gcf()
    fig.subplots_adjust(left=0.15, right=1, top=1, bottom=0.15)

    if dark_theme:
        fig.set_facecolor((38 / 255, 38 / 255, 38 / 255))
    else:
        fig.set_facecolor((255 / 255, 255 / 255, 255 / 255))

    # alter look
    ax = pdos_plot.gca()
    ax.set_xlabel("Energy / eV", fontsize=32)
    ax.set_ylabel("Density of States", fontsize=32)

    # set dark color
    if dark_theme:
        ax.set_facecolor((38 / 255, 38 / 255, 38 / 255))
    else:
        ax.set_facecolor((255 / 255, 255 / 255, 255 / 255))

    # make labels for legend
    labels = [ax.lines[i].get_label() for i in range(len(ax.lines))]

    # change fermi energy line and axis formatting
    ax.lines[-1].set_linestyle("-")
    ax.lines[-1].set_linewidth(1)
    ax.lines[-2].set_linewidth(1)
    if dark_theme:
        ax.lines[-1].set_color("w")
        ax.lines[-2].set_color("w")
    else:
        ax.lines[-1].set_color("black")
        ax.lines[-2].set_color("black")

    # set custom colours (doesn't set for any colours not defined in dict_colours)
    for label, line in zip(labels, ax.lines):
        try:
            line.set_color(colours[label])
            line.set_linewidth(2)
        except KeyError:
            continue

    # light fill below dos
    for key, p in zip(pdos.keys(), pdos.values()):
        if separate_spins:
            pdos_plot.fill_between(
                dos.energies - dos.efermi,
                p.densities[Spin.up],
                color=colours[key.symbol],
                alpha=0.5,
            )
            pdos_plot.fill_between(
                dos.energies - dos.efermi,
                -p.densities[Spin.down],
                color=colours[key.symbol],
                alpha=0.5,
            )
        else:
            pdos_plot.fill_between(
                dos.energies - dos.efermi,
                p.densities[Spin.up],
                color=colours[key.symbol],
                alpha=0.5,
            )

    # alter tick size
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)

    # get legend
    pdos_plot.legend()
    handles, labels = ax.get_legend_handles_labels()
    pdos_plot.legend(
        handles[::-1], labels[::-1], fontsize=24, fancybox=True, framealpha=0
    )

    # change axis limits
    if not xlim:
        pdos_plot.xlim(-5, 5)
    else:
        pdos_plot.xlim(xlim[0], xlim[1])
    if not ylim:
        pdos_plot.ylim(-40, 40)
    else:
        pdos_plot.ylim(ylim[0], ylim[1])

    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, dpi=300)
    else:
        plt.show()


def site_dos_plot(
        dosrun,
        colours,
        dos_sites=None,
        total_dos=False,
        save_file=None,
        dark_theme=True,
        xlim=None,
        ylim=None,
):
    """
    Plot elemental density of states from vasprun.xml files

    Args:
        dosrun: str
            vasprun.xml file to load
        colours: dict
            dict of colors of plot
        process: bool
            If True, process colours before
        total_dos: bool
            If True, plot total density of states
        xlim: list
            Defines the min and max of the x axis
        ylim: list
             Defines the min and max of the y axis
        save_file: str
            Name of file to save plot to
        dark_theme: bool
            If True, plot with a dark theme
        separate_spins: bool
            If True, plot DOS with two spin channels

    Returns:
        None

    """

    if ylim is None:
        ylim = []
    if xlim is None:
        xlim = []
    if dos_sites is None:
        dos_sites = {}
    default_colours_dict = {
        0: [204 / 255, 0 / 255, 0 / 255],
        1: [255 / 255, 189 / 255, 71 / 255],
        2: [0 / 255, 204 / 255, 102 / 255],
        3: [51 / 255, 204 / 255, 255 / 255],
        4: [204 / 255, 51 / 255, 255 / 255],
        5: [204 / 255, 0 / 255, 102 / 255],
        6: [255 / 255, 555 / 255, 555 / 255],
        "Total": [0, 0, 0],
    }
    if not colours:
        colours = default_colours_dict

    fontpath = "/usr/share/fonts/avenir_ff/AvenirLTStd-Black.ttf"
    prop = font_manager.FontProperties(fname=fontpath)

    # padding for axes
    rcParams["xtick.major.pad"] = "10"
    rcParams["ytick.major.pad"] = "10"
    rcParams["font.family"] = prop.get_name()
    rcParams["figure.figsize"] = 4, 6
    if dark_theme:
        rcParams["axes.edgecolor"] = "white"
        rcParams["text.color"] = "white"
        rcParams["axes.labelcolor"] = "white"
        rcParams["xtick.color"] = "white"
        rcParams["ytick.color"] = "white"
    else:
        rcParams["axes.edgecolor"] = "black"
        rcParams["text.color"] = "black"
        rcParams["axes.labelcolor"] = "black"
        rcParams["xtick.color"] = "black"
        rcParams["ytick.color"] = "black"

    # process colours given
    # for key, value in zip(colours.keys(), colours.values()):
    #    colours[key] = [x / 255 for x in colours[key]]

    # load file
    run = Vasprun(dosrun)

    # get the total dos
    dos = run.complete_dos

    # plotter object
    plotter = DosPlotter()

    # get dos of each site element in system
    plot_dict = {}
    for key, value in dos_sites.items():
        new_dos = None

        for site in value:
            if new_dos:
                new_dos += dos.get_site_dos(run.final_structure.sites[site])
            else:
                new_dos = dos.get_site_dos(run.final_structure.sites[site])

        plotter.add_dos(key, new_dos)
        plot_dict[key] = new_dos

    # add the total dos
    if total_dos:
        plotter.add_dos("Total", dos)

    # make plot
    pdos_plot = plotter.get_plot()

    # set face colour
    fig = pdos_plot.gcf()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
    if dark_theme:
        fig.set_facecolor((38 / 255, 38 / 255, 38 / 255))
    else:
        fig.set_facecolor((255 / 255, 255 / 255, 255 / 255))

    # alter look
    ax = pdos_plot.gca()
    ax.set_xlabel("Energy / eV", fontsize=32)
    ax.set_ylabel("Density of States", fontsize=32)

    # set custom colours (doesn't set for any colours not defined in dict_colours)
    for i, line in zip(reversed(list(colours.values())), ax.lines):
        line.set_color(i)
        line.set_linewidth(2)

    # set dark color
    if dark_theme:
        ax.set_facecolor((38 / 255, 38 / 255, 38 / 255))
    else:
        ax.set_facecolor((255 / 255, 255 / 255, 255 / 255))

    # change fermi energy line and axis formatting
    ax.lines[-1].set_linestyle("-")
    ax.lines[-1].set_linewidth(1)
    ax.lines[-2].set_linewidth(1)
    if dark_theme:
        ax.lines[-1].set_color("w")
        ax.lines[-2].set_color("w")
    else:
        ax.lines[-1].set_color("black")
        ax.lines[-2].set_color("black")

    # light fill below dos
    for j, p in zip(list(colours.values()), list(plot_dict.values())):
        pdos_plot.fill_between(
            dos.energies - dos.efermi, p.densities[Spin.up], color=j, alpha=0.5
        )
        pdos_plot.fill_between(
            dos.energies - dos.efermi, -p.densities[Spin.down], color=j, alpha=0.5
        )

    # fill between total dos
    if total_dos:
        pdos_plot.fill_between(
            dos.energies - dos.efermi, dos.densities[Spin.up], color="grey", zorder=0
        )
        pdos_plot.fill_between(
            dos.energies - dos.efermi, -dos.densities[Spin.down], color="grey", zorder=0
        )

    # alter tick size
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)

    # get legend
    pdos_plot.legend()
    handles, labels = ax.get_legend_handles_labels()
    pdos_plot.legend(
        handles[::-1], labels[::-1], fontsize=24, fancybox=True, framealpha=0
    )

    # change axis limits
    # change axis limits
    if not xlim:
        pdos_plot.xlim(-10, 10)
    else:
        pdos_plot.xlim(xlim[0], xlim[1])
    if not ylim:
        pdos_plot.ylim(-40, 40)
    else:
        pdos_plot.ylim(ylim[0], ylim[1])

    if save_file:
        plt.savefig(save_file, dpi=300)
    # plt.show()


def bands_plot(
        bandrun,
        save_directory=None
):
    """

    Args:
        bandrun: str
            File to load and plot bands from
        save_directory: str
            Location to save plot

    Returns:

    """

    # font settings
    fontpath = "/usr/share/fonts/avenir_ff/AvenirLTStd-Black.ttf"
    prop = font_manager.FontProperties(fname=fontpath)

    # padding for axes
    rcParams["xtick.major.pad"] = "10"
    rcParams["ytick.major.pad"] = "10"
    rcParams["font.family"] = prop.get_name()
    rcParams["figure.figsize"] = 4, 6
    rcParams["axes.edgecolor"] = "white"
    rcParams["text.color"] = "white"
    rcParams["axes.labelcolor"] = "white"
    rcParams["xtick.color"] = "white"
    rcParams["ytick.color"] = "white"

    os.chdir(bandrun)

    # get vasprun object from vasprun.xml file
    vaspout = Vasprun(bandrun + "/vasprun.xml")

    # get k-path of sc definition for plotting
    k_path = HighSymmKpath(
        structure=vaspout.final_structure, has_magmoms=True, path_type="sc"
    )
    kpts = Kpoints.automatic_linemode(divisions=50, ibz=k_path)

    # write new k-point that uses Line_mode in VASP
    kpts.write_file("./KPOINTS_LINE_MODE")

    # get band structure object for plotting
    bandstr = vaspout.get_band_structure(kpoints_filename="./KPOINTS_LINE_MODE")

    plot = BSPlotter(bandstr).get_plot(ylim=[-5, 5])

    # set face colour
    fig = plot.gcf()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
    fig.set_facecolor((38 / 255, 38 / 255, 38 / 255))

    # alter look
    ax = plot.gca()

    ax.tick_params(axis="both", which="major", labelsize=14)
    # set dark color
    ax.set_facecolor((38 / 255, 38 / 255, 38 / 255))
    ax.axhline(0, zorder=0)

    # A workaround for changing the colours of the lines in the plot
    for i, line in enumerate(ax.get_lines()):
        if i % 2 == 0:
            line.set_color((204 / 255, 0 / 255, 0 / 255))
        else:
            line.set_color((51 / 255, 204 / 255, 255 / 255))

    # A workaround for adding labels to the legend of the plot
    ax.get_lines()[0].set_label("Spin up")
    ax.get_lines()[1].set_label("Spin down")

    for line in ax.lines[-14:]:
        line.set_color("white")
        line.set_linestyle("-")
        line.set_zorder(0)

    plot.legend(fontsize=16, fancybox=True, framealpha=0)

    if save_directory:
        plot.savefig(save_directory)
    else:
        plot.show()

    return bandstr
