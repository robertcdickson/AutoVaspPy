from ase.calculators.vasp import Vasp
from ase.io import read
from ase.dft.kpoints import bandpath
import os
import shutil
import numpy as np

# --------------------------------------------------------#
#  This module aims to be able to automate calculations  #
#  for my PhD project                                    #
# --------------------------------------------------------#

# Define all possible calculation parameters
systems = ["MnFe2O4", "NiFe2O4", "CuCr2O4", "CuFe2O4"]
inversion = ["inverse", "normal"]
test_names = {"hubbard": [3.5, 4.0, 4.5, 5.0], "k-points": [6, 7, 8], "ecut": [500, 550, 600, 700]}
functionals = ["PBE", "HSE06"]
calculations_types = ["relax", "scf", "bands", "eps"]  # could have parameters for more complex optical calcs
calculation_add_ons = ["functional", "magnetism" "hubbard"]

"""def band_k_path(struct=None, nkpts=20, view_path=False):
    plots the k-path suggested from Setawayan et al
    # get band path
    path = struct.cell.bandpath(npoints=nkpts)

    if view_path:
        path.plot()
        plt.show()
    return path"""


def converged_in_one_scf_cycle(outcar_file):
    """
    Determines whether a VASP calculation converged in a single SCF cycle.
    Parameters
    ----------
    outcar_file : str
        OUTCAR file for this vasp calculation.
    Returns
    -------
    converged : boolean
        True if the calculation has converged, and did so within a single SCF cycle.
    ---------------------------------------------------------------------------
    Paul Sharp 27/10/2017
    """

    aborting_ionic_loop_marker = "aborting loop"
    convergence_marker = "reached required accuracy"

    num_convergence_strings = 0
    num_ionic_loops = 0

    with open(outcar_file, mode="r") as outcar:
        for line in outcar:
            num_convergence_strings += line.count(convergence_marker)
            num_ionic_loops += line.count(aborting_ionic_loop_marker)

    converged = num_convergence_strings > 0 and num_ionic_loops == 1

    return converged


class VaspCalculations(object):
    def __init__(self, structure, calculations=None, tests=None, output_file="output.out", hubbard_parameters=None,
                 write_file=None):
        """

        :param structure:
        :param calculations:
        :param tests:
        :param output_file:
        :param hubbard_parameters:
        """

        self.general_calculation = {"reciprocal": True,
                                    "xc": "PBE",
                                    "setups": 'materialsproject',
                                    "encut": 520,
                                    "kpts": [6, 6, 6],
                                    "ismear": 0,
                                    "sigma": 0.05,
                                    "prec": 'accurate',
                                    "lorbit": 11,
                                    "lasph": True}

        self.relax = {"ibrion": 2,  # determines how ions are moved and updated (MD or relaxation)
                      "nsw": 50,  # number of ionic steps
                      "isif": 3,  # allows for atomic positions, cell shape and cell volume as degrees of freedom
                      }

        self.parameters = {
            # "scf": {"icharg": 2},
            # TODO: Check if these parameters are correct

            "bands": {"icharg": 11},

            "eps": {"algo": "exact",
                    "loptics": True,
                    "nbands": 100,
                    "nedos": 1000},

            "hse06": {"xc": "HSE06"},

            "hubbard": {"ldau": True,
                        "ldau_luj": hubbard_parameters,
                        "ldauprint": 2}
        }

        self.structure = structure

        self.calculations = calculations
        self.output_file = output_file
        self.tests = tests

        self.owd = os.getcwd()
        self.safe_dir = self.owd + "/safe"
        self.last_dir = self.safe_dir

        self.write_file = write_file
        self.f = open(self.write_file, "a+")

    def parameter_testing(self, test, values):
        # ------------------------------------------------------------------ #
        #  This function allows for testing of many different values for a   #
        #                          given test                                #
        # ------------------------------------------------------------------ #

        # TODO: Make option to plot and save figure of convergence test
        # TODO: At the moment, output is only written at end of all calculations as with open statement ends
        #   This needs to be amended so output is written as each calculation finishes

        self.f.write("--------------------" * 50 + "\n")
        self.f.write("{} testing with a list of values of {}\n".format(test, values))

        # define path and check if directory exists
        path_name = "./tests/{}".format(test)

        if not os.path.exists(path_name):
            os.mkdir(path_name)
        os.chdir(path_name)

        self.f.write("Testing calculations running from the directory: {} \n".format(path_name))
        self.f.write("{},Energy \n".format(test))

        # list for storage of tested output
        energies = []

        # append the general VASP keywords for the test
        vasp_keywords = self.general_calculation.copy()

        for test_value in values:
            test_path = "./{}".format(test_value)

            if not os.path.exists(test_path):
                os.mkdir(test_path)
            os.chdir(test_path)

            # statements to find which type of test is being used
            if test == "k-points":
                test_variable = "kpts"
                test_value = [test_value, test_value, test_value]
            elif test == "ecut" or "encut":
                test_variable = "encut"
            else:
                raise ValueError('Error: test requested is not implemented. Please check test argument and try '
                                 'again.')

            # update keywords dictionary to have new test value
            vasp_keywords.update({test_variable: test_value})

            # set calculator
            self.structure.set_calculator(Vasp(**vasp_keywords))

            # run calculation
            try:
                energy = self.structure.get_potential_energy()
            except (TypeError, ValueError):
                print("A VASP error has occurred in test: {} | {}. Please check again".format(test, test_value))
                energy = 0
            self.f.write("{},{}\n".format(test_value, energy))

            energies.append(energy)
            os.chdir("../")

        self.f.write("Testing Finished!")
        self.f.write("-" * 50 + "\n")

        return energies

    def make_hse_kpoints(self, ibz_file, nkpts):
        """ This function aims to make the KPOINTS file needed for hybrid functional band structures

        The required sequence is as follows:
        1.  standard non-hybrid DFT run
        2.  using converged WAVECAR, hybrid-DFT run
        3.  zero-weight KPOINTS run using IBZKPTS file from (2)

        "Mind: Remove from the band structure plot the eigenvalues corresponding to the the regular k-points mesh."
        :return:
        """
        shutil.copy2(ibz_file, "./KPOINTS")
        band_path = self.get_band_path(nkpts=nkpts)

        # TODO: Clean and finish this

    def calc_manager(self, calc_seq=None, add_settings_dict=None, mags=None, hubbard_params=None, nkpts=200,
                     outfile="vasp_seq.out"):

        if add_settings_dict is None:
            add_settings_dict = {}

        relaxation_w_mag = ["relax", "relax-mag"]  # works!
        bands_w_mag = ["scf-mag", "bands-mag"]  # works w/o hubbard or magnetism
        eps = ["scf", "scf-high-k", "eps"]
        hse06_bands = ["scf", "hse06"]

        """
        For HSE06 bands need to do an scf, get the IBZKPTS and WAVECAR files and use these with zero weight band k-path
        for the HSE06 band structure
        """

        # all output of individual calculations is written to the one file (which note: is always open)
        self.f.write("-" * 30 + "\n")
        self.f.write(f"Calculation sequence consists of: {calc_seq} \n")
        self.f.write(f"Additional settings: {add_settings_dict} \n")
        self.f.write("-" * 30 + "\n")

        # default calculation sequence is relaxation and scf
        if calc_seq is None:
            calc_seq = ["relax", "scf"]

        # loop through all calculations
        for i, calc in enumerate(calc_seq):
            self.f.write(f"Beginning calculation: {calc} as calculation {i + 1} in sequence \n")

            if os.path.exists("./safe/POSCAR"):
                self.f.write("POSCAR exists in safe! \n")

            """
            Here we need different settings for all calculations:
                
                relax: unmodified -> magnetic
                scf: magnetic -> hubbard/HSE06 
                dos: usually same as scf; dos_settings can be changed (although usually LORBIT = 11)
                bands: scf CHGCAR always needed (?)
                eps: scf with high bands and k-points needed
                
            These all needed saved individually in their respective directories and need to be able to check for other 
            calculations already done as:
            
                relax <-> scf <-> dos/bands/eps
                
            Would also like a series of standard calculation sequences 
            """

            # make separate directories for each calculation
            path = f"./{calc}"
            self.f.write(f"file path is {path} \n")

            # check if individual calculation is magnetic and if hubbard parameters are specified
            # magnetic check
            if calc.find("mag") != -1:
                self.f.write("Calculation is magnetic \n")
                if not mags:
                    self.f.write("No magnetic moments are specified. Are you sure this is correct? \n")
                mag_moments = mags
                # hubbard check
                if not hubbard_params:
                    self.f.write("No hubbard parameters requested. Are you sure this calculation will converge "
                                 "without? \n")
                else:
                    # check if add_settings already exists and append ldau_luj values
                    self.f.write(f"Hubbard values to be used are as follows: {hubbard_params} \n")

                    if not add_settings_dict[calc]:
                        add_settings_dict[calc] = {}
                    add_settings_dict[calc]["ldau_luj"] = hubbard_params
            else:
                mag_moments = None

            # relax uses self.relax_struct(), whereas all other calculations used self.single_vasp_calc()
            # the string "find" function is used to find any (mag or non-mag) relaxations

            # used to determine whether to write a safe file or not
            if calc.find("scf") != 1:
                wsf = True
            else:
                wsf = False

            if calc.find("relax") != -1:
                current_struct, energy = self.relax_struct(path_name=path, add_settings=add_settings_dict[calc],
                                                           write_safe_files=wsf,
                                                           mags=mag_moments)
            else:
                # run calculation
                current_struct, energy = self.single_vasp_calc(calculation_type=calc,
                                                               add_settings=add_settings_dict[calc],
                                                               path_name=path,
                                                               use_safe_file=True,
                                                               write_safe_files=wsf,
                                                               mags=mag_moments,
                                                               nkpts=nkpts)

            self.f.write("Relaxation successfully converged in single ionic step!")
            self.f.write(f"Calculations on {calc_seq} \n")

    def run_vasp(self, vasp_settings, restart=False):
        """
        Run a single VASP calculation using an ASE calculator.
        This routine will make use of WAVECAR and CONTCAR files if they are available.
        Parameters
        ----------
        structure : ASE atoms
            The structure used in the VASP calculation
        vasp_settings : dict
            The set of VASP options to apply with their values.
        Returns
        -------
        structure : ase atoms
            The structure after performing this calculation.
        energy : float
            Energy of the structure in eV.
        result : string
            The result of the VASP calculation,
            either "converged", "unconverged", "vasp failure", or "timed out"
        ---------------------------------------------------------------------------
        Paul Sharp 25/09/2017
        :param vasp_settings:
        :param restart:
        """

        # Set files for calculation
        energy = 0.0
        result = ""

        # Use ASE calculator -- the use of **kwargs in the function call allows us to set the desired arguments using a
        # dictionary
        structure = self.structure
        structure.set_calculator(Vasp(**vasp_settings, restart=restart))

        # Run calculation, and consider any exception to be a VASP failure
        try:
            energy = structure.get_potential_energy()
        except ValueError:
            result = "vasp failure"

        return structure, energy, result

    def relax_struct(self, add_settings=None, path_name="./relax",
                     write_safe_files=False, mags=False):

        # check if directory already exists and if not change to directory
        if not os.path.exists(path_name):
            os.mkdir(path_name)
        os.chdir(path_name)

        # if safe files already exists use them
        if os.path.exists(self.safe_dir):
            self.structure = read(self.safe_dir + "/POSCAR")
            shutil.copy2(self.safe_dir + "/POSCAR", "./")
            shutil.copy2(self.safe_dir + "/CHGCAR", "./")
            shutil.copy2(self.safe_dir + "/WAVECAR", "./")

        # rewrite the last_dir in case it is needed later
        self.last_dir = path_name

        # copy vasp settings from standard set-up
        vasp_settings = self.general_calculation.copy()

        # update for relaxation type
        vasp_settings.update(self.relax)

        # update for any addition settings wanted
        if add_settings:
            vasp_settings.update(add_settings)

        # add magnetic moments to structure object
        if mags:
            self.structure.set_initial_magnetic_moments(magmoms=mags)
            vasp_settings.update({"isif": 2, "ispin": 2, "icharg": 1})  # isif 2 doesn't change cell shape

        # while loop breaks when a relaxation converges in one ion relaxation step
        converged = False
        while not converged:

            # Copy CONTCAR to POSCAR for next stage of calculation -- use "copy2()" because it copies metadata
            # and permissions
            if os.path.isfile("CONTCAR"):
                shutil.copy2("CONTCAR", "POSCAR")
                self.structure = read("./POSCAR")  # need to read in the new POSCAR after every run for consistency
                vasp_settings.update({"icharg": 1})

            # run calculation
            structure, energy, result = self.run_vasp(vasp_settings)
            if converged_in_one_scf_cycle("OUTCAR"):
                break

        # save files to a safe directory for future use
        if write_safe_files:
            safe_dir = self.safe_dir
            if not os.path.exists(safe_dir):
                os.mkdir(safe_dir)

            # copy CONTCAR to POSCAR to save new structure compatible with WAVECAR and CHGCAR
            shutil.copy2("CONTCAR", safe_dir + "/POSCAR")
            shutil.copy2("CHGCAR", safe_dir)
            shutil.copy2("WAVECAR", safe_dir)

        os.chdir(self.owd)
        return structure, energy

    def single_vasp_calc(self, calculation_type="scf", functional="PBE", add_settings=None, path_name="./", nkpts=200,
                         use_safe_file=False, write_safe_files=False, mags=None):
        """
        A self-contained function that runs a single VASP calculation
        :param functional:
        :param write_safe_files:
        :param use_last_file:
        :param scf_save:
        :param safe_dir:
        :param mags:
        :param use_safe_file:
        :param path_name:
        :param nkpts:
        :param calculation_type:
        :param add_settings:
        :return:
        """

        # check if directory already exists and if not change to directory
        if not os.path.exists(path_name):
            os.mkdir(path_name)
        os.chdir(path_name)

        # if safe files to be used copy to cwd
        if use_safe_file:
            shutil.copy2(self.safe_dir + "/POSCAR", "./")
            shutil.copy2(self.safe_dir + "/CHGCAR", "./")
            shutil.copy2(self.safe_dir + "/WAVECAR", "./")

        # rewrite the last_dir in case it is needed later
        self.last_dir = path_name

        # copy vasp settings from standard set-up
        vasp_settings = self.general_calculation.copy()

        # check for magnetism
        if mags:
            self.structure.set_initial_magnetic_moments(magmoms=mags)
            vasp_settings.update({"ispin": 2})

        # update settings for calculation type
        calc_strip = calculation_type.strip("-mag")
        vasp_settings.update(self.parameters[calc_strip])

        # check for hybrid
        if calculation_type.find("hse06") != -1:
            vasp_settings.update()

        # add any extra settings
        if add_settings:
            vasp_settings.update(add_settings)

        # if band structure type calculation the get_band_path function for the k-point path
        if calculation_type == "bands" or calculation_type == "bands-mag":
            vasp_settings.update(kpts=self.get_band_path(nkpts=nkpts))

        # run energy calculation
        structure, energy, result = self.run_vasp(vasp_settings)

        # save files to a safe directory for future use
        if write_safe_files:
            safe_dir = self.safe_dir

            # copy CONTCAR to POSCAR to save new structure compatible with WAVECAR and CHGCAR
            shutil.copy2("CONTCAR", safe_dir + "POSCAR")
            shutil.copy2("CHGCAR", safe_dir)
            shutil.copy2("WAVECAR", safe_dir)

        os.chdir(self.owd)

        if result == "":
            return structure, energy
        else:
            raise ValueError

    def get_band_path(self, nkpts):
        # This defines the band structures from Setwayan et al
        lattice = self.structure.cell.get_bravais_lattice()
        path = bandpath(str(lattice.special_path), self.structure.cell, npoints=nkpts)
        print(path.kpts)
        return path.kpts

    def self_consistent_hubbard(self, relax=False, species=None, setup_param=None, mag=False):
        """

        This function follows the work of the SI from Curnan and Kitchin

        1. Get the bare intial chi0 and converged chi response matrices
        2. an input U value (Uin) of 0 eV is applied to achieve an output U value (Uout) greater than 0 eV
        3. Uout = χo −1 − χ−1

        The four step scheme needs to be as follows:

        1. Get relaxed structure for primitive and supercell structures
        2. Using previous step, run scf at high precision

        NOTE: the perturbed cation must be distinguished as its own species in the POSCAR and POTCAR files
        3. using ICHARG = 11 and final orbital used to produce the initial response with LDAUTYPE = 3
        and LDAUU and LDAUJ used to vary the quantity of spin-up and down perturbation contributions
        4. scf calculation as in (3) but without using the CHGCAR in the same perturbation range

        NOTE: spin cannot be ignored in these calculations

        calc_seq = ["relax-mag",
         "scf-mag; add_settings={"ediff": "1E-06"},
         "scf-mag-bare"; add_settings={"icharg": 11, "ldautype": 3},
         "scf-mag-"; add_settings={"icharg": 11, "ldautype": 3}]
        :return:
        """

        if species is None:
            species = {"Fe": "d", "Fe": "", "O": ""}
        if setup_param is None:
            setup_param = {1: 'Fe'}

        # hubb_dir is going to store the ldau_luj values
        hubb_dir = {}

        # loop to add all hubbards to hubb_dir
        # TODO: need to adjust this to only put U = 2 for copied species

        for atom in species:

            # chooses which orbitals to put the hubbard on
            if species[atom] == "s":
                orbital = 0
            if species[atom] == "p":
                orbital = 1
            if species[atom] == "d":
                orbital = 2
            if species[atom] == "f":
                orbital = 3
            else:
                orbital = -1

            # select species under scrutiny
            if orbital != -1:
                active_species = atom

            # update hubbard directory
            hubb_dir.update({atom: {'L': orbital, 'U': 0, 'J': 0}})

        # alpha_range defines array of alpha values to be considered
        alpha_range = np.arange(-0.15, 0.15, 0.05)

        """# alter POSCAR so that the hubbard species is separated into two species
        # open POSCAR and read in lines to list
        with open("./POSCAR", "r") as rf:
            lines = rf.readlines()


        # for all line sin list check if the active species is in that line and replace with 2 active species
        for i, line in enumerate(lines):
            if line.find(active_species) != -1:
                lines[i] = line.replace(f"{active_species}", f"{active_species} {active_species}")

        # rewrite whole file
        with open("./POSCAR", "w") as wf:
            wf.writelines(lines)
            
            NB: This commented-out section is now obsolete upon revelation that the set-up keyword splits the POSCAR
            automatically """

        # This explicit setup keyword makes two different Fe species
        setup_settings = {"setups": setup_param}

        # add hubbard settings
        chi_settings = {"ldautype": 3}
        chi_settings.update(setup_settings)

        # run steps 1 and 2
        # need to make the relaxation optional and add in the chi settings
        if relax:
            calc_seq = ["relax", "scf"]
        else:
            calc_seq = ["scf"]

        add_settings = {x: chi_settings for x in calc_seq}
        # run calculations
        self.calc_manager(calc_seq=calc_seq, add_settings_dict={0: chi_settings, 1: chi_settings},
                          mags=None, hubbard_params=hubb_dir, outfile="vasp_seq.out")

        # loop through all requested values of alpha
        for alpha in alpha_range:
            alpha_path = f"./{alpha}"
            if not os.path.exists(alpha_path):
                os.mkdir(alpha_path)
            os.chdir(alpha_path)

            # run steps 3 and 4 for each value of alpha
            steps = ["bare", "inter"]
            for i, step in enumerate(steps):

                # make subdirectory for steps 3 and 4 respectively
                path_name = f"./{step}"
                if not os.path.exists(path_name):
                    os.mkdir(path_name)
                os.chdir(path_name)

                # copy CHGCAR and WAVECAR from scf step

                # update alpha value in hubb_dir
                hubb_dir[active_species].update({'U': alpha, 'J': alpha})
                chi_settings.update(hubb_dir)

                if mag:  # magnetism hasn't worked for MnFe2O4 so far but non-magnetic seems to work quite well
                    mag = "-mag"

                if step == "bare":
                    chi_settings.update({"icharg": 11})

                elif step == "inter":
                    chi_settings.update({"icharg": 1})

                self.single_vasp_calc("scf" + mag, add_settings=chi_settings, path_name="./",
                                      use_safe_file=True)

                os.chdir("../")
            os.chdir("../")

    # TODO: Self-consistent hubbard set-up
    # TODO: Calculation Manager
    # TODO: single calculations for relaxation, scf, bands, eps,
    # TODO: single calculations for HSE06, SCAN, GGA+U,

# MnFe2O4_structure = read("./Cifs/MnFe2O4-Normal.cif")
# MnFe2O4_structure = read("./relax/POSCAR")
# MnFe2O4_calculation = VaspCalculations(MnFe2O4_structure)

# MnFe2O4_calculation.get_band_path(nkpts=200)
# k testing
# k_test = MnFe2O4_tests.parameter_testing("k-points", [1, 2, 3])
# ecut testing
# ecut_test = MnFe2O4_tests.parameter_testing("ecut", [400, 450, 500, 550, 600, 650, 700, 750, 800])
