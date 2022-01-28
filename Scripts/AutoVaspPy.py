import math
import os
import re
import copy
import json
import shutil
import numpy as np
from itertools import product

from ase.io import read
from ase import Atoms
from ase.dft.kpoints import bandpath
from ase.calculators.vasp import Vasp
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.analysis.phase_diagram import PhaseDiagram

from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rcParams
from calculation_settings import *


class VaspCalculations(object):
    def __init__(self, structure: Atoms, write_file="output.out"):
        """

        Args:
            structure: Atoms
                atoms object that defines the system under investigation

            write_file: str
                name of the file to send output


        NB: The general settings are set up for the authors calculations. Please adjust these to your own as you need.

        """

        # calculation settings
        self.general_calculation_settings = general_calculation_settings
        self.calculation_specific_parameters = calculation_specific_parameters

        self.structure = structure

        # directories initialisation
        self.owd = os.getcwd()
        self.safe_dir = self.owd + "/safe"
        self.last_dir = self.safe_dir

        # output writing
        self.write_file = write_file
        self.f = open(self.write_file, "a+")
        self.write_intro()

    def write_intro(self):
        self.f.write(" " + "-" * 78 + " \n")
        self.f.write("|" + " " * 78 + "|\n")
        self.f.write(
            "|    AutoVaspy: A helpful* Python interface to pipeline VASP calculations.    |\n"
        )
        self.f.write("|" + " " * 78 + "|\n")
        self.f.write("|" + " " * 78 + "|\n")
        self.f.write("|" + " " * 78 + "|\n")
        self.f.write("|" + " " * 78 + "|\n")
        self.f.write(
            "|                    *Helpfulness absolutely not guaranteed                    |\n"
        )
        self.f.write(" " + "-" * 78 + " \n")
        self.f.write("\n\n\n\n\n")
        self.f.flush()

    def parameter_testing(
            self,
            test: str,
            values: list,
            additional_settings: dict,
            magnetic_moments: list,
            hubbard_parameters: dict,
            plot=False,
            n_cycles=3,
    ):
        """
        A function for convergence testing of different VASP calculation_specific_parameters


        Args:
            test: str
                Name of test to be conducted. Currently supported are k-points, k-spacing, encut and sigma
            values: list
                A list of values to test
            additional_settings: dict
                A dictionary of additional values to pass through the ase interface for each calculation
            magnetic_moments: list
                A list of magnetic moments associated with the structure
            hubbard_parameters: dict
                A dictionary of hubbard calculation_specific_parameters in ase form for each calculation

        Returns:
            energies: list
                A list of the final energies for each testing value

        """

        owd = os.getcwd()

        # initialise tests directory
        if not os.path.exists(f"{self.owd}/convergence_tests"):
            os.mkdir(f"{self.owd}/convergence_tests")

        self.f.write("-" * 80 + "\n")
        self.f.write(
            "{} testing with values of {}".format(test, values).center(80) + "\n"
        )
        self.f.write("-" * 80 + "\n")

        # define path and check if directory exists
        path_name = f"{self.owd}/convergence_tests/{test}"

        if not os.path.exists(path_name):
            os.mkdir(path_name)

        self.f.write(
            "Testing calculations running from the directory:".center(80) + "\n"
        )
        self.f.write(f"{path_name}".center(80) + "\n")
        self.f.write("-" * 80 + "\n")
        self.f.write(f"{test} Energy".center(80) + "\n")

        # list for storage of tested output
        energies = []

        # iterate over all values
        for test_value in values:
            test_path = f"{path_name}/{test_value}"
            if not os.path.exists(test_path):
                os.mkdir(test_path)
            os.chdir(test_path)

            # statements to find which type of test is being used
            if test == "k-points" or test == "kpts" or test == "nkpts":
                test_variable = "kpts"
                test_value = [test_value, test_value, test_value]
            elif test == "ecut" or test == "encut":
                test_variable = "encut"
            elif test == "hubbard":
                test_variable = "hubbard"
            else:
                test_variable = test

            if not additional_settings:
                additional_settings = {}

            if hubbard_parameters:
                additional_settings.update(
                    self.calculation_specific_parameters["hubbard"]
                )
                additional_settings["ldau_luj"] = hubbard_parameters

            # update keywords dictionary to have new test value
            additional_settings[test_variable] = test_value

            # run calculation
            _, energy = self.single_vasp_calc(
                calculation_type="scf-mag",
                additional_settings=additional_settings,
                magnetic_moments=magnetic_moments,
                max_cycles=n_cycles,
            )

            self.f.write(f"{test_value}, {energy}".center(80) + "\n")
            self.f.flush()

            energies.append(energy)
            os.chdir(owd)

        self.f.write("-" * 80 + "\n")
        self.f.write("Testing Finished!".center(80) + "\n")
        self.f.write("-" * 80 + "\n")
        self.f.flush()

        if plot:
            import matplotlib

            matplotlib.use("Agg")

            x = np.array(values)
            y = np.array(energies)

            plt.plot(x, y)
            plt.xlabel(f"{test}", fontsize=18)
            plt.ylabel("Energy / eV", fontsize=18)
            plt.savefig(f"./convergence_tests/{test}/{test}.png", dpi=300)

        return energies

    def hubbard_testing(
            self,
            values: dict,
            additional_settings: dict,
            magnetic_moments: list,
            hubbard_parameters: dict,
            # TODO: plot to be added
    ):

        # get original directory as root
        owd = os.getcwd()

        # initialise tests directory
        if not os.path.exists(f"{self.owd}/convergence_tests"):
            os.mkdir(f"{self.owd}/convergence_tests")

        self.f.write("-" * 80 + "\n")
        self.f.write(
            "Hubbard testing with values of {}".format(values).center(80) + "\n"
        )
        self.f.write("-" * 80 + "\n")

        # define path and check if directory exists
        path_name = f"{self.owd}/convergence_tests/Hubbard"

        if not os.path.exists(path_name):
            os.mkdir(path_name)

        self.f.write(
            "Testing calculations running from the directory:".center(80) + "\n"
        )
        self.f.write(f"{path_name}".center(80) + "\n")
        self.f.write("-" * 80 + "\n")
        self.f.write(f"Hubbard Energy".center(80) + "\n")

        testing_hubbards = [
            {x: y for x, y in zip(values.keys(), z)}
            for z in list(product(*values.values()))
        ]

        # list for storage of tested output
        energies = []

        for test_set in testing_hubbards:
            test_dict = ""
            for val in test_set.values():
                test_dict += f"{val}_"
            test_dict = test_dict.rstrip("_")

            test_path = f"{path_name}/{test_dict}"
            if not os.path.exists(test_path):
                os.mkdir(test_path)
            os.chdir(test_path)

            if not additional_settings:
                additional_settings = {}

            additional_settings.update(self.calculation_specific_parameters["hubbard"])
            additional_settings["ldau_luj"] = hubbard_parameters

            # update keywords dictionary to have new test value
            for key, value in test_set.items():
                additional_settings["ldau_luj"][key]["U"] = value

            # run calculation
            _, energy = self.single_vasp_calc(
                calculation_type="scf-mag",
                additional_settings=additional_settings,
                magnetic_moments=magnetic_moments,
                max_cycles=1,
            )

            write_data = ""
            for value in test_set.values():
                write_data += f"{value}, "

            self.f.write(f"{write_data}{energy}".center(80) + "\n")
            self.f.flush()

            energies.append(energy)
            os.chdir(owd)

        self.f.write("-" * 80 + "\n")
        self.f.write("Testing Finished!".center(80) + "\n")
        self.f.write("-" * 80 + "\n")
        self.f.flush()

        return energies

    def make_hse_kpoints(self, ibz_file):
        """This function aims to make the KPOINTS file needed for hybrid functional band structures

        The required sequence is as follows:
        1.  standard non-hybrid DFT run
        2.  using converged WAVECAR, hybrid-DFT run
        3.  zero-weight KPOINTS run using IBZKPTS file from (2)

        "Mind: Remove from the band structure plot the eigenvalues corresponding to the regular k-points mesh."
        :return:
        """
        shutil.copy2(ibz_file, "./KPOINTS")
        band_path = self.get_band_path()[0]
        return band_path

    def calculation_manager(
            self,
            calculation_sequence=None,
            additional_settings=None,
            magnetic_moments=None,
            hubbard_parameters=None,
            nkpts=200,
            test_run=False,
            max_cycles=3,
    ):

        """
        A calculation manager that can run a sequence of calculations for a given system.

        Args:
            calculation_sequence: list
                A list of calculations to run. Currently implemented are relax, scf, bands, eps which can be suffixed
                with "-mag" to allow for magnetic calculations

            additional_settings: dict
                A dictionary of dictionaries for extra vasp setting for each calculation stage

            magnetic_moments: list
                The magnetic structure of the system

            hubbard_parameters: dict
                Hubbard calculation_specific_parameters in ase dictionary format

            nkpts: int
                Number of k-points for band structure calculation

            test_run: bool
                If True, a test run with minimal settings is run. Any results from a test run is unlikely to be
                meaningful and should only be used to check for any runtime errors that may occur.

        Returns:
            current_structure: ase atoms

            energy: float
                Final energy of the system
        """

        # default calculation sequence is relaxation and scf
        if calculation_sequence is None:
            raise ValueError(
                "No calculation sequence (calculation_sequence) specified!"
            )

        if additional_settings is None:
            additional_settings = {}

        # deepcopy needed to reset the additional_settings after every calculation sequence
        copy_add_settings_dict = copy.deepcopy(additional_settings)

        # check for test run
        if test_run:
            self.f.write(
                "Test run is selected. This run will use minimal settings to test that transitions"
                "between calculations run smoothly. \n"
            )
            self.f.write(
                "IMPORTANT NOTE!!: Results from test runs are meaningless. Please DO NOT use these results!\n"
            )

        # all output of individual calculations is written to the one file (which note: is always open)
        self.f.write("-" * 20 + "\n")
        self.f.write(28 * " " + "" + "\n")
        self.f.write("\n")
        self.f.write(f"Calculation sequence consists of: {calculation_sequence} \n")

        if copy_add_settings_dict:
            self.f.write(f"Additional settings: {copy_add_settings_dict} \n")
        self.f.write("-" * 20 + "\n")

        current_structure = None
        energy = None

        # loop through all calculations
        for i, calc in enumerate(calculation_sequence):
            self.f.write(
                f"Beginning calculation: {calc} as calculation {i + 1} in sequence. \n"
            )

            # if no additional setting specified for a given calculation, add empty dict
            if calc not in copy_add_settings_dict:
                copy_add_settings_dict[calc] = {}

            # make separate directories for each calculation
            path = f"./{calc}"
            self.f.write(f"file path is {path} \n")

            # check if individual calculation is magnetic and if hubbard calculation_specific_parameters are specified
            if "mag" in calc:
                self.f.write("Calculation is magnetic! \n")
                if not magnetic_moments:
                    self.f.write(
                        "No magnetic moments are specified. Are you sure this is correct? \n"
                    )
                else:
                    self.f.write(f"Initial magnetic moments are: {magnetic_moments} \n")
                mag_moments = magnetic_moments

                # hubbard check
                if not hubbard_parameters:
                    self.f.write(
                        "No hubbard calculation_specific_parameters requested. Are you sure this calculation will "
                        "converge? \n "
                    )
                else:
                    # check if additional_settings already exists and append ldau_luj values
                    self.f.write(
                        f"Hubbard values to be used are as follows: {hubbard_parameters} \n"
                    )

                    copy_add_settings_dict[calc].update(
                        self.calculation_specific_parameters["hubbard"]
                    )
                    copy_add_settings_dict[calc]["ldau_luj"] = hubbard_parameters

            else:
                mag_moments = None

            # determine use of safe files for reading and writing
            if "write_safe_file" in copy_add_settings_dict[calc].keys():
                write_safe_file = copy_add_settings_dict[calc]["write_safe_file"]
                print(f"write: {write_safe_file}")
                del copy_add_settings_dict[calc]["write_safe_file"]
            else:
                write_safe_file = False
                print(f"write: {write_safe_file}")

            if "read_safe_file" in copy_add_settings_dict[calc].keys():
                read_safe_file = copy_add_settings_dict[calc]["read_safe_file"]
                del copy_add_settings_dict[calc]["read_safe_file"]
            else:
                read_safe_file = False

            # run calculation
            current_structure, energy = self.single_vasp_calc(
                calculation_type=calc,
                additional_settings=copy_add_settings_dict[calc],
                path_name=path,
                write_safe_files=write_safe_file,
                read_safe_files=read_safe_file,
                magnetic_moments=mag_moments,
                nkpts=nkpts,
                testing=test_run,
                max_cycles=max_cycles,
            )

        return current_structure, energy

    def single_vasp_calc(
            self,
            calculation_type="scf",
            additional_settings=None,
            path_name="./",
            nkpts=200,
            read_safe_files=False,
            write_safe_files=False,
            magnetic_moments=None,
            testing=False,
            max_cycles=10,
            general_settings="basic",
    ):
        """

        Parameters
        ----------
        calculation_type
        additional_settings
        path_name
        nkpts
        read_safe_files
        write_safe_files
        magnetic_moments
        testing
        max_cycles
        general_settings

        Returns
        -------

        """

        # check if directory already exists and if not change to directory
        if additional_settings is None:
            additional_settings = {}
        if not os.path.exists(path_name):
            os.mkdir(path_name)
        os.chdir(path_name)

        structure = None
        energy = None

        # if safe files to be used copy to cwd
        if read_safe_files:
            shutil.copy2(self.safe_dir + "/POSCAR", "./")
            shutil.copy2(self.safe_dir + "/CHGCAR", "./")
            shutil.copy2(self.safe_dir + "/WAVECAR", "./")

        # rewrite the last_dir in case it is needed later
        self.last_dir = path_name

        # copy vasp settings from standard set-up
        vasp_settings = self.general_calculation_settings[general_settings].copy()

        # check for magnetism
        # can give an explicit list or a string in vasp format that is sorted by multiply_out_moments function
        if magnetic_moments:
            if type(magnetic_moments) == str:
                magnetic_moments = self.multiply_out_moments(magnetic_moments)
            self.structure.set_initial_magnetic_moments(magmoms=magnetic_moments)
            vasp_settings.update({"ispin": 2})

        # update settings for calculation type
        calc_strip = calculation_type.replace("-mag", "").replace("single-", "")
        vasp_settings.update(self.calculation_specific_parameters[calc_strip])

        # check for hybrid
        if calculation_type.find("hse06") != -1:
            vasp_settings.update()

        # add any extra settings
        vasp_settings.update(additional_settings)

        # set testing run
        if testing:
            vasp_settings.update(self.calculation_specific_parameters["test_defaults"])
            if "relax" in calculation_type:
                vasp_settings.update({"nsw": 1})

        # relaxation
        if "relax" in calculation_type and "single" in calculation_type:
            structure, energy, result = self.run_vasp(vasp_settings)

        elif "relax" in calculation_type and "single" not in calculation_type:

            # while loop breaks when a relaxation converges in one ion relaxation step
            converged = False
            steps = 0
            while not converged:
                # Copy CONTCAR to POSCAR for next stage of calculation -- use "copy2()" to copy metadata and permissions
                if os.path.isfile("CONTCAR"):
                    shutil.copy2("POSCAR", "POSCAR-PREVIOUS-RUN")
                    if not os.stat("CONTCAR").st_size == 0:
                        shutil.copy2("CONTCAR", "POSCAR")
                    try:
                        self.structure = read(
                            "./POSCAR"
                        )  # need to read in the new POSCAR after every run for consistency
                    except IndexError:
                        shutil.copy2("POSCAR-PREVIOUS-RUN", "POSCAR")
                        self.structure = read("./POSCAR")

                # run calculation
                structure, energy, result = self.run_vasp(vasp_settings)

                if result == "vasp failure":
                    break

                if self.converged_in_n_scf_cycles("OUTCAR"):
                    break
                steps += 1

                # break if max number of cycles are reached
                if steps >= max_cycles:
                    break

                self.structure.set_initial_magnetic_moments(
                    magmoms=self.structure.get_magnetic_moments()
                )

        else:
            # if band structure type calculation the get_band_path function for the k-point path
            if "bands" in calculation_type:
                vasp_settings.update(kpts=self.get_band_path(nkpts=nkpts)[0])

            # run energy calculation
            structure, energy, result = self.run_vasp(vasp_settings)

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

    def run_vasp(self, vasp_settings, restart=False):
        """
        Run a single VASP calculation using an ASE calculator.
        This routine will make use of WAVECAR and CONTCAR files if they are available.
        Parameters
        ----------

        vasp_settings : dict
            The set of VASP options to apply with their values.
        restart : bool
            Determines if a calculation is to be restarted

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
        """

        # Set files for calculation
        energy = 0.0
        result = ""

        # Use ASE calculator -- the use of **kwargs in the function call allows us to set the desired arguments using a
        # dictionary
        structure = self.structure
        structure.set_calculator(Vasp(**vasp_settings, restart=restart))

        # Run calculation, and consider any exception to be a VASP failure
        # NOTE: you can get a Value error here from a function (ironically) calculating stress due to large stress
        # Make sure your cutoff energy is large enough
        try:
            energy = structure.get_potential_energy()

        except UnboundLocalError:
            print("UnboundLocalError found so trying to turn off symmetry")
            try:
                vasp_settings["isym"] = 0
                structure.set_calculator(Vasp(**vasp_settings, restart=restart))
                energy = structure.get_potential_energy()
            except ValueError:
                result = "vasp failure"

        except IndexError:
            try:
                vasp_settings["isym"] = 0
                structure.set_calculator(Vasp(**vasp_settings, restart=restart))
                energy = structure.get_potential_energy()
            except ValueError:
                result = "vasp failure"

        except ValueError:
            print(
                "String-to-float error implies that energy has blown up. Please check settings!"
            )

        return structure, energy, result

    @staticmethod
    def batch_calculations(self, directories, constants, add_settings_dict=None, *args):
        """

        A method for running many different versions of similar calculations

        Options may include:

            • Constant structure with different magnetic structures
            • Constant crystal structure with swapping/adding/deleting atoms
            • Constant calculations calculation_specific_parameters for different structures

        What also needs to be included is:

            • Ways of comparing calculations (by final energy/structure/dos etc.)
                • This could be formed as other functions


        Assume everything changes unless specified in constants
        """

        num_of_calculations = 10

        # loop over all required directories
        for direct in directories:
            os.chdir(direct)

    def get_band_path(self, nkpts=50):
        # This defines the band structures from Setwayan et al

        """# get structure
        if "xml" in file:
            k_structure = Vasprun(file).final_structure
        elif "POSCAR" in file:
            k_structure = Poscar.from_file(file).structure
        else:
            raise TypeError("File type must be POSCAR or vasprun.xml")"""
        # This defines the band structures from Setwayan et al

        path = bandpath(path=None, cell=self.structure.cell, npoints=nkpts, eps=1e-3)
        return path.kpts, path.path

    @staticmethod
    def plot_band_path(self, file="vasprun.xml", path="./", eng="0", reference=0.0):
        read_obj = read(path + file)

    def self_consistent_hubbard(
            self, relax=False, species=None, setup_param=None, mag=False, ignore_scf=False
    ):
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

        calculation_sequence = ["relax-mag",
         "scf-mag; additional_settings={"ediff": "1E-06"},
         "scf-mag-bare"; additional_settings={"icharg": 11, "ldautype": 3},
         "scf-mag-"; additional_settings={"icharg": 11, "ldautype": 3}]
        :return:
        """

        if species is None:
            species = {"Fe": "d", "O": ""}
        if setup_param is None:
            setup_param = {1: "Fe"}

        # hubb_dir is going to store the ldau_luj values
        hubb_dir = {}

        # loop to add all hubbard_parameters to hubb_dir
        # TODO: need to adjust this to only put U = 2 for copied species

        for atom in species:

            # chooses which orbitals to put the hubbard on
            if species[atom] == "s":
                orbital = 0
            elif species[atom] == "p":
                orbital = 1
            elif species[atom] == "d":
                orbital = 2
            elif species[atom] == "f":
                orbital = 3
            else:
                orbital = -1

            # select species under scrutiny
            if orbital != -1:
                active_species = atom

            # update hubbard directory
            hubb_dir.update({atom: {"L": orbital, "U": 0, "J": 0}})

        if not active_species:
            raise ValueError("No active species specified!")

        # alpha_range defines array of alpha values to be considered
        alpha_range = np.arange(-0.15, 0.15, 0.05)

        # This explicit setup keyword makes two different Fe species
        setup_settings = {"setups": setup_param}

        # add hubbard settings
        chi_settings = {"ldautype": 3, "icharg": 2, "istart": 0}
        chi_settings.update(setup_settings)

        # run steps 1 and 2
        # need to make the relaxation optional and add in the chi settings
        if relax:
            calc_seq = ["relax", "scf"]
        else:
            calc_seq = ["scf"]

        add_settings = {x: chi_settings for x in calc_seq}

        # run calculations
        if not ignore_scf:
            self.calculation_manager(
                calculation_sequence=calc_seq,
                additional_settings=add_settings,
                magnetic_moments=None,
                hubbard_parameters=hubb_dir,
            )

        # loop through all requested values of alpha
        for alpha in alpha_range:
            alpha_path = f"./alpha={alpha}"
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
                hubb_dir[active_species].update({"U": alpha, "J": alpha})
                chi_settings.update(hubb_dir)

                if (
                        mag
                ):  # magnetism hasn't worked for MnFe2O4 so far but non-magnetic seems to work quite well
                    mag = "-mag"

                if step == "bare":
                    chi_settings.update({"icharg": 11})

                elif step == "inter":
                    chi_settings.update({"icharg": 1})

                self.single_vasp_calc(
                    "scf" + mag,
                    additional_settings=chi_settings,
                    path_name="./",
                    write_safe_file=True,
                )

                os.chdir("../")
            os.chdir("../")

    @staticmethod
    def converged_in_n_scf_cycles(outcar_file):
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

    @staticmethod
    def multiply_out_moments(moms=str):
        """
        An overly complicated way of taking a string of magnetic moments and multiplying
        the value out to give a list of individual magnetic moments
        Parameters
        ----------
        moms

        Returns
        -------

        """
        # clear any whitespace and make into list
        list_moments = moms.replace(" ", "").split(",")

        mag_moms = []
        for mom in list_moments:
            if mom.find("*") != -1:
                # split by multiplication sign
                factors = mom.split("*")

                # add to mag moments list each moment x amount of times
                mag_moms.extend(int(factors[1]) for i in range(int(factors[0])))
            else:
                mag_moms.append(int(mom))

        # return flattened list of moments
        return mag_moms

    # TODO: Self-consistent hubbard set-up -- Needs Finished
    # TODO: band plot
    # TODO: DOS plot
    # TODO: EPS plot


def get_convex_hull_species(
        key="0G4rqjSNG4M51Am0JNj",
        species=None,
        must_contain=None,
        filepath="./",
        elim=0.025,
):
    """

    Args:
        key:
        species:
        filepath:
        elim:
    """
    if must_contain is None:
        must_contain = []
    if species is None:
        species = []
    if not species:
        species = ["Mn", "Fe", "O"]

    with MPRester(key) as m:
        mp_entries = m.get_entries_in_chemsys(species, compatible_only=False)

        analyser = PhaseDiagram(mp_entries)
        # e = analyser.get_e_above_hull(mp_entries[0])

        if not os.path.exists(filepath):
            os.mkdir(filepath)
        os.chdir(filepath)
        owd = os.getcwd()

        for entry in mp_entries:

            if must_contain:
                if all(
                        species in entry.composition.alphabetical_formula
                        for species in must_contain
                ):
                    get_entry = True
                else:
                    get_entry = False
            else:
                get_entry = True

            if analyser.get_e_above_hull(entry) < elim and get_entry:

                formula_dir = (
                    f"./{entry.composition.alphabetical_formula.replace(' ', '')}"
                )

                if os.path.exists(formula_dir):
                    os.chdir(formula_dir)
                    if not os.path.exists(f"./{entry.entry_id}"):
                        os.mkdir(entry.entry_id)
                    os.chdir(entry.entry_id)
                else:
                    os.mkdir(formula_dir)
                    os.chdir(formula_dir)
                    if not os.path.exists(f"./{entry.entry_id}"):
                        os.mkdir(entry.entry_id)
                    os.chdir(entry.entry_id)

                with open("energy", "w") as wf:
                    wf.write(f"entry id = {entry.entry_id}")
                    wf.write(f"E above hull = {analyser.get_e_above_hull(entry)}")

                structure = m.get_structure_by_material_id(entry.entry_id)
                structure.to(fmt="poscar", filename="POSCAR")

                os.chdir(owd)


def ch_sorter(filepath):
    os.chdir(filepath)
    subdirectories = [f.path.strip("./") for f in os.scandir("./") if f.is_dir()]
    for sub in subdirectories:
        x = re.split("[0-9]", sub)[:-1]
        filtered_x = sorted(list(filter(None, x)))
        dir_name = ""
        for i in filtered_x:
            dir_name += i
            dir_name += "-"

        dir_name = dir_name.strip("-")

        if dir_name == "":
            print(f"Directory name is empty for {sub}")
            continue

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        print(f"Moving {sub} to {dir_name}")
        shutil.move(sub, dir_name)


def converged_in_n_scf_cycles(outcar_file, n_steps=1):
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

    converged = (
            num_convergence_strings > 0
            and num_ionic_loops <= n_steps
            and num_ionic_loops != 0
    )

    return converged


def convex_hull_relaxations(
        species: dict,
        root_directory="./",
        hubbards=None,
        additional_settings=None,
        testing=False,
        ignore_dirs=None,
):
    """
    directory nesting follows the following pattern:

        Species
        └── Polymorphs
            └── Magnetic Structures

        e.g.

        Mn2Fe4O8
        ├── Normal
        │   └── FM
        │   └── AFM
        └── Inverse
            └── FM
            └── AFM


    Args:
        species (dict):
            A dictionary of each material as a key and value of the magnetic moments of a single magnetic
            structure as a list or a dictionary of lists
        root_directory (str):
            Location of convex hull species subdirectories
        hubbards (dict):
            A dictionary of hubbard calculation_specific_parameters for each atom in the form required for the ase python
            module (see https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#ld-s-a-u)
        additional_settings (dict):
            A dictionary of settings to be used for all systems in the convex hull
        testing (bool):
            If true, will run function without running any DFT calculations to test for errors. If no errors returned,
            all convex hull species should run correctly (assuming no DFT errors or failures)
        ignore_dirs (list):
            Names of directories to ignore if encountered. Useful for ignoring magnetic structures (i.e. "./FiM") from
            being rerun

    Returns:
        None
    """

    # ignore_dirs stops repeated calculations
    if ignore_dirs is None:
        ignore_dirs = []

    # change to location
    os.chdir(root_directory)
    owd = os.getcwd()

    # for each chemical system in composition space
    for system in species.keys():
        system_directory = os.getcwd()

        os.chdir(f"{system}")

        # gets all immediate subdirectories which most often are as mp codes but could be anything
        subdirectories = [
            f.path
            for f in os.scandir("./")
            if f.is_dir() and f.path.lstrip("./") not in ignore_dirs
        ]

        # if there are no subdirectories then just run calculations in the current structure
        # note that if magnetic calculations are required, there still has to be a subdirectory to change into
        if not subdirectories:
            subdirectories = ["./"]

        # for each polymorph
        for subdirectory in subdirectories:

            calculation_directory = os.getcwd()
            print("-" * 50)
            print(
                f"Running relaxation calculation for {system} in {calculation_directory}"
            )
            print(f"Changing directory to {calculation_directory}")
            print("-" * 50)

            # TODO: is this actually necessary for any purpose?
            if not os.path.exists(subdirectory):
                os.mkdir(subdirectory)
            os.chdir(subdirectory)

            # if converged CONTCAR found, calculation does not need rerun and can continue
            if os.path.exists("./relax-mag/OUTCAR"):
                if converged_in_n_scf_cycles(
                        "./relax-max/OUTCAR", 3
                ) and os.path.exists("./relax-mag/vasprun.xml"):
                    print("Converged OUTCAR and vasprun.xml found! Skipping system.")
                    continue

            # read poscar from the subdirectory as any children from here all have the same structures
            initial_poscar = read("./POSCAR")  # TODO: add try-except here?

            calculator = VaspCalculations(
                initial_poscar, write_file=f"{owd}/convex_hull.out"
            )

            # if only one magnetic configuration specified
            if type(species[system]) == list:

                print(
                    f"For {subdirectory} in system {system} a list is given for magnetic moments and therefore\n"
                    f"it is assumed that only one magnetic structure is to be tested"
                )
                if not testing:
                    relax = calculator.calculation_manager(
                        calculation_sequence=["relax-mag"],
                        additional_settings={"relax-mag": additional_settings},
                        magnetic_moments=species[system],
                        hubbard_parameters=hubbards,
                    )

            # elif multiple magnetic configurations specified
            elif type(species[system]) == dict:
                print(
                    f"For {subdirectory} in system {system} a dictionary is given for magnetic moments and therefore\n"
                    f"it is assumed that {len(species[system])} magnetic structures are to be tested"
                )

                for key, magnetic_configuration in zip(
                        species[system].keys(), species[system].values()
                ):

                    if os.path.exists(f"./{key}"):
                        continue

                    os.mkdir(f"./{key}")
                    os.chdir(f"./{key}")

                    if not testing:
                        relax = calculator.calculation_manager(
                            calculation_sequence=["relax-mag"],
                            additional_settings={"relax-mag": additional_settings},
                            magnetic_moments=magnetic_configuration,
                            hubbard_parameters=hubbards,
                        )
                    os.chdir(calculation_directory)

            # elif not magnetic
            elif species[system] is None:
                print(
                    f"For {subdirectory} in system {system} no magnetic moments are specified and therefore\n"
                    f"it is assumed that the calculation is non magnetic"
                )
                if not testing:
                    relax = calculator.calculation_manager(
                        calculation_sequence=["relax"],
                        additional_settings={"relax": additional_settings},
                    )
            else:
                raise TypeError("Wrong type for magnetic moments specified!")

            os.chdir(calculation_directory)
        os.chdir(system_directory)


def get_magnetic_moments_from_mp(root="./", write_file="./magnetic_moments"):
    """
    Fetches all magnetic moments of species in a convex hull and writes these as a dictionary to a file

    Args:
        root (str):
            The location of convex hull directories
        write_file (str):
            File to write the magnetic moments dictionary to

    Returns:
        mags:
    """
    os.chdir(root)
    mp_keys = [
        f[0].split("/")[-1] for f in os.walk("./") if not f[0].split("/")[-1].find("mp")
    ]

    # get_ch(species=["Ni", "Mn", "O", "Fe", "Cr"], filepath="./")
    # ch_sorter(filepath="./")
    mags = {}
    with MPRester("0G4rqjSNG4M51Am0JNj") as m:
        for key in mp_keys:
            material = m.get_structure_by_material_id(key)
            moments = MPRelaxSet(material)
            moments = [round(mom) for mom in moments.incar["MAGMOM"]]
            # with open(f"{material.composition}/moments", "w") as wf:
            #    wf.write(moments.incar["MAGMOM"])
            print(f"{material.formula} has mag moms {moments}")
            mags[
                f"{material.composition.alphabetical_formula.replace(' ', '')}"
            ] = moments

    with open(write_file, "w") as wf:
        wf.write("moments = " + json.dumps(mags))

    return mags


def plot_convex_hull(
        read_files=None,
        load_files=None,
        save_data=False,
        convex_hull_loc="./convex_hull.txt",
        corrected=True,
        other_entries=None,
        new_entries_file=None,
):
    """

    Args:
        read_files (list): a list of directories to search for xml files
        load_data (bool): load data from load files
        new_data (bool): look for new data in directories
        load_files:
        save_data:
        convex_hull_loc:

    Returns:

    """
    if other_entries is None:
        other_entries = []
    from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
    from pymatgen.apps.borg.queen import BorgQueen
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
    from pymatgen.entries.compatibility import (
        MaterialsProjectCompatibility,
        MaterialsProject2020Compatibility,
    )
    import os

    loaded_entries = []
    new_entries = []
    additional_entries = []

    if load_files is not None:
        drone = VaspToComputedEntryDrone()
        queen = BorgQueen(drone)

        # load data to queen
        for file in load_files:
            try:
                queen.load_data(file)
            except FileNotFoundError:
                continue

            # get data to entries
            loaded_entries += queen.get_data()

    if read_files is not None:
        drone = VaspToComputedEntryDrone()

        # load data to queen
        for directory in read_files:
            queen = BorgQueen(drone, directory)
            if save_data:
                queen.save_data(directory + "/convex_hull_data")
            new_entries += queen.get_data()

    if other_entries:
        for entry in other_entries:
            additional_entries.append(entry)

    # compound all entries
    all_entries = loaded_entries + new_entries + additional_entries
    final_entries = []

    # Generate and plot phase diagram
    for i in range(len(all_entries)):
        if all_entries[i] is not None:
            final_entries.append(all_entries[i])

    print(final_entries)

    if corrected:
        # corrected energies
        mp_compat = MaterialsProjectCompatibility()
        final_entries = mp_compat.process_entries(final_entries)

        # for i, entry in enumerate(pd.all_entries):
        #    pd.all_entries[i].correction = sum([j.value for j in mp_compat.get_adjustments(entry)])

    pd = PhaseDiagram(final_entries)

    # make decompositions dictionary to show decomposition products
    energies = {}
    decompositions = {}
    if convex_hull_loc:
        with open(convex_hull_loc, "w") as wf:
            for i, entry in enumerate(pd.all_entries):
                wf.write(
                    f"{entry.composition.alphabetical_formula}-{i}, {pd.get_e_above_hull(entry) * 1000:.2f}\n"
                )
                energies[entry.composition.alphabetical_formula + f"-{i}"] = (
                        pd.get_e_above_hull(entry) * 1000
                )
                try:
                    decompositions[
                        entry.composition.alphabetical_formula + f"-{i}"
                        ] = pd.get_decomposition(entry.composition)
                except RuntimeError:
                    continue

    if new_entries_file:
        with open(new_entries_file, "w") as wf2:
            list_new_entries = [x for x in pd.all_entries if x in new_entries]
            for entry in list_new_entries:
                wf2.write(
                    f"{entry.composition.alphabetical_formula}, "
                    f"{pd.get_e_above_hull(entry) * 1000:.2f}\n"
                )

    return pd, decompositions


def get_reflectivity(epsilon, energy_ev):
    """
    Calculate reflectivity (at normal incidence) from the dielectric function.

    :param epsilon: complex numpy array with the dielectric function (real_part + imaginary_part*j)
    :return reflectivity: numpy array with the reflectivity
    """

    epsilon_re = epsilon.real
    epsilon_im = epsilon.imag
    norm_epsilon = np.sqrt(epsilon_re ** 2 + epsilon_im ** 2)

    refractive_index = np.sqrt((epsilon_re + norm_epsilon) / 2.0)
    ext_coeff = np.sqrt((-epsilon_re + norm_epsilon) / 2.0)
    reflectivity = ((refractive_index - 1.0) ** 2 + ext_coeff ** 2) / (
            (refractive_index + 1.0) ** 2 + ext_coeff ** 2
    )
    absorption = (
            (energy_ev * epsilon_im / refractive_index) / 1.9746 * 10.0e-7
    )  # it is equivalent to 2.0*energies*extint_coeff/c or 4*pi*extint_coeff/lambda
    return reflectivity


def plot_reflectivity(
        energy_ev,
        reflectivity,
        save_file=None,
        colour=(0, 0, 0),
        dark_theme=True,
        x_axis="nm",
):
    wavelength_nm = 1239.8 / energy_ev

    # extract the visible part of the reflectivity curve
    wavelength_visible = []
    reflectivity_visible = []
    energy_visible = []
    for i in range(len(wavelength_nm)):
        if 380.0 <= wavelength_nm[i] <= 780.0:
            wavelength_visible.append(wavelength_nm[i])
            reflectivity_visible.append(reflectivity[i])
            energy_visible.append(energy_ev[i])

    fig = plt.figure(figsize=(8, 6))
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

    if x_axis == "nm":
        plt.plot(wavelength_visible, reflectivity_visible, color=colour, linewidth=2.0)
        # plt.scatter(wavelength_visible, reflectivity_visible, color=colour, linewidth=2.0)
        # plt.scatter(wavelength_visible, reflectivity_visible, color=colour, linewidth=2.0)
        plt.xlabel(r"Wavelength / nm", size=18)
    elif x_axis == "eV":
        plt.plot(energy_ev, reflectivity, color=colour, linewidth=2.0)
        # plt.scatter(wavelength_visible, reflectivity_visible, color=colour, linewidth=2.0)
        plt.xlabel(r"Energy / eV", size=18)

    ax = plt.gca()
    # set dark color
    if dark_theme:
        ax.set_facecolor((38 / 255, 38 / 255, 38 / 255))
    else:
        ax.set_facecolor((255 / 255, 255 / 255, 255 / 255))

    plt.ylabel("R($\lambda$)", size=18)
    # plt.xlim(380.0,780.0)
    # change axes tick labels size
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    plt.title(r"Reflectivity", size=24)
    # legend and tight layout
    plt.legend(fontsize=18, fancybox=True, framealpha=0)
    if save_file:
        plt.savefig(save_file)
    return fig


def get_colour_dict(
        energy_ev,
        reflectivity,
        file_d65illuminant,
        file_cmf,
        wavelength_steps="1nm",
        do_plot=True,
        dark_theme=True,
):
    """
    Calculate colour coordinates from reflectivity.

    :param energy_ev: array with energy steps (must be in eV)
    :param reflectivity: array with reflectivity values corresponding to 'energy_ev'
    :param file_d65illuminant: file with CIE standard illuminant D65. Must contain two columns:
        wavelengths + D65 data
    :param file_cmf: file with Color Matching Functions (CMFs). Must contain four columns:
        wavelengths + CMF_x + CMF_y + CMF_z
    :param wavelength_steps: can be '1nm' (default) or '5nm'. These are the supported wavelength steps
        for data in 'file_d65illuminant' and 'file_cmf'.
    :return a dictionary of the form
        {'Tristimulus values': CIE-XYZ,
        'Chromaticity coordinates': CIE-xyY,
        'CIELAB': CIE-L*a*b*,
        'Yellowness index': D1925 yellowness index,
        'sRGB': standard RGB,
        'HEX': hexadecimals,
        }
    """

    data = np.genfromtxt(file_d65illuminant)
    d65_illuminant = data[:, 1]
    data = np.genfromtxt(file_cmf)
    cmf_x = data[:, 1]
    cmf_y = data[:, 2]
    cmf_z = data[:, 3]

    if len(d65_illuminant) != len(cmf_x):
        raise Exception("D65 and CMFs data do not have the same lenght")

    if len(energy_ev) != len(reflectivity):
        raise Exception("'energy_ev' and 'reflectivity' do not have the same lenght")

    wavelength_nm = 1239.8 / energy_ev

    # extract the visible part of the reflectivity curve
    wavelength_visible = []
    reflectivity_visible = []
    for i in range(len(wavelength_nm)):
        if 380.0 <= wavelength_nm[i] <= 780.0:
            wavelength_visible.append(wavelength_nm[i])
            reflectivity_visible.append(reflectivity[i])
    # Polynomial fit of reflectivity_visible
    fit_params, fit_residuals, _, __, ___ = np.polyfit(
        wavelength_visible, reflectivity_visible, 11, full=True
    )
    polynomial = np.poly1d(fit_params)

    wavelength_uniform = []
    reflectivity_fit = []
    if wavelength_steps == "1nm":
        for i in range(380, 781):
            wavelength_uniform.append(i)
            reflectivity_fit.append(polynomial(i))
    elif wavelength_steps == "5nm":
        for i in range(380, 781, 5):
            wavelength_uniform.append(i)
            reflectivity_fit.append(polynomial(i))
    else:
        raise Exception("Supported 'wavelength_steps' are only '1nm' and '5nm'")

    wavelength_uniform = np.array(wavelength_uniform)
    reflectivity_fit = np.array(reflectivity_fit)

    if len(reflectivity_fit) != len(d65_illuminant):
        raise Exception(
            "Fit of reflectivity in the visible range do not have the same lenght of D65 and CMFs data"
        )

    # Calculate renormalization constant k
    k = 100.0 / sum(cmf_y * d65_illuminant)

    # Calculate Xn, Yn and Zn  (Yn=100)
    x_n = k * sum(cmf_x * d65_illuminant)
    y_n = k * sum(cmf_y * d65_illuminant)
    z_n = k * sum(cmf_z * d65_illuminant)

    # Tristimulus values
    x = k * sum(cmf_x * reflectivity_fit * d65_illuminant)
    y = k * sum(cmf_y * reflectivity_fit * d65_illuminant)
    z = k * sum(cmf_z * reflectivity_fit * d65_illuminant)

    # D1925 yellowness index (for D65 illuminant and 1931 CIE observer)
    d1925_index = 100.0 * (1.2985 * x - 1.1335 * z) / y

    # Chromaticity coordinates
    x_chrom = x / (x + y + z)
    y_chrom = y / (x + y + z)
    z_chrom = z / (x + y + z)

    # Calculate L,a,b of CIELAB
    def f(x):
        if x > (24.0 / 116.0) ** 3:
            return x ** (1.0 / 3.0)
        elif x <= (24.0 / 116.0) ** 3:
            return (841.0 / 108.0) * x + (16.0 / 116.0)

    l = 116.0 * f(y / y_n) - 16.0
    a = 500.0 * (f(x / x_n) - f(y / y_n))
    b = 200.0 * (f(y / y_n) - f(z / z_n))

    # Calculate chroma and hue
    chroma = math.sqrt(a ** 2 + b ** 2)
    hue = math.degrees(math.atan(b / a))

    # Standard RGB (sRGB)
    var_X = x / 100.0
    var_Y = y / 100.0
    var_Z = z / 100.0

    var_R = var_X * 3.2406 + var_Y * (-1.5372) + var_Z * (-0.4986)
    var_G = var_X * -0.9689 + var_Y * 1.8758 + var_Z * 0.0415
    var_B = var_X * 0.0557 + var_Y * (-0.2040) + var_Z * 1.0570

    if var_R > 0.0031308:
        var_R = 1.055 * (var_R ** (1.0 / 2.4)) - 0.055
    else:
        var_R = 12.92 * var_R
    if var_G > 0.0031308:
        var_G = 1.055 * (var_G ** (1.0 / 2.4)) - 0.055
    else:
        var_G = 12.92 * var_G
    if var_B > 0.0031308:
        var_B = 1.055 * (var_B ** (1.0 / 2.4)) - 0.055
    else:
        var_B = 12.92 * var_B

    sR = var_R * 255.0
    sG = var_G * 255.0
    sB = var_B * 255.0

    if sR < 0:
        sR = 0
    elif sR > 255:
        sR = 255

    if sG < 0:
        sG = 0
    elif sG > 255:
        sG = 255

    if sB < 0:
        sB = 0
    elif sB > 255:
        sB = 255

    # Hexadecimal colour
    def clamp(x):
        return max(0, min(x, 255))  # to ensure that  0 < sR,sG,sB < 255

    if do_plot:
        plt.figure(figsize=(8, 8))
        rectangle_GGA = plt.Rectangle((0, 0), 50, 20, fc=[sR / 255, sG / 255, sB / 255])
        plt.gca().add_patch(rectangle_GGA)
        plt.axis("off")

    return {
        "Colours": {
            "Tristimulus values": {"X": x, "Y": y, "Z": z},
            "Chromaticity coordinates": {
                "x": x_chrom,
                "y": y_chrom,
                "z": z_chrom,
            },
            "CIELAB": {"L": l, "a": a, "b": b, "Chroma": chroma, "Hue": hue},
            "Yellowness index": d1925_index,
            "sRGB": {"R": sR, "G": sG, "B": sB},
        },
        "Fit_residuals": fit_residuals,
    }


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
    eps = real_avg + imag_avg * 1j
    return eps


def get_metallic_reflectivity(
        drude_parameters, eps_im_inter, eps_re_inter, energy_eV, formula, broadening=0.1
):
    """
    Calculate the optical constants
    :param parameters: ParameterData with input parameters
                        {'intra_broadening': [eV]}
    :param drude_parameters: output_parameters of a ShirleyCalculation
    :param eps_im_inter: XyData from ShirleyCalculation
    :param eps_re_inter: XyData from ShirleyCalculation
    :param formula: Name of the compound used in the files names
    :return: write to file optical constants
    """
    intra_broadening = broadening

    drude_plasma_freq_x = drude_parameters[0] ** 0.5
    drude_plasma_freq_y = drude_parameters[1] ** 0.5
    drude_plasma_freq_z = drude_parameters[2] ** 0.5

    energies = energy_eV

    # eps interband (imaginary)
    eps_im_inter_x = eps_im_inter[:, 0]
    eps_im_inter_y = eps_im_inter[:, 1]
    eps_im_inter_z = eps_im_inter[:, 2]

    # eps interband (real)
    eps_re_inter_x = eps_re_inter[:, 0]
    eps_re_inter_y = eps_re_inter[:, 1]
    eps_re_inter_z = eps_re_inter[:, 2]

    # eps intraband
    eps_re_intra_x = drude_plasma_freq_x ** 2 / (energies ** 2 + intra_broadening ** 2)
    eps_im_intra_x = (
            drude_plasma_freq_x ** 2
            * intra_broadening
            / (energies * (energies ** 2 + intra_broadening ** 2))
    )
    eps_re_intra_y = drude_plasma_freq_y ** 2 / (energies ** 2 + intra_broadening ** 2)
    eps_im_intra_y = (
            drude_plasma_freq_y ** 2
            * intra_broadening
            / (energies * (energies ** 2 + intra_broadening ** 2))
    )
    eps_re_intra_z = drude_plasma_freq_z ** 2 / (energies ** 2 + intra_broadening ** 2)
    eps_im_intra_z = (
            drude_plasma_freq_z ** 2
            * intra_broadening
            / (energies * (energies ** 2 + intra_broadening ** 2))
    )

    # eps interband+intraband
    eps_re_x = eps_re_inter_x - eps_re_intra_x
    eps_im_x = eps_im_inter_x + eps_im_intra_x
    eps_re_y = eps_re_inter_y - eps_re_intra_y
    eps_im_y = eps_im_inter_y + eps_im_intra_y
    eps_re_z = eps_re_inter_z - eps_re_intra_z
    eps_im_z = eps_im_inter_z + eps_im_intra_z

    # eps
    eps_x = eps_re_x + 1j * eps_im_x
    eps_y = eps_re_y + 1j * eps_im_y
    eps_z = eps_re_z + 1j * eps_im_z

    # eps average x,y,z
    eps = (eps_x + eps_y + eps_z) / 3.0

    # The following quantities are all averaged along x,y,z
    norm_epsilon = np.sqrt(eps.real ** 2 + eps.imag ** 2)
    refractive_index = np.sqrt((eps.real + norm_epsilon) / 2.0)
    extint_coeff = np.sqrt((-eps.real + norm_epsilon) / 2.0)
    reflectivity = ((refractive_index - 1.0) ** 2 + extint_coeff ** 2) / (
            (refractive_index + 1.0) ** 2 + extint_coeff ** 2
    )
    absorption = (
            (energies * eps.imag / refractive_index) / 1.9746 * 10.0e-7
    )  # it is equivalent to 2.0*energies*extint_coeff/c or 4*pi*extint_coeff/lambda
    conductivity = (
            1j * energies / (4.0 * np.pi) * (1 - (eps.real + 1j * eps.imag))
    )  # conductivity = 1j*energies/(4.*np.pi)*(1 - eps)

    # Reflectivity as a function of the wavelength
    wavelengths_nm = 1239.8 / energies  # in nm

    # Drude plasma frequency averaged over the three Cartesian directions x, y, z
    drude_avg = math.sqrt(
        (drude_plasma_freq_x ** 2 + drude_plasma_freq_x ** 2 + drude_plasma_freq_z ** 2)
        / 3.0
    )

    # Dielectric function (interband contribution) averaged over the three Cartesian directions x, y, z
    eps_im_inter_avg = (eps_im_inter_x + eps_im_inter_y + eps_im_inter_z) / 3.0
    eps_re_inter_avg = (eps_re_inter_x + eps_re_inter_y + eps_re_inter_z) / 3.0

    return wavelengths_nm, reflectivity, eps
