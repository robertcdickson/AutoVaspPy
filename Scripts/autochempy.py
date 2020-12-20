from ase.calculators.vasp import Vasp
from ase.io import read
import os
import shutil
import numpy as np
from ase.dft.kpoints import bandpath
import copy
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
import re
import environs
import json
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.analysis.phase_diagram import PhaseDiagram

# TODO: Run directly from here to cluster and oxenhope2

# --------------------------------------------------------#
#  This module aims to be able to automate calculations  #
#  for my PhD project                                    #
# --------------------------------------------------------#

class VaspCalculations(object):
    """

    VaspCalculations class provides and python interface to interact with vasp calculations

    """

    def __init__(self, structure, tests=None, write_file="output.out"):
        """

        Args:
            structure (ase atoms object): structure object that defines system
            tests:  (list): list of convergence tests to be performed
            output_file:
            write_file:
        """
        self.general_calculation = {"reciprocal": True,
                                    "ediff": 10E-5,
                                    "xc": "PBE",
                                    "setups": 'materialsproject',
                                    "encut": 520,
                                    "kpts": [6, 6, 6],
                                    "ismear": 0,
                                    "sigma": 0.05,
                                    "prec": 'accurate',
                                    "lorbit": 11,
                                    "lasph": True,
                                    "nelmin": 6,
                                    "nelm": 30,
                                    }

        self.parameters = {
            "relax": {"nsw": 12,  # number of ionic steps
                      "isif": 3,  # allows for atomic positions, cell shape and cell volume as degrees of freedom
                      "ibrion": 1},

            "scf": {"icharg": 1,
                    "istart": 0},

            "bands": {"icharg": 11,
                      "istart": 0},

            "eps": {"algo": "all",
                    "loptics": True,
                    "cshift": 0.1,
                    "nedos": 2000},

            "hse06": {"xc": "HSE06"},

            "hubbard": {"ldau": True,
                        "ldauprint": 2,
                        "lmaxmix": 4},

            "test_defaults": {"kpts": [2, 2, 2],
                              "encut": 50,
                              "nelmin": 0,
                              "nelm": 1,
                              "algo": "VeryFast"}
        }

        self.structure = structure
        self.tests = tests

        self.owd = os.getcwd()
        self.safe_dir = self.owd + "/safe"
        self.last_dir = self.safe_dir

        self.write_file = write_file
        self.f = open(self.write_file, "w+")

    def parameter_testing(self, test, values, add_settings, mags, hubbards):
        # ------------------------------------------------------------------ #
        #  This function allows for testing of many different values for a   #
        #                          given test                                #
        # ------------------------------------------------------------------ #

        # TODO: Make option to plot and save figure of convergence test
        # TODO: At the moment, output is only written at end of all calculations as with open statement ends
        #   This needs to be amended so output is written as each calculation finishes

        if not os.path.exists("./tests"):
            os.mkdir("./tests")
        os.chdir("./tests")

        self.f.write("-" * 80 + "\n")
        self.f.write("{} testing with a list of values of {}\n".format(test, values))

        # define path and check if directory exists
        path_name = "./{}".format(test)

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
            if not add_settings:
                add_settings = {}
            if hubbards:
                add_settings.update(self.parameters["hubbard"])
                add_settings["ldau_luj"] = hubbards

            # update keywords dictionary to have new test value
            add_settings[test_variable] = test_value

            # run calculation

            self.single_vasp_calc(calculation_type="scf-mag", add_settings=add_settings, magnetic_moments=mags)
            """except (TypeError, ValueError):
                print("A VASP error has occurred in test: {} | {}. Please check again".format(test, test_value))
                energy = 0"""
            # self.f.write("{},{}\n".format(test_value, energy))

            #energies.append(energy)
            os.chdir("../")

        self.f.write("Testing Finished!")
        self.f.write("-" * 80 + "\n")

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
        band_path = self.get_band_path()[0]

        # TODO: Clean and finish this

    def calc_manager(self, calc_seq=None, add_settings_dict=None, magnetic_moments=None, hubbard_params=None,
                     nkpts=200, test_run=False):

        """
        For HSE06 bands need to do an scf, get the IBZKPTS and WAVECAR files and use these with zero weight band k-path
        for the HSE06 band structure
        """

        # default calculation sequence is relaxation and scf
        if calc_seq is None:
            raise ValueError("No calculation sequence (calc_seq) specified!")

        if add_settings_dict is None:
            add_settings_dict = {}

        # deepcopy needed to reset the add_settings_dict after every calculation sequence
        # TODO: there will be a non-destructive way to do this but this is a quick and easy fix
        copy_add_settings_dict = copy.deepcopy(add_settings_dict)

        # check for test run
        if test_run:
            self.f.write("Test run is selected. This run will use minimal settings to test that transitions"
                         "between calculations run smoothly. \n")
            self.f.write("!!IMPORTANT NOTE!!: Results from test runs are meaningless. Please DO NOT use these results!")

        # all output of individual calculations is written to the one file (which note: is always open)

        self.f.write("-" * 80 + "\n")
        self.f.write(28 * " " + "NEW CALCULATION SEQUENCE" + "\n")
        self.f.write("\n")
        self.f.write(f"Calculation sequence consists of: {calc_seq} \n")

        if copy_add_settings_dict:
            self.f.write(f"Additional settings: {copy_add_settings_dict} \n")
        self.f.write("-" * 80 + "\n")

        # loop through all calculations
        for i, calc in enumerate(calc_seq):
            self.f.write(f"Beginning calculation: {calc} as calculation {i + 1} in sequence. \n")

            # if no additional setting specified for a given calculation, add empty dict
            if calc not in copy_add_settings_dict:
                copy_add_settings_dict[calc] = {}

            # make separate directories for each calculation
            path = f"./{calc}"
            self.f.write(f"file path is {path} \n")

            # check if individual calculation is magnetic and if hubbard parameters are specified
            if "mag" in calc:
                self.f.write("Calculation is magnetic! \n")
                if not magnetic_moments:
                    self.f.write("No magnetic moments are specified. Are you sure this is correct? \n")
                else:
                    self.f.write(f"Initial magnetic moments are: {magnetic_moments} \n")
                mag_moments = magnetic_moments

                # hubbard check
                if not hubbard_params:
                    self.f.write("No hubbard parameters requested. Are you sure this calculation will converge?\n")
                else:
                    # check if add_settings already exists and append ldau_luj values
                    self.f.write(f"Hubbard values to be used are as follows: {hubbard_params} \n")

                    copy_add_settings_dict[calc].update(self.parameters["hubbard"])
                    copy_add_settings_dict[calc]["ldau_luj"] = hubbard_params

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
            current_struct, energy = self.single_vasp_calc(calculation_type=calc,
                                                           add_settings=copy_add_settings_dict[calc],
                                                           path_name=path,
                                                           write_safe_files=write_safe_file,
                                                           read_safe_files=read_safe_file,
                                                           magnetic_moments=mag_moments,
                                                           nkpts=nkpts, testing=test_run, max_cycles=10)

        return current_struct, energy

    @staticmethod
    def relax_struct(self, add_settings=None, path_name="./relax", read_safe_files=False,
                     write_safe_files=False, magnetic_moments=False):

        print("This method has now been discontinued. Please use the `single_vasp_valc` method to perform relaxations")

    def single_vasp_calc(self, calculation_type="scf", add_settings=None, path_name="./", nkpts=200,
                         read_safe_files=False, write_safe_files=False, magnetic_moments=None, testing=False, max_cycles=10):
        """
        A self-contained function that runs a single VASP calculation
        :param write_safe_files:
        :param magnetic_moments:
        :param path_name:
        :param nkpts:
        :param calculation_type:
        :param add_settings:
        :return:

        Parameters
        ----------

        safe_file
        """

        # check if directory already exists and if not change to directory
        if not os.path.exists(path_name):
            os.mkdir(path_name)
        os.chdir(path_name)

        # if safe files to be used copy to cwd
        if read_safe_files:
            shutil.copy2(self.safe_dir + "/POSCAR", "./")
            shutil.copy2(self.safe_dir + "/CHGCAR", "./")
            shutil.copy2(self.safe_dir + "/WAVECAR", "./")

        # rewrite the last_dir in case it is needed later
        self.last_dir = path_name

        # copy vasp settings from standard set-up
        vasp_settings = self.general_calculation.copy()

        # check for magnetism
        # can give an explicit list or a string in vasp format that is sorted by multiply_out_moments function
        if magnetic_moments:
            if type(magnetic_moments) == str:
                magnetic_moments = self.multiply_out_moments(magnetic_moments)
            self.structure.set_initial_magnetic_moments(magmoms=magnetic_moments)
            vasp_settings.update({"ispin": 2})

        # update settings for calculation type
        calc_strip = calculation_type.replace("-mag", "").replace("single-", "")
        vasp_settings.update(self.parameters[calc_strip])

        # check for hybrid
        if calculation_type.find("hse06") != -1:
            vasp_settings.update()

        # add any extra settings
        vasp_settings.update(add_settings)

        # set testing run
        if testing:
            vasp_settings.update(self.parameters["test_defaults"])
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
                        self.structure = read("./POSCAR")  # need to read in the new POSCAR after every run for consistency
                    except IndexError:
                        if not os.path.exists("./POSCAR-PREVIOUS-RUN"):
                            print("no help 4 u")
                        shutil.copy2("POSCAR-PREVIOUS-RUN", "POSCAR")
                        self.structure = read("./POSCAR")

                    vasp_settings.update({"icharg": 1, "istart": 1})

                # run calculation
                structure, energy, result = self.run_vasp(vasp_settings)

                if result == "vasp failure":
                    break

                if self.converged_in_one_scf_cycle("OUTCAR"):
                    break
                steps += 1

                # break if max number of cycles are reached
                if steps >= max_cycles:
                    break

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
        except UnboundLocalError:
            print("UnboundLocalError found so trying to turn off symmetry")
            try:
                vasp_settings["isym"] = 0
                structure.set_calculator(Vasp(**vasp_settings, restart=restart))
                energy = structure.get_potential_energy()
            except:
                result = "vasp failure"
        except IndexError:
            try:
                vasp_settings["isym"] = 0
                structure.set_calculator(Vasp(**vasp_settings, restart=restart))
                energy = structure.get_potential_energy()
            except:
                result = "vasp failure"
            result = "vasp failure"

        return structure, energy, result

    def batch_calculations(self, directories, constants, add_settings_dict=None, *args):
        """

        A method for running many different versions of similar calculations

        Options may include:

            • Constant structure with different magnetic structures
            • Constant crystal structure with swapping/adding/deleting atoms
            • Constant calculations parameters for different structures

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

        """        # get structure
        if "xml" in file:
            k_structure = Vasprun(file).final_structure
        elif "POSCAR" in file:
            k_structure = Poscar.from_file(file).structure
        else:
            raise TypeError("File type must be POSCAR or vasprun.xml")"""
        # This defines the band structures from Setwayan et al
        from matplotlib import pyplot as plt

        path = bandpath(path=None, cell=self.structure.cell, npoints=nkpts, eps=1E-3)
        return path.kpts, path.path

    @staticmethod
    def plot_band_path(self, file="vasprun.xml", path="./", eng="0", reference=0.0):
        read_obj = read(path + file)

    def self_consistent_hubbard(self, relax=False, species=None, setup_param=None, mag=False, ignore_scf=False):
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
            hubb_dir.update({atom: {'L': orbital, 'U': 0, 'J': 0}})

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
            self.calc_manager(calc_seq=calc_seq, add_settings_dict=add_settings,
                              magnetic_moments=None, hubbard_params=hubb_dir, outfile="vasp_seq.out")

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
                hubb_dir[active_species].update({'U': alpha, 'J': alpha})
                chi_settings.update(hubb_dir)

                if mag:  # magnetism hasn't worked for MnFe2O4 so far but non-magnetic seems to work quite well
                    mag = "-mag"

                if step == "bare":
                    chi_settings.update({"icharg": 11})

                elif step == "inter":
                    chi_settings.update({"icharg": 1})

                self.single_vasp_calc("scf" + mag, add_settings=chi_settings, path_name="./",
                                      write_safe_file=True)

                os.chdir("../")
            os.chdir("../")

    @staticmethod
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


def get_ch(key="0G4rqjSNG4M51Am0JNj", must_contain=[], species=None, filepath="./", elim=0.025):
    """

    Args:
        key:
        species:
        filepath:
        elim:
    """
    if species is None:
        species = ["Mn", "Fe", "O"]

    with MPRester(key) as m:
        mp_entries = m.get_entries_in_chemsys(species)

        analyser = PhaseDiagram(mp_entries)
        # e = analyser.get_e_above_hull(mp_entries[0])

        if not os.path.exists(filepath):
            os.mkdir(filepath)
        os.chdir(filepath)
        owd = os.getcwd()

        for entry in mp_entries:

            if must_contain:
                if all(species in entry.composition.alphabetical_formula for species in must_contain):
                    get_entry = True
                else:
                    get_entry = False
            else:
                get_entry = True

            if analyser.get_e_above_hull(entry) < elim and get_entry:

                formula_dir = f"./{entry.composition.alphabetical_formula.replace(' ', '')}"

                if os.path.exists(formula_dir):
                    os.chdir(formula_dir)
                    os.mkdir(entry.entry_id)
                    os.chdir(entry.entry_id)
                else:
                    os.mkdir(formula_dir)
                    os.chdir(formula_dir)
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
        x = re.split('[0-9]', sub)[:-1]
        filtered_x = sorted(list(filter(None, x)))
        dir_name = ""
        for i in filtered_x:
            dir_name += i
            dir_name += "-"

        dir_name = dir_name.strip("-")

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        print(f"Moving {sub} to {dir_name}")
        shutil.move(sub, dir_name)

def convex_hull_relaxations(species, root_directory="./", hubbards=None, additional_settings=None, grouped_directories=True, testing=False):
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
        species:
        hubbards:
        additional_settings:
        grouped_directories:
        testing:

    Returns:

    """
    
    # change to location
    os.chdir(root_directory)
    # for each chemical system in composition space
    for system in species.keys():
        system_directory = os.getcwd()

        print("-" * 50)
        print(f"Running relaxation calculation for {system} in {system_directory}")
        print(f"Changing directory to {system_directory}")
        
        os.chdir(f"./{system}")

        # gets all immediate subdirectories which should be named as mp codes but could be anything
        subdirectories = [f.path for f in os.scandir("./") if f.is_dir()]
        
        # if there are no subdirectories then just run calculations in the current structure
        # not that if magnetic calculations are required, there still has to be a subdirectory to change into
        if not subdirectories:
            subdirectories = ["./"]

        # for each polymorph
        for subdirectory in subdirectories:
            
            calculation_directory = os.getcwd()
            print("-" * 50)
            print(f"Running relaxation calculation for {system} in {calculation_directory}")
            print(f"Changing directory to {calculation_directory}")
            print("-" * 50)

            if not os.path.exists(subdirectory):
                os.mkdir(subdirectory)
            os.chdir(subdirectory)
            
            # read poscar from the subdirectory as any children from here all have the same structures
            initial_poscar = read("./POSCAR")  # TODO: add try-except here

            calculator = VaspCalculations(initial_poscar, write_file=f"./{system}.out")

            # if only one magnetic configuration specified
            if type(species[system]) == list:

                print(f"For {subdirectory} in system {system} a list is given for magnetic moments and therefore\n"
                      f"it is assumed that only one magnetic structure is to be tested")
                if not testing:
                    relax = calculator.calc_manager(calc_seq=["relax-mag"],
                                                    add_settings_dict={"relax-mag": additional_settings},
                                                    magnetic_moments=species[system],
                                                    hubbard_params=hubbards)

            # elif multiple magnetic configurations specified
            elif type(species[system]) == dict:
                print(f"For {subdirectory} in system {system} a dictionary is given for magnetic moments and therefore\n"
                      f"it is assumed that {len(species[system])} magnetic structures are to be tested")
                for key, magnetic_configuration in zip(species[system].keys(), species[system].values()):

                    if os.path.exists(f"./{key}"):
                        rmtree(f"./{key}")

                    os.mkdir(f"./{key}")
                    os.chdir(f"./{key}")

                    if not testing:
                        relax = calculator.calc_manager(calc_seq=["relax-mag"],
                                                        add_settings_dict={"relax-mag": additional_settings},
                                                        magnetic_moments=magnetic_configuration,
                                                        hubbard_params=hubbards)
                    os.chdir("../")
            
            # elif not magnetic
            elif species[system] is None:
                print(f"For {subdirectory} in system {system} no magnetic moments are specified and therefore\n"
                      f"it is assumed that the calculation is non magnetic")
                if not testing:
                    relax = calculator.calc_manager(calc_seq=["relax"],
                                                    add_settings_dict={"relax": adds})
            else:
                raise TypeError("Wrong type for magnetic moments specified!")
            
            os.chdir(calculation_directory)
        os.chdir(system_directory)

def get_magnetic_moments_from_mp(root="./", write_file="./magnetic_moments"):
    os.chdir(root)
    mp_keys = [f[0].split("/")[-1] for f in os.walk("./") if not f[0].split("/")[-1].find("mp")]

    # get_ch(species=["Ni", "Mn", "O", "Fe", "Cr"], filepath="./")
    # ch_sorter(filepath="./")
    mags = {}
    with MPRester("0G4rqjSNG4M51Am0JNj") as m:
        for key in mp_keys:
            material = m.get_structure_by_material_id(key)
            moments = MPRelaxSet(material)
            moments = [round(mom) for mom in moments.incar["MAGMOM"]]
            print(f"{material.formula} has mag moms {moments}")
            mags[f"{material.composition.alphabetical_formula.replace(' ', '')}"] = moments
    
    with open(write_file, 'w') as wf:
        wf.write(json.dumps(mags))
    
    return mags