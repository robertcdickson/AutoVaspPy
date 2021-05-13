AutoVaspPy
---

AutoVaspPy is a high-level python module that allows for performing complex vasp calculations and analysis
using python. AutoVaspPy makes extensive use of the ase and pymatgen modules to run and analyse vasp calculations.

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
      <ul>
        <li><a href="#vaspcalculations">VaspCalculations</a></li>
        <li><a href="#postprocessing">Postprocessing</a></li>
      </ul>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- USAGE EXAMPLES -->
## Usage
Structures are processed using ase format and can be loaded into python using ase's `read()` function

```python
from ase.io import read
structure = read("/path/to/file/")
```
A full list of file formats is listed in the [ASE documentation](https://wiki.fysik.dtu.dk/ase/dev/ase/io/io.html).

An instance of the `VaspCalculations` can then be created using the loaded structure
```python
from AutoVaspPy import VaspCalculations
vasp_calculator = VaspCalculations(structure)
```
The `VaspCalculations` class is initialised with general calculation parameters that should be adjusted to the users
needs. It is recommended that all parameters are explicitly defined to ensure the user is fully aware of all the elements
that may affect their results. The `vasp_calculator` instance has many functions that can be used for running different calculations.

The `parameter_testing` function allows for an automatic workflow to converge various calculation parameters of interest
(k-point spacing, cut off energy *etc.*). Here is an example of k-point spacing convergence on a fictious magnetic system

```python
test_values = [0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.5]
k_spacing_convergence = vasp_calculator.parameter_testing(test="k-spacing",
                                                          values=test_values,
                                                          magnetic_moments=[5, 5, 0, 0],
                                                          hubbard_parameters=None,
                                                          plot=True)
```

Where `plot=True` the matplotlib package is used to save a figure plotting the energy of the system against the parameter
tested. After the calculation is finished, a numpy array is returned which can be used for analysis and more adavanced 
plotting.

### Pipelining Calculations
The `calculation_manager` function allows for the pipelining of different VASP calculations to have a fully autonomous
workflow. An example is given here where the structure is relaxed, the DOS, band structure and dielectric tensor are all
calculated in sequence with different settings.

```python

calculation_settings = {

    "relax-mag":
        {"algo": "normal",
         "kspacing": 0.15,
         "write_safe_file": True},

    "scf-mag":
        {"write_safe_file": True,
         "nedos": 5000},

    "bands-mag":
        {"algo": 2,
         "read_safe_file": True,
         },

    "eps-mag":
        {"write_safe_file": True,
         "read_safe_file": True}
}

vasp_calculator.calculation_manager(calculation_sequence=calculation_settings.keys(),
                                    additional_settings=calculation_settings,
                                    magnetic_moments=[5, 5, 0, 0],
                                    hubbard_parameters=None,
                                    nkpts=300,
                                    test_run=False)
}
```
The `calculation_sequence` parameter is used to list the pipeline of calculations. The `additional_settings` keyword
allows for different VASP parameters to be used at each stage of the calculation. The number of k-points for a band structure
calculation currently has to be set with the `nkpts` keyword. `test_run` allows for testing of the pipeline with minimal
VASP settings (useful for testing a pipeline locally before running a full-scale calculation with HPC resources).

