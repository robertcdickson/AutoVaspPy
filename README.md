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
k_spacing_convergence = vasp_calculator.parameter_testing(test="k-spacing",
                                                          values=[0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.5],
                                                          magnetic_moments=[5, 5, 0, 0],
                                                          hubbard_parameters=None,
                                                          plot=True)
```

Where `plot=True` the matplotlib package is used to save a figure plotting the energy of the system against the parameter tested


