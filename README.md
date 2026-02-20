# Project EEEEE: Molecular Property Calculator
**Advanced Quantum Chemistry (2302638) Capstone Project** **Chulalongkorn University, Semester 2/2025**

## Project Overview
This project focuses on the implementation of a **Molecular Property Calculator**. The goal is to demonstrate mastery of the Hartree-Fock computational pipeline by implementing an RHF SCF loop and using the resulting converged density matrix to extract physical quantities and energy components.

## Repository Structure 
* `src/`: Contains the core Python implementation and unit tests.
    * `main.py`: Core RHF SCF and property calculation implementation.
    * `test_*.py`: Unit tests (minimum 3 tests per core function).
* `results/`: Output tables, energy decomposition plots, and figures.
* `report.pdf`: 6–10 page formal report (Theory, Results, and Analysis).
* `requirements.txt`: Python dependencies (NumPy, SciPy, PySCF).
* `README.md`: Project description and instructions.

## Implemented Features
In accordance with the requirements for Project E, the following components are implemented from scratch:
* **RHF SCF Loop**: Full implementation of Algorithm 6.1.
* **Energy Decomposition**: Calculation of $E_1$, $E_J$, $E_K$, and $E_{nuc}$.
* **Dipole Moment**: Evaluation of $\mu = \mu_{nuc} - \text{tr}\{\mathbf{P}\mathbf{r}\}$.
* **Mulliken Population Analysis**: Calculation of atomic charges $q_A$.
* **Virial Ratio**: Diagnostic calculation of $\eta = -\langle V \rangle / \langle T \rangle$.

## How to Run
### 1. Installation
Clone the repository and install the required dependencies using:
```bash
pip install -r requirements.txt
