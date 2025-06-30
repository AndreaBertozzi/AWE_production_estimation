# AWE Production Estimation

**AWE Production Estimation** is a tool for estimating production output and performance of ground generation, flexible wing **Airborne Wind Energy (AWE)** systems. It provides a framework to evaluate and forecast the energy yield of AWE technologies based on configurable parameters and input data.

## Installation

To get started, first clone the repository:

```bash
git clone https://github.com/AndreaBertozzi/AWE_production_estimation.git
cd awe-production-estimation
```

### Option 1: Using Python `venv`

Ensure you have **Python 3.6 or later** installed.

1. Create a virtual environment inside the project folder:

    ```bash
    python -m venv venv
    ```

2. Activate the virtual environment:

   - On **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```
   - On **Windows**:
     ```cmd
     venv\Scripts\activate
     ```

3. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Option 2: Using Conda

Ensure you have **conda** installed (Anaconda or Miniconda).

1. Create a conda environment inside the project directory:

    ```bash
    conda create --prefix ./venv python=3.8
    ```

2. Activate the environment:

   - On **Linux/macOS**:
     ```bash
     source activate ./venv
     ```
   - On **Windows**:
     ```cmd
     conda activate .\venv
     ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

> 📝 **Note**: Replace `python=3.8` in the conda command with your desired version (must be >= 3.6) if needed.
---

## Usage

The **AWE Production Estimation** software consists of several modular components for simulating and optimizing the performance of ground generation, flexible wing **Airborne Wind Energy (AWE)** systems. All modules are configured using a single `config.yaml` file, which defines the physical system setup, operational strategy, optimization parameters, and solver settings.

### Modules Overview

- `qsm.py`: Implements the **Quasi-Steady Model** for simulating full operation cycles of an AWE system.
- `exp_validation_utils.py`: Provides utilities for **experimental validation**, including packing and exporting simulation results.
- `cycle_optimizer.py`: Contains the **Cycle Optimizer** to find optimal operational settings for given wind conditions.
- `power_curve_constructor.py`: Builds full **power curves** across a range of wind speeds using repeated optimization.

## Input Configuration

All inputs are defined in a central `config.yaml` file, with the following sections:

- **environment**: Wind profile setup (e.g., logarithmic or tabulated)
- **kite, tether, ground station**: Physical properties of the AWE system
- **sim_settings**: Simulation parameters like timestep and control strategy (`force`, `speed`, or `hybrid`)
- **bounds**: Operational limits for optimization (e.g., force, speed, angles)
- **constraints**: Inequality constraints used in the optimization routines
- **opt_variables**: Variables to be optimized with their initial values
- **opt_settings**: Optimizer configuration (e.g., max iterations, tolerance)

For a full list and example values, see the detailed configuration section below.

> ⚠️ **Note:** Currently, full `config.yaml` support is implemented for force-controlled simulations only. Hybrid control modes may cause convergence issues in the `cycle_optimizer` and `power_curve_constructor` modules.
---

## Output

Each module produces structured outputs:

- **`qsm.py`**: Access to detailed kinematics and steady-state values per phase
- **`cycle_optimizer.py`**: Returns optimized cycle parameters and performance summary
- **`power_curve_constructor.py`**: Saves full power curve results to `.csv` and `.pickle` for analysis and plotting
---
