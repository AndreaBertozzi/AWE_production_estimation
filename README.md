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
    conda create --prefix ./venv python=3.6.8
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

> 📝 **Note**: Replace `python=3.8` in the conda command with your desired version (must be >= 3.6). The installation of `pygrib` from pip when using `python=3.6` could give some error. Consider upgrading your `python` or install pygrib via conda.

### 📂 Create Config and Output Folder

Before running simulations or optimizations, please create an `output/` directory inside the project folder to store results and generated files and a `config/` directory inside the project folder to store .yaml configuration files (see `config_template.yaml` for example):

```bash
mkdir output
mkdir config
```
---

## Usage

The **AWE Production Estimation** software consists of several modular components for simulating and optimizing the performance of ground generation, flexible wing **Airborne Wind Energy (AWE)** systems. All modules are configured using a single `config.yaml` file, which defines the physical system setup, operational strategy, optimization parameters, and solver settings.

### Modules Overview

- `qsm.py`: Implements the **Quasi-Steady Model** for simulating full operation cycles of an AWE system.
- `exp_validation_utils.py`: Provides utilities for **experimental validation**, including packing and exporting simulation results.
- `cycle_optimizer.py`: Contains the **Cycle Optimizer** to find optimal operational settings for given wind conditions.
- `power_curve_constructor.py`: Builds full **power curves** across a range of wind speeds using repeated optimization.
- `energy_estimator.py`: Leverages the optimal **power curve** to estimate the **productivity** of  AWE system. 

## Input Configuration

All inputs are defined in a single `config.yaml` file, with the following sections:

- **environment**: Wind profile setup (e.g., logarithmic or tabulated)
- **kite, tether, ground station**: Physical properties of the AWE system
- **sim_settings**: Simulation parameters like timestep and control strategy (`force`, `speed`, or `hybrid`)
- **bounds**: Operational limits for optimization (e.g., force, speed, angles)
- **constraints**: Inequality constraints used in the optimization routines
- **opt_variables**: Variables to be optimized with their initial values
- **opt_settings**: Optimizer configuration (e.g., max iterations, tolerance)
- **power_curve_smoothing**: Settings for the power curve smoothing (e.g. polynomial approximation order)
- **trajectory_etas**: Representing the efficiency linked to the variability of the trajectories
- **electrical_etas**: Efficiencies and self-consumption of the electrical equipment in the ground station

For a full list and example values, see the detailed configuration section below.

> ⚠️ **Note:** Currently, full `config.yaml` support is implemented for force-controlled simulations only. Hybrid control modes may cause convergence issues in the `cycle_optimizer` and `power_curve_constructor` modules.
---

## Output

Each module produces structured outputs:

- **`qsm.py`**: Access to detailed kinematics and steady-state values per phase
- **`cycle_optimizer.py`**: Returns optimized cycle parameters and performance summary
- **`power_curve_constructor.py`**: Saves full power curve results to `.csv` and `.pickle` for analysis and plotting
- **`energy_estimator.py`**: Generates a number of plots showcasing daily, monthly, and yearly energy production and flight statistics
---
## Configuration File Reference

All simulations and optimizations are controlled via a `config.yaml` file. Below is a structured reference of all configurable sections and parameters.

---

### 🔁 `environment`

Defines the wind profile model used in simulations.

```yaml
environment:
  profile: logarithmic            # Type of wind profile ('logarithmic', 'table1D', 'table2D')
  roughness_length: 0.07          # Roughness length for log profile [m]
  ref_height: 100                 # Reference height for wind speed [m]
  ref_windspeeds: [8, 10, 12]     # Wind speeds at reference height [m/s]
```

---

### 🪁 `kite`, `tether`, and `ground station`

Defines physical properties of the AWE system components.

```yaml
kite:
  mass:                         # Kite mass [kg]
  projected_area:          # Projected area [m²]
  drag_coefficient:
    powered:
    depowered:
  lift_coefficient:
    powered: 
    depowered: 

tether:
  length:                    # Max tether length [m]
  diameter:                  # Tether diameter [m]
  density:                   # Tether density [kg/m³]
  drag_coefficient:
```

---

### 🛠️ `sim_settings`

Sets simulation behavior and control method.

```yaml
sim_settings:
  force_or_speed_control: 'force' # Control strategy: 'force', 'speed', or 'hybrid'
  time_step_RO:              # Timestep during reel-out [s]
  time_step_RI:              # Timestep during reel-in [s]
  time_step_RIRO:            # Timestep during RIRO transition [s]
```

> ⚠️ Hybrid mode is experimental and not fully supported in all modules.

---

### 📏 `bounds`

Defines constraints for optimization and physical feasibility.

```yaml
bounds:
  avg_elevation:
    min:
    max:
  max_azimuth:
    min:
    max: 
  relative_elevation:
    min: 
    max: 
  force_limits:
    min:                      # Minimum control force [kgf]
    max:
  speed_limits:
    min:
    max:
  tether_stroke:
    min: 
    max: 
  minimum_tether_length:
    min: 
    max: 
```

---

### 🚦 `constraints`

Activates and parameterizes model constraints.

```yaml
constraints:
  force_out_setpoint_min:
    enabled: true
  force_in_setpoint_max:
    enabled: true
  ineq_cons_traction_max_force:
    enabled: true
  ineq_cons_min_tether_length:
    enabled: true
  ineq_cons_max_tether_length:
    enabled: true
  ineq_cons_cw_patterns:
    enabled: true
    min_patterns: 
  ineq_cons_max_elevation:
    enabled: true
    max_elevation: 
  ineq_cons_max_course_rate:
    enabled: true
    max_course_rate: 
```

---

### 🧮 `opt_variables`

Lists optimization variables, their initial values, and units.

```yaml
opt_variables:
  F_RO:
    enabled: true
    init_value:    # N
    unit: N
  F_RI:
    enabled: true
    init_value:   # N
    unit: N
  average_elevation:
    enabled: true
    init_value:   # deg
  relative_elevation:
    enabled: false
    init_value:     # deg
  maximum_azimuth:
    enabled: false
    init_value:     # deg
  minimum_tether_length:
    enabled: true
    init_value:      # m
  tether_stroke:
    enabled: true
    init_value:      # m
```

> Units for force can be `N` or `kgf`. Angular values are given in degrees and internally converted to radians.

---

### ⚙️ `opt_settings`

Configures the optimizer (currently uses SciPy SLSQP).

```yaml
opt_settings:
  maxiter: 30            # Max optimization iterations
  iprint: 2              # Verbosity level
  ftol: 1e-3             # Function tolerance
  eps: 1e-6              # Step size for numerical gradient
```

### 🪁 
```yaml
trajectory_etas:
  efficiency: 
```

---
## 🚀 Example: Cycle Optimization

The following script demonstrates how to run a full cycle optimization using the `OptimizerCycle` class with a wind profile and parameters defined in a YAML configuration file.

```python
import yaml
import numpy as np
import matplotlib.pyplot as plt
from cycle_optimizer import OptimizerCycle
from qsm import LogProfile, TractionPhasePattern, SystemProperties
from utils import *

# --- Load configuration from YAML ---
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# --- System properties ---
sys_props_dict = parse_system_properties_and_bounds(config)
sys_props = SystemProperties(sys_props_dict)

# --- Environment settings ---
env_state = LogProfile()
env_state.set_reference_wind_speed(6.5)     # [m/s]
env_state.set_reference_height(100)         # [m]
env_state.set_roughness_length(0.07)        # [m]

# --- Simulation settings ---
cycle_sim_settings = {
    'cycle': {
        'traction_phase': TractionPhasePattern,
        'include_transition_energy': True,
    },
    'retraction': {'time_step': 0.5},
    'transition': {'time_step': 0.5},
    'traction': {'time_step': 0.25},
}

# --- Optimization variable and constraint selection ---
opt_var_enabled_idx, init_vals = parse_opt_variables(config)
cons_enabled_idx, cons_param_vals = parse_constraints(config)

# --- Set up optimizer ---
oc = OptimizerCycle(
    cycle_sim_settings,
    sys_props,
    env_state,
    reduce_x=opt_var_enabled_idx,
    reduce_ineq_cons=cons_enabled_idx,
    parametric_cons_values=cons_param_vals,
    force_or_speed_control='force'
)

# Provide good initial guess to improve convergence
oc.x0_real_scale = init_vals

# --- Run optimization ---
x_opt = oc.optimize(iprint=2, maxiter=30, ftol=1e-3, eps=1e-4)

# --- Post-process results ---
print('Optimal solution:', x_opt)
ylabs = (
    r'$L_{tether}$ [m]',
    r'$v_{reeling~out}$ [m/s]',
    r'$F_{tether}$ [N]',
    r'$P_{ground}$ [W]'
)
constraints, kpis = oc.eval_point(plot_results=True, x_real_scale=x_opt, labels=ylabs)

print('Successful optimization:', oc.op_res['success'])

plt.show()
```

> ✅ **Note:** Make sure your working directory contains a valid `config/config.yaml` file, as described in the [Configuration File Reference](#configuration-file-reference).

---

## 🗂️ Project Structure

The codebase is organized into modular components, each responsible for a specific stage of modeling, simulation, or analysis.

```
AWE-Production-Estimation/
|
├── examples/                      # Folder containing some examples
├── examples_data/                 # Folder containing the input data for the examples
├── wind_resource/                 # Folder containing wind profiles and relative occurrence
├── config_template.yaml           # Template for the configuration file
│
├── cycle_optimizer.py             # Cycle optimization logic using QSM
├── exp_validation_utils.py        # Experimental validation utilities and results packaging
├── power_curve_constructor.py     # Class for power curve generation over multiple wind speeds
├── qsm.py                         # Quasi-Steady Model implementation
│
├── power_curve_single_profile     # Script to calculate the power curve using a single wind profile, i.e. logarithmic
│
├── config/
│   └── config.yaml                # Input configuration file (create directory manually,
│                                    and create config.yaml according to template)
├── output/                        # Folder to store simulation and optimization results (create manually)
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── LICENSE                        # MIT license file
└── ...
```

### Main Modules Overview

- **`qsm.py`**  
  Implements the core physical model using a quasi-steady formulation. Supports 1D/2D wind profiles and constraint enforcement.

- **`exp_validation_utils.py`**  
  Contains helper functions for validation and result processing.

- **`cycle_optimizer.py`**  
  Uses the QSM model to perform constrained optimization over a full operational cycle.

- **`power_curve_constructor.py`**  
  Automates the generation of power curves by running cycle optimizations over a range of wind speeds.

---

> ⚠️ **Important:** Please create the `output/` folder manually inside the project root before running simulations or optimizations. The power curves are saved in this directory.

## 🚀 How to Run

Once your environment is set up and the `output/` folder is created, you're ready to run simulations or optimizations.

### 🧪 Example Scripts

A set of example scripts is provided in the `examples/` folder. These demonstrate typical use cases such as:

- Running a single quasi-steady simulation
- Comparing quasi-steady simulation with experimental data
- Performing a full cycle optimization

To run any example:

```bash
python examples/your_example_script.py
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software, provided that proper credit is given.

---

## 🙏 Acknowledgments
If you use this software in a scientific publication, please consider citing it appropriately or linking back to this repository.
