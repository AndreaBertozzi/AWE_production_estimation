# -*- coding: utf-8 -*-
"""Utility functions."""
import matplotlib.pyplot as plt
import yaml
from numpy import all, diff, array

import math

 

def flatten_dict(input_dict, parent_key='', sep='.'):
    """Recursive function to convert multi-level dictionary to flat dictionary.

    Args:
        input_dict (dict): Dictionary to be flattened.
        parent_key (str): Key under which `input_dict` is stored in the higher-level dictionary.
        sep (str): Separator used for joining together the keys pointing to the lower-level object.

    """
    items = []  # list for gathering resulting key, value pairs
    for k, v in input_dict.items():
        new_key = parent_key + sep + k.replace(" ", "") if parent_key else k.replace(" ", "")
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def zip_el(*args):
    """"Zip iterables, only if input lists have same length.

    Args:
        *args: Variable number of lists.

    Returns:
        list: Iterator that aggregates elements from each of the input lists.

    Raises:
        AssertError: If input lists do not have the same length.

    """
    lengths = [len(l) for l in [*args]]
    assert all(diff(lengths) == 0), "All the input lists should have the same length."
    return zip(*args)


def plot_traces(x, data_sources, source_labels, plot_parameters, y_labels=None, y_scaling=None, fig_num=None,
                plot_kwargs={}, plot_markers=None, x_label='Time [s]'):
    """Plot the time trace of a parameter from multiple sources.

    Args:
        x (tuple): Sequence of points along x.
        data_sources (tuple): Sequence of time traces of the different data sources.
        source_labels (tuple): Labels corresponding to the data sources.
        plot_parameters (tuple): Sequence of attributes/keys of the objects/dictionaries of the time traces.
        y_labels (tuple, optional): Y-axis labels corresponding to `plot_parameters`.
        y_scaling (tuple, optional): Scaling factors corresponding to `plot_parameters`.
        fig_num (int, optional): Number of figure used for the plot, if None a new figure is created.
        plot_kwargs (dict, optional): Line plot keyword arguments.

    """
    if y_labels is None:
        y_labels = plot_parameters
    if y_scaling is None:
        y_scaling = [None for _ in range(len(plot_parameters))]
    if fig_num:
        axes = plt.figure(fig_num).get_axes()
    else:
        axes = []
    if not axes:
        _, axes = plt.subplots(len(plot_parameters), 1, sharex=True, num=fig_num)
    if len(axes) == 1:
        axes = (axes,)

    for p, y_lbl, f, ax in zip_el(plot_parameters, y_labels, y_scaling, axes):
        for trace, s_lbl in zip_el(data_sources, source_labels):
            y = None
            #TODO: see if it is a better option to make this function a method and check if p is an attribute as condition
            if p == s_lbl:
                y = trace
            elif isinstance(trace[0], dict):
                if p in trace[0]:
                    y = [item[p] for item in trace]
            elif hasattr(trace[0], p):
                y = [getattr(item, p) for item in trace]
            if y:
                if f:
                    y = array(y)*f
                ax.plot(x, y, label=s_lbl, **plot_kwargs)
                if plot_markers:
                    marker_vals = [y[x.index(t)] for t in plot_markers]
                    ax.plot(plot_markers, marker_vals, 's', markerfacecolor='None')
        ax.set_ylabel(y_lbl)
        ax.grid(True)
        # ax.legend()
    axes[-1].set_xlabel(x_label)
    axes[-1].set_xlim([0, None])

def parse_system_properties_and_bounds(config):
    kite = config["kite"]
    tether = config["tether"]
    bounds = config["bounds"]

    params_dict = {
        # Kite
        "kite_mass": kite["mass"],
        "kite_projected_area": kite["projected_area"],
        "kite_drag_coefficient_powered": kite["drag_coefficient"]["powered"],
        "kite_drag_coefficient_depowered": kite["drag_coefficient"]["depowered"],
        "kite_lift_coefficient_powered": kite["lift_coefficient"]["powered"],
        "kite_lift_coefficient_depowered": kite["lift_coefficient"]["depowered"],

        # Tether
        "total_tether_length": tether["length"],
        "tether_diameter": tether["diameter"],
        "tether_density": tether["density"],
        "tether_drag_coefficient": tether["drag_coefficient"],

        # Bounds
        "avg_elevation_min_limit": bounds["avg_elevation"]["min"]*math.pi/180,
        "avg_elevation_max_limit": bounds["avg_elevation"]["max"]*math.pi/180,
        "max_azimuth_min_limit": bounds["max_azimuth"]["min"]*math.pi/180,
        "max_azimuth_max_limit": bounds["max_azimuth"]["max"]*math.pi/180,
        "rel_elevation_min_limit": bounds["relative_elevation"]["min"]*math.pi/180,
        "rel_elevation_max_limit": bounds["relative_elevation"]["max"]*math.pi/180,
        "reeling_speed_min_limit": bounds["speed_limits"]["min"],
        "reeling_speed_max_limit": bounds["speed_limits"]["max"],
        "tether_force_min_limit": bounds["force_limits"]["min"]*9.806,
        "tether_force_max_limit": bounds["force_limits"]["max"]*9.806,
        "tether_stroke_min_limit": bounds["tether_stroke"]["min"],
        "tether_stroke_max_limit": bounds["tether_stroke"]["max"],
        "min_tether_length_min_limit": bounds["minimum_tether_length"]["min"],
        "min_tether_length_max_limit": bounds["minimum_tether_length"]["max"],        
    }

    return params_dict

def parse_opt_variables(config):
    control_mode, _, _, _ = parse_sim_settings(config)

    angle_vars = {"average_elevation", "relative_elevation", "maximum_azimuth"}

    if control_mode == 'force':
        expected_order = [
            "F_RO",
            "F_RI",
            "average_elevation",
            "relative_elevation",
            "maximum_azimuth",
            "tether_stroke",
            "minimum_tether_length",
        ]
    
    elif control_mode == 'speed':
        expected_order = [
            "v_RO",
            "v_RI",
            "average_elevation",
            "relative_elevation",
            "maximum_azimuth",
            "tether_stroke",
            "minimum_tether_length",
        ]
    
    elif control_mode == 'hybrid':
        expected_order = [
            "v_RO",
            "F_RI",
            "average_elevation",
            "relative_elevation",
            "maximum_azimuth",
            "tether_stroke",
            "minimum_tether_length",
        ]

    opt_vars = config.get("opt_variables", {})
    init_values = []
    enabled_flags = []

    for var in expected_order:
        if var not in opt_vars:
            raise ValueError(f"Missing variable '{var}' in opt_variables section.")

        var_data = opt_vars[var]
        enabled = var_data.get("enabled", False)
        init_val = var_data.get("init_value")
        unit = var_data.get("unit", None)

        if var.startswith("F_"):  # force variable
            if unit == "kgf":
                init_val *= 9.80665
            elif unit in (None, "N"):
                pass
            else:
                raise ValueError(f"Invalid unit '{unit}' for variable '{var}'. Use 'kgf' or 'N'.")

        elif var in angle_vars:
            # Convert degrees to radians
            init_val = math.pi*init_val/180.

        # No conversion for length or unitless values
        init_values.append(init_val)
        enabled_flags.append(enabled)

    enabled_indices = array([i for i, flag in enumerate(enabled_flags) if flag])
    init_values_array = array(init_values)

    return enabled_indices, init_values_array

def parse_constraints(config):
    constraint_section = config.get("constraints", {})

    # Define expected order of all constraints (enabled flags)
    expected_constraints = [
        "force_out_setpoint_min",
        "force_in_setpoint_max",
        "ineq_cons_traction_max_force",
        "ineq_cons_cw_patterns",
        "ineq_cons_min_tether_length",
        "ineq_cons_max_tether_length",        
        "ineq_cons_max_elevation",
        "ineq_cons_max_course_rate",
    ]

    # Parametric constraints and the value key to extract
    parametric_constraints = {
        "ineq_cons_cw_patterns": "min_patterns",
        "ineq_cons_max_elevation": "max_elevation",
        "ineq_cons_max_course_rate": "max_course_rate",
    }

    enabled_flags = []
    param_values = []

    for name in expected_constraints:
        if name not in constraint_section:
            raise ValueError(f"Missing constraint '{name}' in YAML.")

        item = constraint_section[name]
        enabled = item.get("enabled", False)
        enabled_flags.append(enabled)

        # Handle parametric constraints
        if name in parametric_constraints:
            key = parametric_constraints[name]
            val = item.get(key)
            if val is None:
                raise ValueError(f"Missing '{key}' for constraint '{name}'")
            if name == "ineq_cons_max_elevation":
                val = math.radians(val)  # convert to radians
            param_values.append(val)

    enabled_indices = array([i for i, flag in enumerate(enabled_flags) if flag])
    param_values_array = array(param_values)

    return enabled_indices, param_values_array

def parse_opt_settings(config):
    required_keys = ["maxiter", "iprint", "ftol", "eps"]
    settings = config.get("opt_settings", {})

    missing = [k for k in required_keys if k not in settings]
    if missing:
        raise ValueError(f"Missing keys in 'opt_settings': {missing}")

    return {
        "maxiter": int(settings["maxiter"]),
        "iprint": int(settings["iprint"]),
        "ftol": float(settings["ftol"]),
        "eps": float(settings["eps"]),
    }

def parse_sim_settings(config):
    required_keys = ["force_or_speed_control", "time_step_RO", "time_step_RI", "time_step_RIRO"]
    valid_control_modes = {"force", "speed", "hybrid"}

    settings = config.get("sim_settings", {})
    missing = [k for k in required_keys if k not in settings]
    if missing:
        raise ValueError(f"Missing keys in 'sim_settings': {missing}")

    control_mode = settings["force_or_speed_control"].lower()
    if control_mode not in valid_control_modes:
        raise ValueError(f"Invalid 'force_or_speed_control': '{control_mode}'. Must be one of {valid_control_modes}.")

    time_step_RO = float(settings["time_step_RO"])
    time_step_RI = float(settings["time_step_RI"])
    time_step_RIRO = float(settings["time_step_RIRO"])

    return control_mode, time_step_RO, time_step_RI, time_step_RIRO

def parse_environment(config):
    env = config.get("environment", {})
    required_keys = ["profile", "roughness_length", "ref_height", "ref_windspeeds"]

    missing = [k for k in required_keys if k not in env]
    if missing:
        raise ValueError(f"Missing keys in 'environment': {missing}")

    profile = env["profile"].lower()
    if profile != "logarithmic":
        raise ValueError(f"Invalid profile type: '{profile}'. Only 'logarithmic' is supported.")

    try:
        roughness_length = float(env["roughness_length"])
        ref_height = float(env["ref_height"])
        ref_windspeeds = [float(w) for w in env["ref_windspeeds"]]
    except (TypeError, ValueError):
        raise ValueError("Environment values must be numeric.")

    return profile, roughness_length, ref_height, ref_windspeeds


def parse_electrical_etas(config):
    section = config.get("electrical_etas", {})
    required_blocks = ["motor_controller", "DC_AC_converter", "external_battery"]

    electrical_efficiency = {}

    for block in required_blocks:
        if block not in section:
            raise ValueError(f"Missing electrical component: '{block}'")
        
        entry = section[block]

        # Basic validation
        if "self_consumption" not in entry or "efficiency" not in entry:
            raise ValueError(f"Missing fields in '{block}' (need 'self_consumption' and 'efficiency')")

        # Convert values
        self_consumption = float(entry["self_consumption"])
        efficiency = float(entry["efficiency"])

        # External battery has an additional field
        connected = entry.get("connected", True if block != "external_battery" else None)
        if block == "external_battery":
            if connected is None:
                raise ValueError("'connected' field is required for 'external_battery'")
            connected = bool(connected)

        electrical_efficiency[block] = {
            "self_consumption": self_consumption,
            "efficiency": efficiency,
        }
        if block == "external_battery":
            electrical_efficiency[block]["connected"] = connected

    return electrical_efficiency

def parse_power_curve_smoothing(config):
    pcs = config.get("power_curve_smoothing", {})

    only_successful_opts = bool(pcs.get("only_successful_opts", True))
    smooth = bool(pcs.get("smooth", False))
    plot_results = bool(pcs.get("plot_results", False))
    end_index = pcs.get("end_index", None)

    def extract_dict(subsection_name):
        sub = pcs.get(subsection_name, {})
        if not isinstance(sub, dict):
            raise ValueError(f"'{subsection_name}' must be a dictionary")
        return dict(sub)  # ensures we return a copy

    fit_order = extract_dict("fit_order")
    ineq_tols = extract_dict("ineq_tols")
    index_offset = extract_dict("index_offset")
    
    fit_settings = {
        "fit_order": fit_order,
        "ineq_tols": ineq_tols,
        "index_offset": index_offset,
        "end_index": end_index
    }

    return only_successful_opts, smooth, plot_results, fit_settings


def parse_trajectory_etas(config):
    traj = config.get("trajectory_etas", {})

    if "efficiency" not in traj:
        raise ValueError("'efficiency' field is required in 'trajectory_etas' section.")
    try:
        trajectory_eta = float(traj["efficiency"])
    except (TypeError, ValueError):
        raise ValueError("'efficiency' in 'trajectory_etas' must be a number.")

    return trajectory_eta