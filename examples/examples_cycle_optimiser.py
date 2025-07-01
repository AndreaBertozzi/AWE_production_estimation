import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys

sys.path.append(os.path.abspath('./'))
from cycle_optimizer import OptimizerCycle
from qsm import LogProfile, TractionPhasePattern, SystemProperties
from utils import *

def example_1():          
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    sys_props = parse_system_properties_and_bounds(config)
    sys_props = SystemProperties(sys_props)

    env_state = LogProfile()
    env_state.set_reference_wind_speed(6.5)
    env_state.set_reference_height(100)
    env_state.set_roughness_length(0.07)

    cycle_sim_settings = {
        'cycle': {
            'traction_phase': TractionPhasePattern,
            'include_transition_energy': True,
        },
        'retraction': {'time_step': 0.5
                       },
        'transition': {'time_step': 0.5,
        },
        'traction': {'time_step': 0.25
        },
    }
    
    # with reduce_x we select which optimisation variables to consider: all the possible variables are in order:
    # F_RO, F_RI, theta_avg, theta_rel, max_phi, Lmax_RO - Lmin_RO, Lmin_RO
    # In this case we are optimising: F_RO, F_RI, theta_avg, Lmax_RO - Lmin_RO, Lmin_RO
    opt_var_enabled_idx, init_vals = parse_opt_variables(config)
    cons_enabled_idx, cons_param_vals = parse_constraints(config)

    oc = OptimizerCycle(cycle_sim_settings, sys_props, env_state, reduce_x = opt_var_enabled_idx,
                         reduce_ineq_cons=cons_enabled_idx, parametric_cons_values=cons_param_vals, force_or_speed_control='force')
    
    # A good initial condition is fundamental to speed up convergence
    oc.x0_real_scale = init_vals

    # iprint > 1 is verbose
    x_opt = oc.optimize(iprint=2, maxiter=30, ftol=1e-3, eps=1e-4)

    print('Opt. solution: ', x_opt)
    ylabs = (r'$L_{tether}$ [m]', r'$v_{reeling out}$ [m/s]', r'$F_{tether}$ [N]', r'$P_{ground}$ [W]')
    cons, kpis = oc.eval_point(True, x_real_scale=x_opt, labels=ylabs)
    print('Successful optimisation: ', oc.op_res['success'])    
   
    plt.show()

def example_2():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    sys_props = parse_system_properties_and_bounds(config)
    sys_props = SystemProperties(sys_props)

    env_state = LogProfile()
    env_state.set_reference_wind_speed(10.)

    cycle_sim_settings = {
        'cycle': {
            'traction_phase': TractionPhasePattern,
            'include_transition_energy': True,
        },
        'retraction': {'time_step': 0.5
                       },
        'transition': {'time_step': 0.5,
        },
        'traction': {'time_step': 0.5
        },
    }

    cons_enabled_idx, cons_param_vals = parse_constraints(config)
    oc = OptimizerCycle(cycle_sim_settings, sys_props, env_state, reduce_x = np.array([0, 1, 2, 5, 6]),
                         reduce_ineq_cons=cons_enabled_idx, parametric_cons_values=cons_param_vals, force_or_speed_control='speed')
    
    oc.x0_real_scale = np.array([3, -7., 0.5235,  9.5*np.pi/180, 25*np.pi/180,  100, 200])
    x_opt = oc.optimize(iprint=2, maxiter=30, ftol=1e-3)

    print('Opt. solution: ', x_opt)
    'straight_tether_length', 'reeling_speed', 'tether_force_ground', 'power_ground'
    ylabs = (r'$L_{tether}$ [m]', r'$v_{reeling out}$ [m/s]', r'$F_{tether}$ [N]', r'$P_{ground}$ [W]')
    cons, kpis = oc.eval_point(True, x_real_scale=x_opt, labels=ylabs)
    print('Successful optimisation: ', oc.op_res['success'])        
    print('Constraints: ', cons) 
    plt.show()

def example_3():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    sys_props = parse_system_properties_and_bounds(config)
    sys_props = SystemProperties(sys_props)

    env_state = LogProfile()
    env_state.set_reference_wind_speed(10.)

    cycle_sim_settings = {
        'cycle': {
            'traction_phase': TractionPhasePattern,
            'include_transition_energy': True,
        },
        'retraction': {'time_step': 0.5
                       },
        'transition': {'time_step': 0.5,
        },
        'traction': {'time_step': 0.5
        },
    }
    oc = OptimizerCycle(cycle_sim_settings, sys_props, env_state, reduce_x = np.array([0, 1, 2, 5, 6]),
                         reduce_ineq_cons=np.array([0, 1, 2, 3, 4, 5, 6]), force_or_speed_control='hybrid')
    
    oc.x0_real_scale = np.array([3, 2900., 0.5235,  9*np.pi/180, 32.5*np.pi/180,  100, 200])
    x_opt = oc.optimize(iprint=2, maxiter=30, ftol=1e-3)

    print('Opt. solution: ', x_opt)
    cons, kpis = oc.eval_point(True, x_real_scale=x_opt)
    
    print('Successful optimisation: ', oc.op_res['success'])        
    print('Constraints: ', cons) 
    plt.show()

def main():
    # Mapping from option number to example function
    example_funcs = {
        "1": example_1,
        "2": example_2,
        "3": example_3,
    }

    print("Examples about the QSM model\n")
    print("Select an example to run:")
    for i in range(1, 4):
        print(f"  {i}. example_{i}()")

    choice = input("\nEnter the number of the example to run (1-3), or 'q' to quit: ").strip()

    if choice.lower() == 'q':
        print("Exiting.")
        sys.exit(0)

    if choice in example_funcs:
        print(f"\nRunning example_{choice}()...\n")
        try:
            example_funcs[choice]()
        except Exception as e:
            print(f"An error occurred while running example_{choice}: {e}")
    else:
        print("Invalid choice. Please enter a number between 1 and 3.")

# Optional: run main if script is called directly
if __name__ == "__main__":
    main()