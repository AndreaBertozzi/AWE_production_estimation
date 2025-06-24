import numpy as np
import matplotlib.pyplot as plt
import yaml
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
    x_opt = oc.optimize(iprint=2, maxiter=30, ftol=1e-3)

    print('Opt. solution: ', x_opt)
    ylabs = (r'$L_{tether}$ [m]', r'$v_{reeling out}$ [m/s]', r'$F_{tether}$ [N]', r'$P_{ground}$ [W]')
    cons, kpis = oc.eval_point(True, x_real_scale=x_opt, labels=ylabs)
    print('Successful optimisation: ', oc.op_res['success'])    
   
    plt.show()

def example_2():
    # TODO: Understand why this does not converge  
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
    # TODO: Understand why this does not converge          
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

if __name__ == "__main__":
    example_1()
    
    # EXAMPLE OUTPUT FROM TEST:
    """
    Optimization terminated successfully    (Exit mode 0)
                Current function value: -0.7207391615666787
                Iterations: 12
                Function evaluations: 78
                Gradient evaluations: 12
    Opt. solution:  [2.28363626e+04 3.18573904e+03 5.23598776e-01 1.57079633e-01
    5.67232007e-01 1.49998266e+02 2.48020651e+02]
    Successful optimisation:  True
    Constraints:  [9.99999964e-07 1.00000000e-06 1.53159082e-01 3.01644069e+00
    2.10266713e+01 1.98108317e+00 5.57715138e-06]
    """