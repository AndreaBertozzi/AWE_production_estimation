import sys, os

sys.path.append(os.path.abspath('./'))
from awe_pe.cycle_optimizer import OptimizerCycle
from awe_pe.qsm import LogProfile, TractionPhasePattern, SystemProperties
from awe_pe.utils import *

import pandas as pd

with open("config/config_V3_speed.yaml") as f:
    config = yaml.safe_load(f)

sys_props = parse_system_properties_and_bounds(config)
sys_props = SystemProperties(sys_props)

env_state = LogProfile()

env_state.set_reference_wind_speed(5.654951533540675)
env_state.set_reference_height(6)
env_state.set_roughness_length(0.1)

env_state.plot_wind_profile()

cycle_sim_settings = {
    'cycle': {
        'traction_phase': TractionPhasePattern,
        'include_transition_energy': True,
    },
    'retraction': {'time_step': 0.25
                    },
    'transition': {'time_step': 0.25,
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
                        reduce_ineq_cons=range(12), parametric_cons_values=cons_param_vals,
                          force_or_speed_control='hybrid')

opt_sol = [1.33785903e+00, 8.82540000e+02, 5.23598776e-01, 8.72664626e-02,
 4.57444322e-01, 1.00000000e+02, 2.00000000e+02]
# A good initial condition is fundamental to speed up convergence
oc.x0_real_scale = opt_sol
# iprint > 1 is verbose

import time

# Start measuring CPU time
start_time = time.process_time()

x_opt = oc.optimize(iprint=2, maxiter=30, ftol=1e-6, eps=1e-6)

end_time = time.process_time()

# Calculate CPU time used
cpu_time = end_time - start_time
print(f"CPU time: {cpu_time} seconds")

print('Opt. solution: ', x_opt)
ylabs = (r'$L_{tether}$ [m]', r'$v_{RO}$ [m/s]', r'$F_{tether}$ [N]', r'$P_{ground}$ [W]')
cons, kpis = oc.eval_point(True, x_real_scale=x_opt, labels=ylabs)
print('Successful optimisation: ', oc.op_res['success'])

plt.show()