import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from power_curve_constructor import PowerCurveConstructor
from power_curves import compare_kpis
from utils import *
from qsm import *
from cycle_optimizer import OptimizerCycle

data_path = 'Results/'
tols = ['1e-3', '1e-4', '1e-5', '1e-6']

pcs = []
lgd = []

for tol in tols:
    pc = PowerCurveConstructor(None)    
    pc.import_results(data_path + 'tol' +  tol + '/power_curve_log_profile.pickle')
    #pc.plot_optimal_trajectories()
    #pc.plot_optimization_results()
    lgd.append('ftol: ' + tol)
    pcs.append(pc)

pc = PowerCurveConstructor(None)    
pc.import_results(data_path + 'power_curve_log_profile.pickle')
pcs.append(pc)
lgd.append('tight')
    

#compare_kpis(pcs)
#plt.legend(lgd)
#plt.show()

idx = 40
print('Wind speed:', pcs[4].wind_speeds[idx])
cycle_set = {
        'cycle': {
            'traction_phase': TractionPhasePattern,
            'include_transition_energy': True,
        },
        'retraction': {
            'time_step': 0.25,},

        'transition': {
            'time_step': 0.25,
        },
        'traction': {
            'time_step': 0.25,
        },
    }
sys_props = SystemProperties(load_config('config.yaml'))
env_state = LogProfile()
env_state.set_reference_height(100)
env_state.set_reference_wind_speed(pcs[4].wind_speeds[idx])

oc = OptimizerCycle(cycle_set, sys_props, env_state, reduce_x=np.array([0, 1, 2, 5, 6]))
oc.eval_point(True, False, pcs[4].x_opts[idx])

plt.show()