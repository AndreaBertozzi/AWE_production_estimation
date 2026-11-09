import sys, os

sys.path.append(os.path.abspath('./'))
from awe_pe.cycle_optimizer import OptimizerCycle
from awe_pe.qsm import LogProfile, TractionPhasePattern, SystemProperties
from awe_pe.utils import *
from awe_pe.exp_validation_utils import pack_operational_parameters_and_results, pack_results_sim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})
        
with open("config/config_V3_speed.yaml") as f:
    config = yaml.safe_load(f)

sys_props = parse_system_properties_and_bounds(config)
sys_props = SystemProperties(sys_props)

env_state = LogProfile()

env_state.set_reference_wind_speed(5.654951533540675)
env_state.set_reference_height(6)
env_state.set_roughness_length(0.1)

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

x_opt = [1.36214107e+00, 8.82540000e+02, 6.10865238e-01, 7.39951045e-02,
 4.12125197e-01, 9.00000000e+01, 2.17000000e+02]

x_opt2 = [1.34898359e+00, 1.00021200e+03, 6.10865238e-01, 7.49756265e-02,
 4.20550891e-01, 9.00000000e+01, 2.17000000e+02]


cons, kpis = oc.eval_point(False, x_real_scale=x_opt)

power = [ss.power_ground for ss in kpis['steady_states']]
tether_length =  [kin.straight_tether_length for kin in kpis['kinematics']]
tether_force = [ss.tether_force_ground for ss in kpis['steady_states']]
reeling_speed = [ss.reeling_speed for ss in kpis['steady_states']]
time = kpis['time']
course_angle = [kin.course_angle for kin in kpis['kinematics']]
course_rate = np.diff(course_angle)
# plt.plot(cons)
# plt.show()
# plt.plot(time[:-1], course_rate)
# plt.xlim([0, 65])
# plt.ylim([-0.4, 0.4])
# plt.show()


baseline = pd.read_csv('baseline_reference_sim.csv')

sim_ph_power = np.mean(baseline.ground_tether_force*baseline.ground_tether_reelout_speed)
baseline_phase = baseline[baseline.flight_phase_index == 1]
sim_ph_power_RO = np.mean(baseline_phase.ground_tether_force*baseline_phase.ground_tether_reelout_speed)
baseline_phase = baseline[baseline.flight_phase_index == 3]
sim_ph_power_RI = np.mean(baseline_phase.ground_tether_force*baseline_phase.ground_tether_reelout_speed)

baseline_phase = baseline[baseline.flight_phase_index == 4]
sim_ph_power_RIRO = np.mean(baseline_phase.ground_tether_force*baseline_phase.ground_tether_reelout_speed)


baseline_values = [
    sim_ph_power_RO/1000,
    sim_ph_power_RI/1000,
    sim_ph_power_RIRO/1000,
    sim_ph_power/1000,
]

opt_values = [
    kpis['average_power']['out']/1000,
    kpis['average_power']['in']/1000,
    kpis['average_power']['trans']/1000,
    kpis['average_power']['cycle']/1000,
]


fig = plt.figure(figsize=(18,4))
font_size = 22
gs = gridspec.GridSpec(2, 3, height_ratios=[1,1], width_ratios=[1.2,1.2,1])

# ---------------------------
# First subplot (1,3,1)
# ---------------------------
plt.rcParams.update({'font.size': font_size, 'axes.linewidth': 1.5})

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(baseline.time - baseline.iloc[0].time, baseline.ground_tether_length, color = "C01", linewidth = 2, label='Reference')
ax1.plot(time, tether_length, color = 'green', linewidth = 2, label = 'Optimized')
ax1.set_ylabel(r'$L$ (m)')
ax1.set_xticks([0, 20, 40, 60, 80, 100, 120])
ax1.set_xticklabels([])
ax1.text(-5, 290, s='a)', fontdict={'fontsize': font_size})

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(baseline.time - baseline.iloc[0].time, baseline.ground_tether_reelout_speed, color = "C01", linewidth = 2, label='Reference')
ax2.plot(time, reeling_speed, color = 'green', linewidth = 2, label='Optimized')
ax2.set_ylabel(r'$v_\mathrm{r}$ (m/s)')
ax2.set_xticks([0, 20, 40, 60, 80, 100, 120])
ax2.set_xticklabels([])
ax2.text(-5, .2, s='b)', fontdict={'fontsize': font_size})
leg = ax2.legend(
    edgecolor='black',
    framealpha=1.0,     # fully opaque
    fancybox=False,
    facecolor='white',
    draggable = True,
    ncols = 2
)
leg.get_frame().set_linewidth(1.5)

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(baseline.time - baseline.iloc[0].time, baseline.ground_tether_force/1000, color = "C01", linewidth = 2, label='Reference')
ax3.plot(time, np.array(tether_force)/1000, color = 'green', linewidth = 2)

ax3.hlines([sys_props.tether_force_min_limit/1000], time[0], max(baseline.time - baseline.iloc[0].time), colors='gray', linestyles= '--',)
ax3.text(time[0], sys_props.tether_force_min_limit/1000 + 0.2, s=r'$F_\mathrm{t}$ lower bound', fontdict={'fontsize':18})

ax3.text(-5, 5.3, s='c)', fontdict={'fontsize': font_size})
ax3.set_xticks([0, 20, 40, 60, 80, 100, 120])
ax3.set_yticks([1, 3, 5])
ax3.set_ylabel(r'$F_\mathrm{t}$ (kN)')
ax3.set_xlabel(r'Time (s)')

ax4 = fig.add_subplot(gs[1, 1])   # spans both rows, first column
ax4.plot(baseline.time - baseline.iloc[0].time, baseline.ground_mech_power/1000, color = "C01", linewidth = 2,  label='Optimized')
ax4.plot(time, np.array(power)/1000, color = 'green', linewidth = 2)
ax4.set_ylabel(r'Power (kW)')
ax4.set_xlabel(r'Time (s)')
ax4.text(-5, 5.6, s='d)', fontdict={'fontsize': font_size})
ax4.set_xticks([0, 20, 40, 60, 80, 100, 120])
gs_right = gridspec.GridSpecFromSubplotSpec(
    3, 1,
    subplot_spec=gs[:, 2],
    height_ratios=[0.3, 1, 0.3]
)
ax1 = fig.add_subplot(gs_right[1])

labels = ["RO", "RI", "RIRO", "Cycle"]


gains = (np.array(opt_values)-np.array(baseline_values))/np.array(baseline_values)

x = np.arange(len(labels))
width = 0.35
# Send grid to the background
ax1.set_axisbelow(True)
ax1.grid(True ,axis='y')  # Show grid, but send it to the background
ax1.bar(x + width / 2, baseline_values, width, label="Reference", linewidth = 1.5, edgecolor='black' , facecolor='C01')
ax1.bar(x - width / 2, opt_values, width, label="Optimized", linewidth = 1.5, edgecolor='black', facecolor = 'green')

for ii, x_pos in enumerate(x):
    if opt_values[ii] >= 0:
        y_pos = np.max([baseline_values[ii], opt_values[ii]])
        ax1.text(x_pos-width-0.15, y_pos+0.4, rf"$\Delta \approx {gains[ii]*100:+.0f}\%$", fontsize=18)
    else:
        y_pos = np.min([baseline_values[ii], opt_values[ii]])
        ax1.text(x_pos- width-0.15, y_pos-1.5, rf"$\Delta \approx {gains[ii]*100:+.0f}\%$", fontsize=18)

    

xlb, xub = ax1.get_xlim()
ax1.hlines(0, xlb, xub, colors=['black'], linewidth = 1.5)
# Adding labels, title, and formatting
# ax1.set_xlabel("Power Metrics")
ax1.set_ylabel("Power (kW)")
#ax1.set_title("Comparison of average mechanical power")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim([-5.3, 6.8])
leg = ax1.legend(
    edgecolor='black',
    framealpha=1.0,     # fully opaque
    fancybox=False,
    facecolor='white',
    draggable = True
)
leg.get_frame().set_linewidth(1.5)
ax1.text(-0.7, 5., s='e)', fontdict={'fontsize': font_size})
plt.tight_layout()



print(np.mean(power), np.mean(baseline.ground_mech_power))
plt.show()