import pickle 
import pprint
from qsm import *
from power_curves import *

def load_power_curve_results_and_plot_trajectories(loc='mmc', n_clusters=1, i_profile=1, result_path=''):
    """Plot trajectories from previously generated power curve."""
    pc = PowerCurveConstructor(None)
    suffix = '_{}{}{}'.format(n_clusters, loc, i_profile)
    pc.import_results(result_path + '/power_curve{}.pickle'.format(suffix))
    pc.plot_optimal_trajectories()
    plt.gcf().set_size_inches(5.5, 3.5)
    plt.subplots_adjust(top=0.99, bottom=0.1, left=0.12, right=0.65)
    pc.plot_optimization_results(opt_variable_labels=[r'$F_o$', r'$F_i$', r'$\theta_o$', 'stroke', r'$L_{min}$'])
    plt.gcf().set_size_inches(10.5, 5.5)
    return pc

    

results_folder = 'Results_AWE_prod_model/V2_kite/'
file_name = 'power_curve_1mmc1.pickle' # This is common for both models 
liss_path = 'Lissajous_no_trac_fixed_alpha/phi_o_12/'  # This identifies 
                                                      # the model leveraging Lissajous pattern  
straight_path = 'Straight_no_trac_fixed_alpha/phi_o_12/'  # This identifies 
                                                      # the model leveraging Lissajous pattern  

# Use import_results to import results as dict
pc1 = load_power_curve_results_and_plot_trajectories(result_path= results_folder + liss_path)
pc2 = load_power_curve_results_and_plot_trajectories(result_path= results_folder + straight_path)

plt.show()