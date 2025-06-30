import os
import sys
sys.path.append(os.path.abspath('./'))
from qsm import *
from utils import *

# Simulation of the complete pumping cycle simulation using the quasi steady model. System properties are declared as dictionary
# The environmental state is represented by a logarithmic profile. 
# The traction phase of the cycle is simplified to a straight line (with fixed elevation, azimuth, and course angles) 
# representative of the crosswind fligth of the kite
def example_1():
    # Create environment object.
    env_state = LogProfile()
    # Set reference height and wind speed
    env_state.set_reference_height(100)
    env_state.set_reference_wind_speed(7.0)

    # System properties can either be declared using a dictionary passed to the SystemProperties constructor, or read from the config.yaml file, see next example.
    # Create system properties object (setting only some properties here, but all the properties can be seen running ```print(sys_props.__dict__)``` after call to
    # the constructor)

    sys_props = {
            'kite_projected_area': 18, # [m^2]
            'kite_mass': 20,  # [kg]
            'tether_density': 724,  # [kg/m^3]
            'tether_diameter': 0.004,  # [m]
            'tether_force_min_limit': 300, # [N]
    }

    sys_props = SystemProperties(sys_props)
    # In addition to system properties, operational settings should be introduced. This is how we fly the kite. In this example, we are using a simplified
    # description of the traction phase (TractionPhase), and we include transition energy in the final power calculations.
    # We set the max and min tether lenght, the average elevation during traction. 
    # During retraction, we aim for a constant tether tension of tether_force_ground, during transition we aim for 0. reeling speed (fly down on the sphere of radius
    # tether_length_end_retraction), during traction we aim for a certain reeling factor (defined as v_reeling_out/v_wind). All the "control modes" are interchangable
    # within phases. For reel-out, the representative state is completed by a course angle and an azimuth angle, in addition to the elevation_angle_traction set in
    # cycle settings. Setting the elevation angle in the phase settings is possible but more cumbersome
    settings = {
            'cycle': {
                'traction_phase':                   TractionPhase,
                'include_transition_energy':        True,
                'tether_length_start_retraction':   385.,      
                'tether_length_end_retraction':     240.,  
                'elevation_angle_traction':         25.0*np.pi/180
            },
            'retraction': {
                'control': ('tether_force_ground', 1200),
            },
            'transition': {
                'control': ('reeling_speed', 0.),
            },
            'traction': {
                'control': ('reeling_factor', 0.2),
                'course_angle': 110*np.pi/180,
                'azimuth_angle': 12*np.pi/180,
                'time_step': .05,
            },
        }


    # Create pumping cycle simulation object, run simulation, and plot results.
    cycle = Cycle(settings)
    cycle.follow_wind = True 
    # Specifies whether kite is 'aligned' with the wind. Controlled azimuth angle is 
    # expressed w.r.t. wind reference frame if True, or ground reference frame if False.

    # TODO: Investigate the use of this function to check productivity with wind misalignment

    cycle.run_simulation(sys_props, env_state, print_summary=True)

    # Plotting different quantities of interests
    cycle.time_plot(('reeling_speed', 'power_ground', 'apparent_wind_speed', 'tether_force_ground'),
                        ('Reeling speed [m/s]', 'Power [W]', 'Apparent wind speed [m/s]', 'Tether force [N]'))

    # Plotting other quantitities, and scaling the angles to have them in degrees
    cycle.time_plot(('straight_tether_length', 'elevation_angle', 'azimuth_angle', 'course_angle'),
                        ('Tether lenght [m]', 'Elevation [deg]', 'Azimuth [deg]', 'Course [deg]'),
                        (None, 180./np.pi, 180./np.pi, 180./np.pi))

    # Plotting the trajectory
    cycle.trajectory_plot(steady_state_markers=True)
    cycle.trajectory_plot3d()
    plt.show()

# Simulation of the complete pumping cycle simulation using the quasi steady model. System properties are loaded from the config.yaml file.
# The environmental state is represented by a logarithmic profile. 
# The traction phase of the cycle is represented by a lemniscate (figure of eight trajectory)
def example_2():
    # Create environment object.
    env_state = LogProfile()
    # Set reference height and wind speed
    env_state.set_reference_height(100)
    env_state.set_reference_wind_speed(8.0)

    # Load system properties from config.yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    sys_props = parse_system_properties_and_bounds(config)
    sys_props = SystemProperties(sys_props)

    # In terms of operational settings, in this example, we are using a simplified representation of an actual figure eight pattern based on 
    # Lissajous pattern (TractionPhasePattern). Such pattern is governed by a dictionary passed to the traction settings, with relative elevation
    # and max azimuth angle. 

    settings = {
            'cycle': {
                'traction_phase':                   TractionPhasePattern,
                'include_transition_energy':        True,
                'tether_length_start_retraction':   350.,      
                'tether_length_end_retraction':     250.,  
                'elevation_angle_traction':         30*np.pi/180
            },
            'retraction': {
                'control': ('tether_force_ground', 3000),
            },
            'transition': {
                'control': ('reeling_speed', 0.),
            },
            'traction': {
                'control': ('reeling_speed', 1.2),
                'pattern': {'rel_elevation_angle': 9.2*np.pi/180, 'azimuth_angle': 35*np.pi/180},
                'time_step': .05,
            },
        }
    
    # Create pumping cycle simulation object, run simulation, and plot results.
    cycle = Cycle(settings)
    cycle.run_simulation(sys_props, env_state, print_summary=True)

    # Plotting different quantities of interests
    cycle.time_plot(('reeling_speed', 'power_ground', 'apparent_wind_speed', 'tether_force_ground'),
                        ('Reeling speed [m/s]', 'Power [W]', 'Apparent wind speed [m/s]', 'Tether force [N]'))

    # Plotting other quantitities, and scaling the angles to have them in degrees
    cycle.time_plot(('straight_tether_length', 'elevation_angle', 'azimuth_angle', 'course_angle'),
                        ('Tether length [m]', 'Elevation [deg]', 'Azimuth [deg]', 'Course [deg]'),
                        (None, 180./np.pi, 180./np.pi, 180./np.pi))

    # Plotting the trajectory
    cycle.trajectory_plot3d()
    plt.show()

def main():
    # Mapping from option number to example function
    example_funcs = {
        "1": example_1,
        "2": example_2,
    }

    print("Examples about the QSM model\n")
    print("Select an example to run:")
    for i in range(1, 3):
        print(f"  {i}. example_{i}()")

    choice = input("\nEnter the number of the example to run (1-2), or 'q' to quit: ").strip()

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
        print("Invalid choice. Please enter a number between 1 and 2.")

# Optional: run main if script is called directly
if __name__ == "__main__":
    main()