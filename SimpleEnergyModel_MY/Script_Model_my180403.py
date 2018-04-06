"""
file name: Script_Model.py
    set up simulation and cases to run
    note: run simulations from this file
    
authors:
    Fan (original model)
    Ken (reorganized model)
    Mengyao (nuclear vs. renewables)

version: _my180403
    modified from _kc180312 for preliminary analysis of nuclear vs. renewables 
    
updates:    
    added cases for nuclear vs. renewables
    cleaned up code and comments
    
to-dos:
    confirm length of time simulated

"""

#%% import function

from Core_Model_my180403 import core_model_loop
from Preprocess_Input_my180403 import preprocess_input
#from Postprocess_Results_my180403 import postprocess_key_scalar_results,merge_two_dicts
from Save_Basic_Results_my180403 import save_basic_results

#%% set up simulation

# root directory for reading data and saving results

root_directory = 'D:/Mengyao @ Carnegie/research/models/model_my180403/'

# -----------------------------------------------------------------------------
# length of time simulated

hour_simulation_start = 0
hours_of_simulation = 24        # 1 day
#hours_of_simulation = 240       # 10 days
#hours_of_simulation = 30*24     # 30 days
#hours_of_simulation = 240       # 10 days
#hours_of_simulation = 8640      # 1 year

verbose = True                  # print output on each loop

# -----------------------------------------------------------------------------
# select cases to run simulations

base_case_switch = 'idealized'
case_list = [
        'nuc_vs_renew_ng',
        'nuc_vs_renew_ngccs',
        'nuc_vs_renew_no_ng'
#        'natgas_storage',
#        'wind_storage',
#        'nuclear_storage',
#        'solar_storage'
#        'nuc_bat'
#        'solar_wind_bat'
#        'nuclear_solar_wind_bat'
#        'gas_bat'
#        'solar_bat',
#        'wind_bat'
#        'solar_wind_bat_unmet',
#        'gas_solar_wind_bat_unmet',
#        'nuc_solar_wind_bat_unmet',
#        'natgas_bat',
#        'solar_wind_bat'
#        'nuc_bat'
#        'all'
#        'nate2_store_big_num',
#        'idealized2'
#        'nate_spectrum'
#        'battery_natgas'
#        'battery_nuclear'
#        'nate2_battery',
#        'nate2_solar'
#        'test_bat'
#        'test3'
        ]
        
# -----------------------------------------------------------------------------
        
for case_switch in case_list:
    
    file_info, time_series, assumption_list = preprocess_input(
        root_directory,
        base_case_switch,
        case_switch,
        hour_simulation_start,
        hours_of_simulation,
        verbose
        )  

    result_list = core_model_loop(
        time_series,
        assumption_list,
        verbose
        )

    scalar_names, scalar_table = save_basic_results(
        file_info,
        time_series,
        assumption_list,
        result_list,
        verbose)            
