"""
file name: Script_Model.py
    set up simulation and cases to run
    note: run simulations from this file
    
authors:
    Fan (original model)
    Ken (reorganized model)
    Mengyao (nuclear vs. renewables)

version: _my180406
    modified from _my180403 for preliminary analysis of nuclear vs. renewables 
    
updates:    
    (a) updated case list
    (b) changed folder and filenames to be consistent with this version
    
to-dos:
    change simulation time to 8760 hours

"""

#%% import function

from Core_Model_my180406 import core_model_loop
from Preprocess_Input_my180406 import preprocess_input
from Save_Basic_Results_my180406 import save_basic_results

#%% set up simulation

# root directory for reading data and saving results

root_directory = 'D:/Mengyao @ Carnegie/research/models/model_my180406/'

# -----------------------------------------------------------------------------
# length of time simulated

hour_simulation_start = 0
#hours_of_simulation = 24        # 1 day
#hours_of_simulation = 7*24      # 1 week
#hours_of_simulation = 240       # 10 days
#hours_of_simulation = 30*24     # 30 days
#hours_of_simulation = 240       # 10 days
hours_of_simulation = 8640      # 1 year

verbose = True                  # print output on each loop

# -----------------------------------------------------------------------------
# select cases to run simulations

base_case_switch = 'idealized'
# cases for nuclear vs. renewables analysis
case_list = [
        'ng_flex_nuc',      # baseline natural gas, flexible nuclear
        'ng_fixed_nuc',     # baseline natural gas, fixed nuclear
        'ngccs_flex_nuc',   # natural gas + CCS, flexible nuclear
        'ngccs_fixed_nuc',  # natural gas + CCS, fixed nuclear
        'no_ng_flex_nuc',   # no natural gas, flexible nuclear
        'no_ng_fixed_nuc'   # no natural gas, fixed nuclear
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
