# -*- codiNatgas: utf-8 -*-
"""
  NOTE: This a generalized version but cost of everythiNatgas but Nuclear and battery and unmet demand
          are extremely high to examine tradeoffs amoNatgas them. Cases highly idealized.
"""

"""
Created on Fri Jan 26 21:40:43 2018

@author: Fan
"""

#!/usr/bin/env python2
# -*- codiNatgas: utf-8 -*-
"""
Created on Tue Dec 26 21:52:04 2017

@author: FanMacbook
"""

# -*- codiNatgas: utf-8 -*-
"""
Created on Mon Dec 11 17:24:28 2017

@author: Fan
"""

from Core_Model_kc180312 import core_model_loop
from Preprocess_Input_kc180328 import preprocess_input
#from Postprocess_Results_kc180214 import postprocess_key_scalar_results,merge_two_dicts
from Save_Basic_Results_kc180312 import save_basic_results

#%%

# directory = "D:/M/WORK/"
#root_directory = "/Users/kcaldeira/Google Drive/simple energy system model/Kens version/"
root_directory = "/Users/Ken/Google Drive/simple energy system model/20180316/"


# -----------------------------------------------------------------------------
hour_simulation_start = 0
#hours_of_simulation = 24  # 1 day
#hours_of_simulation = 240  # 10 days
#hours_of_simulation = 30*24  # 30 days
#hours_of_simulation = 240  # 10 days
hours_of_simulation = 8640 # 1 year

verbose = True # print output on each loop
# =============================================================================
# Set base case switch

#base_case_switch = 'realistic'
#base_case_switch = 'idealized'

# -----------------------------------------------------------------------------
# Set case switch
#case_switch = 'default'
#case_switch = 'nuc_bat_unmet'
#case_switch = 'gas_bat_unmet'
#case_switch = 'solar_bat_unmet'
#case_switch = 'wind_bat_unmet'
#case_switch = 'solar_wind_bat_unmet'
#case_switch = 'gas_solar_wind_bat_unmet'
#case_switch = 'nuc_solar_wind_bat_unmet'
#case_switch = 'gas_nuc_bat_unmet'
#case_switch = 'all'
#case_switch = 'all_2years'

# =============================================================================


case_list = [
        'natgas_storage'
#        'wind_storage'
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
#         'natgas_bat',
#         'solar_wind_bat'
#         'nuc_bat'
#        'all'
#         'nate2_store_big_num',
#         'idealized2'
#         'nate_spectrum'
#        'battery_natgas'
#        'battery_nuclear'
#        'nate2_battery',
#        'nate2_solar'
#         'test_bat'
#        'test3'
        ]
# =============================================================================
for case_switch in case_list:
    
    file_info, time_series, assumption_list = preprocess_input(
        root_directory,
        case_switch,
        hour_simulation_start,
        hours_of_simulation,
        verbose
        )  

    result_list = core_model_loop (
        time_series,
        assumption_list,
        verbose
        )

    scalar_names,scalar_table = save_basic_results(
        file_info,
        time_series,
        assumption_list,
        result_list,
        verbose) 

             
