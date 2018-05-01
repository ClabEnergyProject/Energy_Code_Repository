"""
file name: Preprocess_Input.py
    list of parameters (costs, efficiencies, etc.) and assumptions for all cases
    
authors:
    Fan (original model)
    Ken (reorganized model)
    Mengyao (nuclear vs. renewables)

version: _my180406
    modified from _my180403 for preliminary analysis of nuclear vs. renewables
    
updates:
    added flag (flex_nuc_flag) for choosing constraints for nuclear depending on whether nuclear is assumed to be flexible or fixed

to-dos:
    (a) vary baseline fixed and variable costs - currently using EIA's assumptions - see Fan's notes "generator and storage LCOE" - other sources?
    (b) read "original" cost and efficiency data as input from .csv files (instead of "hardwiring" assumptions)?
    (c) use consistent discounting method - currently using mix of exponential and linear methods
    (d) confirm units of costs used in script and output files - see notes
    (e) normalize wind and solar capacity to demand? - clarify: same as setting dispatch == capacity * capacity_factor?
    (f) put cost assumptions inside each "if" section instead of having set of one-for-all costs    
    
"""

#%% import modules

import os
import numpy as np
import itertools, functools, operator
import datetime

#%% constants

# extreme values for testing hypotheses
small_num = 1e-10
big_num = 1e10

#%% function: define input parameters and assumptions

def preprocess_input(
        root_directory,
        base_case_switch,
        case_switch,
        hour_simulation_start,
        hours_of_simulation,
        verbose
        ):

    # data input folder
    input_folder = root_directory + 'Input_Data\\'
    
    # results folder
    today = datetime.datetime.now()
    output_folder = root_directory + 'Results/' + base_case_switch + '_' + case_switch + ' ' + \
        str(today.year) + str(today.month).zfill(2) + str(today.day).zfill(2) + '_' + \
        str(today.hour).zfill(2) + str(today.minute).zfill(2) + str(today.second).zfill(2)
    os.makedirs(output_folder)

    # -------------------------------------------------------------------------    
    
    # number of hours in a year
        # time resolution is hourly
        # maximum 30 years data available
    hour_in_1_year = 365*24     # one year = 8760 hours
    
    # define end time point
    start_time = hour_simulation_start
    end_time = hour_simulation_start + hours_of_simulation
    
    #%% energy systems setup
    
    # default parameters for basecases (realistic vs. idealized)
        # reference: https://www.bv.com/docs/reports-studies/nrel-cost-report.pdf
        # note to self: confirm units in comments
      
    if base_case_switch == 'realistic':
        
        system_components = ['natural_gas','wind','solar','nuclear','storage','unmet_demand']
        
        # power generation - fixed costs
        # note to self: check how capital charge rate is defined - why square root?
        capital_charge_rate_power = (1+0.08)**(1. / hour_in_1_year ) -1.    # capital recovery rate = 8% per year (discount rate = 7% per year, lifetime = 30 yrs)
        fix_cost_natgas = capital_charge_rate_power * np.array([1000]) + np.array([10])/hour_in_1_year  # $/kW per hour
        fix_cost_wind = capital_charge_rate_power * np.array([1500]) + np.array([45])/hour_in_1_year    # $/kW per hour
        fix_cost_solar = capital_charge_rate_power * np.array([1500]) + np.array([20])/hour_in_1_year   # $/kW per hour
        fix_cost_nuclear = capital_charge_rate_power * np.array([6000]) + np.array([100])/hour_in_1_year # $/kW per hour
        
        # power generation - variable costs
        natgas_fuel_cost = np.array([5])        # $/mmBTU [typically 5]
        efficiency_natgas = 0.55
        var_cost_natgas = natgas_fuel_cost / 293.07 / efficiency_natgas     # 293.07 = mmBTU to kWh conversion, $.kWh
        var_cost_wind = np.array([0])           # these micro values are just to get a dispatch order
        var_cost_solar = np.array([0])
#        nuclear_fuel_cost = 2.3/1000 #$/kWh
        var_cost_nuclear  = np.array([0.0023])  # $/kWh
        var_cost_unmet_demand = np.array([10])
        
        # storage - fixed costs
        capital_charge_rate_storage = (1+0.08)**(1. / hour_in_1_year ) -1.  # 8% per year
        fix_cost_storage = capital_charge_rate_storage * np.array([1500])+ np.array([25.2])/hour_in_1_year  # $/kWh (based on NREL-COST-REPORT.PDF, Table 37)

        # storage - variable costs
        storage_charging_efficiency = np.array([0.86])
        var_cost_storage = np.array([10**-1,10**-2,10**-3,10**-4,10**-6,10**-8])    # $/kWh dispatched
        var_cost_dispatch_to_storage = np.array([0])    # $/kWh (small value to prevent unnecessary use of battery)
        var_cost_dispatch_from_storage = np.array([0])  # $/kWh (small value to prevent unnecessary use of battery)
    
        # unmet demand - variable costs
#        var_cost_unmet_demand = np.linspace(0,0.3,31)
#        var_cost_unmet_demand = np.array([6,15])       # $/kWh
        
    elif base_case_switch == 'idealized':

        system_components = ['natural_gas','wind','solar','nuclear','storage','unmet_demand']
        
        fix_cost_natgas  = np.array([0.010])    # $/kW per hour
        fix_cost_wind    = np.array([0.013])    # $/kW per hour
        fix_cost_solar   = np.array([0.013])    # $/kW per hour
        fix_cost_nuclear = np.array([0.050])    # $/kW per hour
        fix_cost_storage = np.array([0.016])    # $/kW per hour
    
        var_cost_natgas  = np.array([0.030])    # $/kWh
        var_cost_wind = np.array([2e-6])
        var_cost_solar = np.array([1e-6])
#        var_cost_wind    = np.array([0.002])    # $/kWh
#        var_cost_solar   = np.array([0.001])    # $/kWh
        var_cost_nuclear  = np.array([0.001])   # $/kWh
        var_cost_unmet_demand = np.array([10])  # $/kWh 
        var_cost_storage = np.array([1e-8])     # $/kWh (small value to prevent unnecessary use of battery)
        var_cost_dispatch_to_storage = np.array([small_num])    # $/kWh (small value to prevent unnecessary use of battery)
        var_cost_dispatch_from_storage = np.array([small_num])  # $/kWh (small value to prevent unnecessary use of battery)
    
        storage_charging_efficiency = np.array([1.])

#        fix_cost_natgas  = np.array([0.0])  # $/kW per hour
#        fix_cost_wind    = np.array([0.0])  # $/kW per hour
#        fix_cost_solar   = np.array([0.0])  # $/kW per hour
#        fix_cost_nuclear = np.array([0.0])  # $/kW per hour
#        fix_cost_storage = np.array([0.0])  # $/kW per hour
#    
#        var_cost_natgas  = np.array([0.0])  # $/kWh
#        var_cost_wind    = np.array([0.0])  # $/kWh
#        var_cost_solar   = np.array([0.0])  # $/kWh
#        var_cost_nuclear  = np.array([0.0]) # $/kWh
#        var_cost_unmet_demand = np.array([0.0])     # $/kWh 
#        var_cost_storage = np.array([0.0])          # $/kWh (small value to prevent unnecessary use of battery)
#        var_cost_dispatch_to_storage = np.array([0.0])      # $/kWh (small value to prevent unnecessary use of battery)
#        var_cost_dispatch_from_storage = np.array([0.0])    # $/kWh (small value to prevent unnecessary use of battery)
#    
#        storage_charging_efficiency = np.array([0.0])

    else:
        print 'no base configuration specified'
        exit()
    
    # -------------------------------------------------------------------------
    # by default, search for capacities of everything and use time_series demand data
    
    optimize_flag = -1  # just important to be < 0
    capacity_natgas = np.array([optimize_flag])     # set to numbers if goal is to set capacity rather than optimize it
    capacity_wind = np.array([optimize_flag])       # set to numbers if goal is to set capacity rather than optimize it
    capacity_solar = np.array([optimize_flag])      # set to numbers if goal is to set capacity rather than optimize it
    capacity_nuclear = np.array([optimize_flag])    # set to numbers if goal is to set capacity rather than optimize it
    capacity_storage = np.array([optimize_flag])    # set to numbers if goal is to set capacity rather than optimize it
    demand_flag = np.array([-1]) # use time series or constant, use time series if < 0
#    demand_flag = np.array([1]) # use a constant of that value
    
    # -------------------------------------------------------------------------
    # specific / modified parameters for each case
    
    # mutlipliers for sensitivity analysis, sorted in descending order in place
        # sorting approach from: https://stackoverflow.com/questions/26984414/efficiently-sorting-a-numpy-array-in-descending-order
    sa_fix_cost_nuclear = np.array([2,1,0.5,0.3,0.1,0.005,0.001])
#    sa_var_cost_nuclear = np.array([2,1,0.5,0.3,0.1,0.005,0.001])
#    sa_fix_cost_nuclear = np.append([2,1.5,0.005,0.001],np.linspace(1,0.1,10))
#    sa_fix_cost_nuclear[::-1].sort()
#    sa_var_cost_nuclear = np.append([2,1.5,0.005,0.001],np.linspace(1,0.1,10)) 
#    sa_var_cost_nuclear[::-1].sort()
    
    if case_switch == 'ng_flex_nuc':
        system_components = ['natural_gas','wind','solar','nuclear','storage']
        flex_nuc_flag = np.array([1])   # nuclear is fully flexible / "rampable"
        fix_cost_nuclear  = sa_fix_cost_nuclear * fix_cost_nuclear  # vary nuclear fixed costs
#        var_cost_nuclear = sa_var_cost_nuclear * var_cost_nuclear   # vary nuclear variable costs
    elif case_switch == 'ng_fixed_nuc':
        system_components = ['natural_gas','wind','solar','nuclear','storage']
        flex_nuc_flag = np.array([0])   # nuclear is fixed at capacity
        fix_cost_nuclear  = sa_fix_cost_nuclear * fix_cost_nuclear  # vary nuclear fixed costs
#        var_cost_nuclear = sa_var_cost_nuclear * var_cost_nuclear   # vary nuclear variable costs
    elif case_switch == 'ngccs_flex_nuc':
        system_components = ['natural_gas','wind','solar','nuclear','storage']
        flex_nuc_flag = np.array([1])   # nuclear is fully flexible / "rampable"
        fix_cost_natgas = 2 * fix_cost_natgas   # assume 2 x current cost for natural gas with CCS
        var_cost_natgas = 2 * var_cost_natgas   # assume 2 x current cost for natural gas with CCS
        fix_cost_nuclear  = sa_fix_cost_nuclear * fix_cost_nuclear  # vary nuclear fixed costs
#        var_cost_nuclear = sa_var_cost_nuclear * var_cost_nuclear   # vary nuclear variable costs
    elif case_switch == 'ngccs_fixed_nuc':
        system_components = ['natural_gas','wind','solar','nuclear','storage']
        flex_nuc_flag = np.array([0])   # nuclear is fixed at capacity
        fix_cost_natgas = 2 * fix_cost_natgas   # assume 2 x current cost for natural gas with CCS
        var_cost_natgas = 2 * var_cost_natgas   # assume 2 x current cost for natural gas with CCS
        fix_cost_nuclear  = sa_fix_cost_nuclear * fix_cost_nuclear  # vary nuclear fixed costs
#        var_cost_nuclear = sa_var_cost_nuclear * var_cost_nuclear   # vary nuclear variable costs
    elif case_switch == 'no_ng_flex_nuc':
        system_components = ['wind','solar','nuclear','storage']
        flex_nuc_flag = np.array([1])   # nuclear is fully flexible / "rampable"
        fix_cost_nuclear  = sa_fix_cost_nuclear * fix_cost_nuclear  # vary nuclear fixed costs
#        var_cost_nuclear = sa_var_cost_nuclear * var_cost_nuclear   # vary nuclear variable costs
    elif case_switch == 'no_ng_fixed_nuc':
        system_components = ['wind','solar','nuclear','storage']
        flex_nuc_flag = np.array([0])   # nuclear is fixed at capacity
        fix_cost_nuclear  = sa_fix_cost_nuclear * fix_cost_nuclear  # vary nuclear fixed costs
#        var_cost_nuclear = sa_var_cost_nuclear * var_cost_nuclear   # vary nuclear variable costs
        
    # -------------------------------------------------------------------------
    # parameters for other cases - keep for reference        

    elif case_switch == 'nuclear_storage':
        system_components = ['nuclear','storage']
        fix_cost_nuclear = np.array([0.050])
        var_cost_nuclear = np.array([0.001])
#        fix_cost_storage = np.array([.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001])  # $/kWh
#        fix_cost_storage = np.array([1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001,0.0000005,0.0000002,0.0000001,0.00000005,0.00000002,0.00000001])  # $/kWh
        fix_cost_storage = np.array([1e-8])  # $/kWh
        var_cost_storage = np.array([1e-8])  # no cost to storage, $/kWh/h
#        var_cost_dispatch_from_storage = np.array([10**-2,10**-3,10**-4,10**-6,10**-8]) # $/kWh dispatched roundtrip
        var_cost_dispatch_from_storage = np.array([0]) # $/kWh dispatched roundtrip
        storage_charging_efficiency = np.array([1.0])
        capacity_nuclear = np.array([-1,1.,1./(1.-0.1),1./(1.-0.2),1./(1.-0.25),1./(1.-1./3.),1./(1.-0.50)])
        demand_flag = np.array([-1,1])  # time series, constant
    elif case_switch == 'wind_storage':
        system_components = ['wind','storage']
        fix_cost_wind = np.array([0.013])
        var_cost_wind = np.array([1e-8])
#        fix_cost_storage = np.array([.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001])  # $/kWh
#        fix_cost_storage = np.array([1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001,0.0000005,0.0000002,0.0000001,0.00000005,0.00000002,0.00000001])  # $/kWh
        fix_cost_storage = np.array([1e-8])  # $/kWh
        var_cost_storage = np.array([1e-8])  # no cost to storage, $/kWh/h
#        var_cost_dispatch_from_storage = np.array([10**-2,10**-3,10**-4,10**-6,10**-8]) # $/kWh dispatched roundtrip
        var_cost_dispatch_from_storage = np.array([0]) # $/kWh dispatched roundtrip
        storage_charging_efficiency = np.array([1.0])
        capacity_wind = np.array([-1,1.,1./(1.-0.1),1./(1.-0.2),1./(1.-0.25),1./(1.-1./3.),1./(1.-0.50)])/0.222 # divide by time-series mean
        demand_flag = np.array([-1,1])  # time series, constant
    elif case_switch == 'solar_storage':
        system_components = ['solar','storage']
        fix_cost_solar = np.array([0.013])
        var_cost_solar = np.array([1e-8])
#        fix_cost_storage = np.array([.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001])  # $/kWh
#        fix_cost_storage = np.array([1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001,0.0000005,0.0000002,0.0000001,0.00000005,0.00000002,0.00000001])  # $/kWh
        fix_cost_storage = np.array([1e-8])  # $/kWh
        var_cost_storage = np.array([1e-8])  # no cost to storage, $/kWh/h
#        var_cost_dispatch_from_storage = np.array([10**-2,10**-3,10**-4,10**-6,10**-8]) # $/kWh dispatched roundtrip
        var_cost_dispatch_from_storage = np.array([0]) # $/kWh dispatched roundtrip
        storage_charging_efficiency = np.array([1.0])
        capacity_solar = np.array([-1,1.,1./(1.-0.1),1./(1.-0.2),1./(1.-0.25),1./(1.-1./3.),1./(1.-0.50)])/0.344999999999999 # divide by time-series mean
        demand_flag = np.array([-1,1])  # time series, constant
    elif case_switch == 'natgas_storage':
        system_components = ['natgas','storage']
        fix_cost_natgas = np.array([0.050])
        var_cost_natgas = np.array([0.001])
#        fix_cost_storage = np.array([.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001])  # $/kWh
#        fix_cost_storage = np.array([1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001,0.0000005,0.0000002,0.0000001,0.00000005,0.00000002,0.00000001])  # $/kWh
        fix_cost_storage = np.array([1e-8])  # $/kWh
        var_cost_storage = np.array([1e-8])  # no cost to storage, $/kWh/h
#        var_cost_dispatch_from_storage = np.array([10**-2,10**-3,10**-4,10**-6,10**-8]) # $/kWh dispatched roundtrip
        var_cost_dispatch_from_storage = np.array([0]) # $/kWh dispatched roundtrip
        storage_charging_efficiency = np.array([1.0])
        capacity_natgas = np.array([1./(1.-0.25),1./(1.-1./3.),1./(1.-0.50)])
        demand_flag = np.array([-1,1])  # time series, constant
    elif case_switch == 'nuc_bat':
        system_components = ['nuclear','storage']
        fix_cost_nuclear = np.array([0.050])
        var_cost_nuclear = np.array([0.001])
#        fix_cost_storage = np.array([.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001])  # $/kWh
#        fix_cost_storage = np.array([1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001,0.0000005,0.0000002,0.0000001,0.00000005,0.00000002,0.00000001])  # $/kWh
        fix_cost_storage = np.array([1e-8])  # $/kWh
        var_cost_storage = np.array([1e-8])  # no cost to storage, $/kWh/h
#        var_cost_dispatch_from_storage = np.array([10**-2,10**-3,10**-4,10**-6,10**-8]) # $/kWh dispatched roundtrip
        var_cost_dispatch_from_storage = np.array([0]) # $/kWh dispatched roundtrip
        storage_charging_efficiency = np.array([1.0])
    elif case_switch == 'nuc_bat_unmet':
        system_components = ['nuclear','storage','unmet_demand']
        fix_cost_nuclear = np.array([0.050])
        var_cost_nuclear = np.array([0.001])
#        fix_cost_storage = np.array([.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001])  # $/kWh
#        fix_cost_storage = np.array([1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001,0.0000005,0.0000002,0.0000001,0.00000005,0.00000002,0.00000001])  # $/kWh
        fix_cost_storage = np.array([1e-4])  # $/kWh
        var_cost_storage = np.array([1e-8])  # no cost to storage, $/kWh/h
#        var_cost_dispatch_from_storage = np.array([10**-2,10**-3,10**-4,10**-6,10**-8]) # $/kWh dispatched roundtrip
        var_cost_dispatch_from_storage = np.array([0]) # $/kWh dispatched roundtrip
        storage_charging_efficiency = np.array([1.0])
        var_cost_unmet_demand = np.array([0.1])
    elif case_switch == 'solar_wind_bat':
        system_components = ['solar','wind','storage']
        fix_cost_solar = np.array([0.013])
        var_cost_solar = np.array([0.001])
        fix_cost_wind = np.array([0.013])
        var_cost_wind = np.array([0.002])
#        fix_cost_storage = np.array([.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001])  # $/kWh
#        fix_cost_storage = np.array([1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001,0.0000005,0.0000002,0.0000001,0.00000005,0.00000002,0.00000001])  # $/kWh
        fix_cost_storage = np.array([0.1,1e-8])
        var_cost_storage = np.array([1e-8])  # no cost to storage, $/kWh/h
#        var_cost_dispatch_from_storage = np.array([10**-2,10**-3,10**-4,10**-6,10**-8]) # $/kWh dispatched roundtrip
        var_cost_dispatch_from_storage = np.array([0]) # $/kWh dispatched roundtrip
        storage_charging_efficiency = np.array([1.0])
    elif case_switch == 'nuclear_solar_wind_bat':
        system_components = ['nuclear','solar','wind','storage']
        fix_cost_nuclear = np.array([0.050])
        var_cost_nuclear = np.array([0.001])
        fix_cost_solar = np.array([0.013])
        var_cost_solar = np.array([0.001])
        fix_cost_wind = np.array([0.013])
        var_cost_wind = np.array([0.002])
#        fix_cost_storage = np.array([.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001])  # $/kWh
#        fix_cost_storage = np.array([1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001,0.0000005,0.0000002,0.0000001,0.00000005,0.00000002,0.00000001])  # $/kWh
        fix_cost_storage = np.array([0.1,1e-8])
        var_cost_storage = np.array([1e-8])  # no cost to storage, $/kWh/h
#        var_cost_dispatch_from_storage = np.array([10**-2,10**-3,10**-4,10**-6,10**-8]) # $/kWh dispatched roundtrip
        var_cost_dispatch_from_storage = np.array([0]) # $/kWh dispatched roundtrip
        storage_charging_efficiency = np.array([1.0])
    elif case_switch == 'gas_bat':
        system_components = ['natural_gas','storage']
        fix_cost_natgas  = np.array([0.010]) # $/kW per hour
        var_cost_natgas  = np.array([0.030])  # $/kWh
        fix_cost_storage = np.array([0.001])  # $/kWh
        var_cost_storage = np.array([10**-8])  # no cost to storage, $/kWh/h
        var_cost_dispatch_to_storage = np.array([0])  # no cost to storage, $/kWh
        var_cost_dispatch_from_storage = np.array([0])  # no cost to storage, $/kWh
        storage_charging_efficiency = np.array([1])
    elif case_switch == 'solar_bat':
        system_components = ['solar','storage']
        fix_cost_solar = np.array([0.013])  # $/kWh
        var_cost_solar = np.array([10**-8,0])
        fix_cost_storage = np.array([0.001])  # $/kWh
        var_cost_storage = np.array([10**-8])  # no cost to storage, $/kWh/h
        var_cost_dispatch_to_storage = np.array([0])  # no cost to storage, $/kWh
        var_cost_dispatch_from_storage = np.array([0])  # no cost to storage, $/kWh
        storage_charging_efficiency = np.array([1])
    elif case_switch == 'wind_bat':
        system_components = ['wind','storage']
        fix_cost_wind = np.array([0.013])  # $/kWh
        var_cost_wind = np.array([10**-8,0])
        fix_cost_storage = np.array([0.001])  # $/kWh
        var_cost_storage = np.array([10**-8])  # no cost to storage, $/kWh/h
        var_cost_dispatch_to_storage = np.array([0])  # no cost to storage, $/kWh
        var_cost_dispatch_from_storage = np.array([0])  # no cost to storage, $/kWh
        storage_charging_efficiency = np.array([1])
    elif case_switch == 'solar_wind_bat_unmet':
        system_components = ['solar','wind','storage','unmet_demand']
        fix_cost_storage = np.array([0.000001,0.0001,1])  # $/kWh
    elif case_switch == 'nate':
        fix_cost_storage = np.array([0.003,0.001])  # $/kWh
        fix_cost_wind = np.array([0.013,0.013/2.])  # $/kWh
        fix_cost_solar = np.array([0.013,0.013/2.])  # $/kWh
        fix_cost_natgas  = np.array([big_num])
        fix_cost_nuclear = np.array([big_num])
    elif case_switch == 'nate2_battery':
        fix_cost_storage = np.array([0.0005,0.001,0.002,0.003])  # $/kWh
        fix_cost_wind = np.array([0.013])  # $/kWh
        fix_cost_solar = np.array([0.013])  # $/kWh
        fix_cost_natgas  = np.array([big_num])
        fix_cost_nuclear = np.array([big_num])
    elif case_switch == 'nate2':
        system_components = ['solar','wind']
        fix_cost_storage = np.array([0.001])  # $/kWh
        fix_cost_wind = np.array([0.013])  # $/kWh
        fix_cost_solar = np.array([0.013,0.013*0.5])  # $/kWh
    elif case_switch == 'nate_spectrum':
        system_components = ['solar','wind']
        fix_cost_storage = np.array([0.001])  # $/kWh
        fix_cost_wind = np.array([0.013])  # $/kWh
        fix_cost_solar = np.array(0.013 * np.array([0.1,0.2,0.5,1,2,5,10]))  # $/kWh
    elif case_switch == 'battery_natgas':
        fix_cost_wind = np.array([big_num])  # $/kWh
        fix_cost_solar = np.array([big_num])  # $/kWh
        fix_cost_natgas  = np.array([0.010]) # $/kW per hour
        fix_cost_storage = np.array([0.001])  # $/kWh
        fix_cost_nuclear = np.array([big_num])
        var_cost_unmet_demand = np.array([big_num])
        var_cost_storage = np.array([10**-8])  # no cost to storage, $/kWh/h
        var_cost_dispatch_to_storage = np.array([0])  # no cost to storage, $/kWh
        var_cost_dispatch_from_storage = np.array([0])  # no cost to storage, $/kWh
        var_cost_natgas  = np.array([0.030])  # $/kWh
    elif case_switch == 'battery_nuclear':
        fix_cost_storage = np.array([0.001])  # $/kWh
        fix_cost_wind = np.array([big_num])  # $/kWh
        fix_cost_solar = np.array([big_num])  # $/kWh
        fix_cost_natgas = np.array([big_num])  # $/kWh
        fix_cost_storage = np.array([0.001])  # $/kWh
        fix_cost_nuclear = np.array([0.050])
        var_cost_unmet_demand = np.array([big_num])
        var_cost_storage = np.array([10**-8])  # cost of storage, $/kWh/h
        var_cost_dispatch_to_storage = np.array([0])  # no cost to storage, $/kWh
        var_cost_dispatch_from_storage = np.array([0])  # no cost to storage, $/kWh
    elif case_switch == 'nate2_store':
        fix_cost_storage = np.array([0.001])  # $/kWh
        fix_cost_wind = np.array([0.013])  # $/kWh
        fix_cost_solar = np.array([0.013,0.013/2.])  # $/kWh
        fix_cost_natgas  = np.array([big_num])
        fix_cost_nuclear = np.array([big_num])
        var_cost_storage = np.array([10**-8])  # cost of storage, $/kWh/h
    elif case_switch == 'nate2_store_big_num':
        fix_cost_storage = np.array([0.001])  # $/kWh
        fix_cost_wind = np.array([0.013])  # $/kWh
        fix_cost_solar = np.array([0.013,0.013/2.])  # $/kWh
        fix_cost_natgas  = np.array([0.9* big_num])
        fix_cost_nuclear = np.array([0.9* big_num])
        var_cost_storage = np.array([10**-8])  # cost of storage, $/kWh/h
    elif case_switch == 'nate2':
        system_components = ['solar','wind','storage']
        fix_cost_storage = np.array([0.001])  # $/kWh
        fix_cost_wind = np.array([0.013])  # $/kWh
        fix_cost_solar = np.array([0.013,0.013/2.])  # $/kWh
    elif case_switch == 'idealized':
        system_components = ['solar','wind','storage']
        fix_cost_storage = np.array([0.001])  # $/kW/h
        var_cost_storage = np.array([1e-8])
        storage_charging_efficiency = np.array([1.])
        fix_cost_wind = np.array([0.013])  # $/kW/h
        var_cost_wind = np.array([0,0.000001,0.00001,0.0001,0.001,0.01,0.1])
        fix_cost_solar = np.array([0.013/2.])  # $/kW/h
        var_cost_solar = np.array([0.0]) # $kWh
    elif case_switch == 'idealized2':
        system_components = ['solar','nuclear','storage']
        fix_cost_storage = np.array([0.001])  # $/kW/h
        var_cost_storage = np.array([0,1e-10,1e-9,1e-8,1e-7,1e-6])
        storage_charging_efficiency = np.array([1.])
        fix_cost_nuclear = np.array([0.05])  # $/kW/h
        var_cost_solar = np.array([0.0])
        fix_cost_solar = np.array([0.013/2.])  # $/kW/h
    else:
        print 'Running default case'
    
    # -------------------------------------------------------------------------
    # read in demand and capacity factor data
    
    demand_raw = np.load(input_folder + 'conus_real_demand.npy')[start_time:end_time] * 1e3     # time series of demand in kW
    demand = demand_raw / np.average(demand_raw)    # normalize to average demand
    
    wind_capacity_factor_reanalysis = np.load(input_folder + 'United States of America_CFwind_area-weighted-mean.npy')[start_time:end_time]
    solar_capacity_factor_reanalysis = np.load(input_folder + 'United States of America_CFSolar_area-weighted-mean.npy')[start_time:end_time]
    
    wind_capacity_factor = (0.345 / np.mean(wind_capacity_factor_reanalysis)) * wind_capacity_factor_reanalysis     # time series of wind capacity factor data
    solar_capacity_factor = (0.222 / np.mean(solar_capacity_factor_reanalysis)) * solar_capacity_factor_reanalysis  # time series of solar capacity factor data
    
    time_series = {
            'demand_series':demand,
            'wind_series':wind_capacity_factor,
            'solar_series':solar_capacity_factor      
            }
    
    print start_time, end_time, end_time-start_time
    
    # -------------------------------------------------------------------------
    # generate assumption list (array) to step through
        
    list_of_assumptions = [
            fix_cost_natgas,
            fix_cost_nuclear,
            fix_cost_wind,
            fix_cost_solar,
            fix_cost_storage,
            var_cost_natgas,
            var_cost_nuclear,
            var_cost_wind,
            var_cost_solar,
            var_cost_storage,
            var_cost_dispatch_to_storage,
            var_cost_dispatch_from_storage,
            var_cost_unmet_demand,
            storage_charging_efficiency,
            capacity_natgas,    # set to numbers if goal is to set capacity rather than optimize for it
            capacity_wind,      # set to numbers if goal is to set capacity rather than optimize for it
            capacity_solar,     # set to numbers if goal is to set capacity rather than optimize for it
            capacity_nuclear,   # set to numbers if goal is to set capacity rather than optimize for it
            capacity_storage,   # set to numbers if goal is to set capacity rather than optimize for it
            demand_flag,        # whether to use time series or constant
            flex_nuc_flag       # whether nuclear is fully rampable or fixed at capacity, flag for switching constraints for nuclear in Core_Model
            ]
    
    # Cartesian product of assumptions
    product_result = itertools.product(*list_of_assumptions)
    num_cases = functools.reduce(operator.mul, map(len,list_of_assumptions),1)
    product_list = [item for item in product_result]
    # note: this coding approach from:
        # https://stackoverflow.com/questions/533905/get-the-cartesian-product-of-a-series-of-lists
    
    assumption_list = [dict() for x in range(num_cases)]
 
    for idx in range(num_cases):
        assumption_list[idx] = {
            'fix_cost_natgas':product_list[idx][0],
            'fix_cost_nuclear':product_list[idx][1],
            'fix_cost_wind':product_list[idx][2],
            'fix_cost_solar':product_list[idx][3],
            'fix_cost_storage':product_list[idx][4],
            'var_cost_natgas':product_list[idx][5],
            'var_cost_nuclear':product_list[idx][6],
            'var_cost_wind':product_list[idx][7],
            'var_cost_solar':product_list[idx][8],
            'var_cost_storage':product_list[idx][9],
            'var_cost_dispatch_to_storage':product_list[idx][10],
            'var_cost_dispatch_from_storage':product_list[idx][11],
            'var_cost_unmet_demand':product_list[idx][12],
            'storage_charging_efficiency':product_list[idx][13],
            'system_components':system_components,
            'capacity_natgas':product_list[idx][14],    # set to numbers if goal is to set capacity rather than optimize for it
            'capacity_wind':product_list[idx][15],      # set to numbers if goal is to set capacity rather than optimize for it
            'capacity_solar':product_list[idx][16],     # set to numbers if goal is to set capacity rather than optimize for it
            'capacity_nuclear':product_list[idx][17],   # set to numbers if goal is to set capacity rather than optimize for it
            'capacity_storage':product_list[idx][18],   # set to numbers if goal is to set capacity rather than optimize for it
            'demand_flag':product_list[idx][19],        # whether to use time series or constant
            'flex_nuc_flag':product_list[idx][20]       # whether nuclear is fully rampable or fixed at capacity, flag for switching constraints for nuclear in Core_Model
            }
                                                                        
    # -------------------------------------------------------------------------
    
    file_info = { 
            'input_folder':input_folder, 
            'output_folder':output_folder, 
            'output_file_name':base_case_switch + '_' + case_switch,
            'base_case_switch':base_case_switch,
            'case_switch':case_switch
            }
    if verbose:
        print 'case ', base_case_switch + '_' + case_switch + ' prepared'
    
    return file_info, time_series, assumption_list
            