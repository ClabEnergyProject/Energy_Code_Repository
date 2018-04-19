"""
file name: Save_Basic_Results.py
    pickle results and write results to .csv files
    
authors:
    Fan (original model)
    Ken (reorganized model)
    Mengyao (nuclear vs. renewables)

version: _my180406
    modified from _my180403 for preliminary analysis of nuclear vs. renewables 

updates:
    added flex_nuc_flag and curtailment (wind, solar, and nuclear) to output files
    switched headers for wind and solar capacity factors
    moved column "var cost storage" after "var cost from dispatch" to be consistent with order of storage results - for ease of cost calculation
    
to-dos:
    clarify meaning of "wind capacity (kW)" and "solar capacity (kW)" - see notes

"""

#%% import modules

import os
import numpy as np
import csv
import shelve
import contextlib
import pickle

#%% functions: save results

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modify z with y's keys and values and return None
    return z

def save_basic_results(
        file_info,
        time_series,
        assumption_list,
        result_list,
        verbose
        ):
    
    # put raw results in file for later analysis
    pickle_raw_results(
        file_info,
        time_series,
        assumption_list,
        result_list,
        verbose
        )
    
    # save time series results in .csv files
    save_vector_results_as_csv(
        file_info,
        time_series,
        result_list,
        verbose
        )
    
    # save key assumptions and results in summary .csv file for each case
    scalar_names, scalar_table = postprocess_key_scalar_results(
        file_info,
        time_series,
        assumption_list,
        result_list,
        verbose
        )
    
    return scalar_names, scalar_table

# -----------------------------------------------------------------------------
# put raw results in file for later analysis

def pickle_raw_results(
        file_info,
        time_series,
        assumption_list,
        result_list,
        verbose
        ):
    output_folder = file_info['output_folder']
    output_file_name = file_info['base_case_switch'] + '_' + file_info['case_switch'] + '.pickle'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    with open(output_folder + '/' + output_file_name, 'wb') as db:
        pickle.dump([file_info,time_series,assumption_list,result_list], db, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print 'data pickled to '+output_folder + '/' + output_file_name

# -----------------------------------------------------------------------------
# save time series results in .csv files

def save_vector_results_as_csv(
        file_info,
        time_series,
        result_list,
        verbose
        ):
    
    output_folder = file_info['output_folder']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    num_time_periods = result_list[0]['dispatch_natgas'].size
    outarray = np.zeros((1+num_time_periods,8))

    # output file for each set of assumptions
    for idx in range(len(result_list)):
        output_file_name = file_info['output_file_name'] + '_series_' +  str(idx).zfill(3)
        
        result = result_list[idx]
        
        header_list = []
        series_list = []
        
        header_list += ['demand (kW)']
        series_list.append( time_series['demand_series'] )
                
        header_list += ['solar (kW)']
        series_list.append( time_series['solar_series'] )

        header_list += ['wind (kW)']
        series_list.append( time_series['wind_series'] )
        
        header_list += ['dispatch natgas (kW)']
        series_list.append( result['dispatch_natgas'].flatten() )
        
        header_list += ['dispatch solar (kW)']
        series_list.append( result['dispatch_solar'].flatten() )
        
        header_list += ['dispatch wind (kW)']
        series_list.append( result['dispatch_wind'].flatten() )
        
        header_list += ['dispatch nuclear (kW)']
        series_list.append( result['dispatch_nuclear'].flatten() )
        
        header_list += ['dispatch to storage (kW)']
        series_list.append( result['dispatch_to_storage'].flatten() )
        
        header_list += ['dispatch from storage (kW)']
        series_list.append( result['dispatch_from_storage'].flatten() )

        header_list += ['energy storage (kWh)']
        series_list.append( result['energy_storage'].flatten() )
        
        header_list += ['curtailment solar (kW)']
        series_list.append( result['curtailment_solar'].flatten() )

        header_list += ['curtailment wind (kW)']
        series_list.append( result['curtailment_wind'].flatten() )

        header_list += ['curtailment nuclear (kW)']
        series_list.append( result['curtailment_nuclear'].flatten() )
        
        header_list += ['dispatch unmet demand (kW)']
        series_list.append( result['dispatch_unmet_demand'].flatten() )

        out_array = np.array(series_list)
        
        with contextlib.closing(open(output_folder + '/' + output_file_name + '.csv', 'wb')) as output_file:
            writer = csv.writer(output_file)
            writer.writerow(header_list)
            writer.writerows((np.asarray(series_list)).transpose())
            output_file.close()
        
        if verbose: 
            print 'file written: ' + output_file_name + '.csv'

# -----------------------------------------------------------------------------
# save key assumptions and results in summary .csv file for each case

def postprocess_key_scalar_results(
        file_info,
        time_series,
        assumption_list,
        result_list,
        verbose
        ):
    
    combined_dic = map(merge_two_dicts, assumption_list, result_list)
    
    scalar_names = [
            'fix cost natgas ($/kW/h)',
            'fix cost solar ($/kW/h)',
            'fix cost wind ($/kW/h)',
            'fix cost nuclear ($/kW/h)',
            'fix cost storage ($/kW/h)',
            
            'var cost natgas ($/kWh)',
            'var cost solar ($/kWh)',
            'var cost wind ($/kWh)',
            'var cost nuclear ($/kWh)',
            'var cost dispatch to storage ($/kWh)',
            'var cost dispatch from storage ($/kWh)',
            'var cost storage ($/kWh/h)',
            'var cost unmet demand ($/kWh)',
            
            'storage charging efficiency',
            
            'flex nuc flag',
            
            'demand flag',
            'demand (kW)',
            # note to self: clarify: total available capacity normalized to mean? - rename to "max wind/solar capacity" or just "wind/solar" to be consistent with time series .csv?
            'solar capacity (kW)',
            'wind capacity (kW)',   
            
            'capacity natgas (kW)',
            'capacity solar (kW)',
            'capacity wind (kW)',
            'capacity nuclear (kW)',
            'capacity storage (kW)',
            'system cost ($/kW/h)', # assuming demand normalized to 1 kW
            'problem status',
            
            'dispatch natgas (kW)',
            'dispatch solar (kW)',
            'dispatch wind (kW)',
            'dispatch nuclear (kW)',
            'dispatch to storage (kW)',
            'dispatch from storage (kW)',
            'energy storage (kWh)',
            'curtailment solar (kW)',
            'curtailment wind (kW)',
            'curtailment nuclear (kW)',
            'dispatch unmet demand (kW)'
            ]

    num_time_periods = combined_dic[0]['dispatch_natgas'].size
    
    scalar_table = [
            [
                    # assumptions - fixed costs
                    d['fix_cost_natgas'],
                    d['fix_cost_solar'],
                    d['fix_cost_wind'],
                    d['fix_cost_nuclear'],
                    d['fix_cost_storage'],
                    
                    # assumptions - variable costs
                    d['var_cost_natgas'],
                    d['var_cost_solar'],
                    d['var_cost_wind'],
                    d['var_cost_nuclear'],
                    d['var_cost_dispatch_to_storage'],
                    d['var_cost_dispatch_from_storage'],
                    d['var_cost_storage'],
                    d['var_cost_unmet_demand'],
                    
                    # assumption - charging efficiency
                    d['storage_charging_efficiency'],

                    # assumption - nuclear rampability
                    d['flex_nuc_flag'],

                    # mean of time series data
                    d['demand_flag'],
                    np.sum(time_series['demand_series'])/num_time_periods,
                    np.sum(time_series['solar_series'])/num_time_periods,
                    np.sum(time_series['wind_series'])/num_time_periods,
                    
                    # scalar results
                    d['capacity_natgas'],
                    d['capacity_solar'],
                    d['capacity_wind'],
                    d['capacity_nuclear'],
                    d['capacity_storage'],
                    d['system_cost'],
                    d['problem_status'],
                    
                    # mean of time series results
                    np.sum(d['dispatch_natgas'])/num_time_periods,
                    np.sum(d['dispatch_solar'])/num_time_periods,
                    np.sum(d['dispatch_wind'])/num_time_periods,
                    np.sum(d['dispatch_nuclear'])/num_time_periods,
                    np.sum(d['dispatch_to_storage'])/num_time_periods,
                    np.sum(d['dispatch_from_storage'])/num_time_periods,
                    np.sum(d['energy_storage'])/num_time_periods,
                    np.sum(d['curtailment_solar'])/num_time_periods,
                    np.sum(d['curtailment_wind'])/num_time_periods,
                    np.sum(d['curtailment_nuclear'])/num_time_periods,
                    np.sum(d['dispatch_unmet_demand'])/num_time_periods
             ]
            for d in combined_dic
            ]
            
    output_folder = file_info['output_folder']
    output_file_name = file_info['output_file_name']
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    with contextlib.closing(open(output_folder + '/' + output_file_name + '.csv', 'wb')) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(scalar_names)
        writer.writerows(scalar_table)
        output_file.close()
        
    if verbose: 
        print 'file written: ' + file_info['output_file_name'] + '.csv'
    
    return scalar_names, scalar_table
    