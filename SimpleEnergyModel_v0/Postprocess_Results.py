 #!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""

File name: Core_Model.py

Idealized energy system models

Spatial scope: U.S.
Data: Matt Shaner's paper with reanalysis data and U.S. demand.

Technology:
    Generation: natural gas, wind, solar, nuclear
    Energy storage: one generic (a pre-determined round-trip efficiency)
    Curtailment: Yes (free)
    Unmet demand: No
    
Optimization:
    Linear programming (LP)
    Energy balance constraints for the grid and the energy storage facility.

@author: Fan
Time
    Dec 1, 4-8, 11, 19, 22
    Jan 2-4, 24-27
    
"""

# -----------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

#===============================================================================
#================================================= DEFINITION SECTION ==========
#===============================================================================

#------------------------------------------------------------------------------
#------------------------------------------File output and import -------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def unpickle_raw_results(
        file_path_name,
        verbose
        ):
    
    with open(file_path_name, 'rb') as db:
       file_info, time_series, assumption_list, result_list = pickle.load (db)
    
    if verbose:
        print 'data unpickled from '+file_path_name
    
    return file_info, time_series, assumption_list, result_list

# Core function
#   Linear programming
#   Output postprocessing


 
#def postprocess_key_scalar_results(
#        file_info,
#        time_series,
#        assumption_list,
#        result_list,
#        verbose
#        ):
#    
#    combined_dic = map(merge_two_dicts,assumption_list,result_list)
#    
#    scalar_names = [
#            'fix_cost_natgas ($/kW/h)',
#            'fix_cost_solar ($/kW/h)',
#            'fix_cost_wind ($/kW/h)',
#            'fix_cost_nuclear ($/kW/h)',
#            'fix_cost_storage ($/kW/h)',
#            
#            'var_cost_natgas ($/kWh)',
#            'var_cost_solar ($/kWh)',
#            'var_cost_wind ($/kWh)',
#            'var_cost_nuclear ($/kWh)',
#            'var_cost_storage ($/kWh)',
#            'var_cost_unmet_demand ($/kWh)',
#            
#            'storage_charging_efficiency',
#            
#            'demand (kW)',
#            'wind capacity (kW)',
#            'solar capacity (kW)',
#            
#            'capacity_natgas (kW)',
#            'capacity_solar (kW)',
#            'capacity_wind (kW)',
#            'capacity_nuclear (kW)',
#            'capacity_storage (kW)',
#            'system_cost ($/kW/h)', # asnp.suming demand normalized to 1 kW
#            
#            'dispatch_natgas (kW)',
#            'dispatch_solar (kW)',
#            'dispatch_wind (kW)',
#            'dispatch_nuclear (kW)',
#            'dispatch_to_storage (kW)',
#            'dispatch_from_storage (kW)',
#            'dispatch_curtailment (kW)',
#            'dispatch_unmet_demand (kW)',
#            
#            'energy_storage (kWh)'
#            
#            ]
#
#    num_time_periods = combined_dic[0]['dispatch_natgas'].value.size
#    
#    scalar_table = [
#            [
#                    # assumptions
#                    
#                    d['fix_cost_natgas'],
#                    d['fix_cost_solar'],
#                    d['fix_cost_wind'],
#                    d['fix_cost_nuclear'],
#                    d['fix_cost_storage'],
#                    
#                    d['var_cost_natgas'],
#                    d['var_cost_solar'],
#                    d['var_cost_wind'],
#                    d['var_cost_nuclear'],
#                    d['var_cost_storage'],
#                    d['var_cost_unmet_demand'],
#                    
#                    d['storage_charging_efficiency'],
#                    
#                    # mean of time series assumptions
#                    
#                    np.asscalar(np.sum(time_series['demand_series']))/num_time_periods,
#                    np.asscalar(np.sum(time_series['solar_series']))/num_time_periods,
#                    np.asscalar(np.sum(time_series['wind_series']))/num_time_periods,
#                    
#                    # scalar results
#                    
#                    d['capacity_natgas'].value,
#                    d['capacity_solar'].value,
#                    d['capacity_wind'].value,
#                    d['capacity_nuclear'].value,
#                    d['capacity_storage'].value,
#                    d['system_cost'],
#                    
#                    # mean of time series results                
#                                
#                    np.asscalar(np.sum(d['dispatch_natgas'].value))/num_time_periods,
#                    np.asscalar(np.sum(d['dispatch_solar'].value))/num_time_periods,
#                    np.asscalar(np.sum(d['dispatch_wind'].value))/num_time_periods,
#                    np.asscalar(np.sum(d['dispatch_nuclear'].value))/num_time_periods,
#                    np.asscalar(np.sum(d['dispatch_to_storage'].value))/num_time_periods,
#                    np.asscalar(np.sum(d['dispatch_from_storage'].value))/num_time_periods,
#                    np.asscalar(np.sum(d['dispatch_curtailment'].value))/num_time_periods,
#                    np.asscalar(np.sum(d['dispatch_unmet_demand'].value))/num_time_periods,
#                    
#                    np.asscalar(np.sum(d['energy_storage'].value))/num_time_periods
#                    
#             ]
#            for d in combined_dic
#            ]
            
#    result = {
#            'capacity_natgas':capacity_natgas,
#            'capacity_solar':capacity_solar,
#            'capacity_wind':capacity_wind,
#            'capacity_nuclear':capacity_nuclear,
#            'capacity_storage':capacity_storage,
#            
#            'dispatch_natgas':dispatch_natgas,
#            'dispatch_solar':dispatch_solar,
#            'dispatch_wind':dispatch_wind,
#            'dispatch_nuclear':dispatch_nuclear,
#            'dispatch_to_storage':dispatch_to_storage,
#            'dispatch_from_storage':dispatch_from_storage,
#            'dispatch_curtailment':dispatch_curtailment,
#            'dispatch_unmet_demand':dispatch_unmet_demand,
#            
#            'energy_storage':energy_storage,
#            
#            'system_cost':prob.value
#            
#            }
    
#------------------------------------------------------------------------------
#---------------- Convert list of dictionaries to dictionary of lists ---------
#------------------------------------------------------------------------------
  
#------------------------------------------------------------------------------
def postprocess_key_scalar_results(
        file_info,
        time_series,
        assumption_list,
        result_list,
        verbose
        ):
    
    combined_dic = map(merge_two_dicts,assumption_list,result_list)
    
    scalar_names = [
            'fix_cost_natgas ($/kW/h)',
            'fix_cost_solar ($/kW/h)',
            'fix_cost_wind ($/kW/h)',
            'fix_cost_nuclear ($/kW/h)',
            'fix_cost_storage ($/kW/h)',
            
            'var_cost_natgas ($/kWh)',
            'var_cost_solar ($/kWh)',
            'var_cost_wind ($/kWh)',
            'var_cost_nuclear ($/kWh)',
            'var_cost_storage ($/kWh)',
            'var_cost_unmet_demand ($/kWh)',
            
            'storage_charging_efficiency',
            
            'demand (kW)',
            'wind capacity (kW)',
            'solar capacity (kW)',
            
            'capacity_natgas (kW)',
            'capacity_solar (kW)',
            'capacity_wind (kW)',
            'capacity_nuclear (kW)',
            'capacity_storage (kW)',
            'system_cost ($/kW/h)', # assuming demand normalized to 1 kW
            
            'dispatch_natgas (kW)',
            'dispatch_solar (kW)',
            'dispatch_wind (kW)',
            'dispatch_nuclear (kW)',
            'dispatch_to_storage (kW)',
            'dispatch_from_storage (kW)',
            'dispatch_curtailment (kW)',
            'dispatch_unmet_demand (kW)',
            
            'energy_storage (kWh)'
            
            ]

    num_time_periods = combined_dic[0]['dispatch_natgas'].value.size
    
    scalar_table = [
            [
                    # assumptions
                    
                    d['fix_cost_natgas'],
                    d['fix_cost_solar'],
                    d['fix_cost_wind'],
                    d['fix_cost_nuclear'],
                    d['fix_cost_storage'],
                    
                    d['var_cost_natgas'],
                    d['var_cost_solar'],
                    d['var_cost_wind'],
                    d['var_cost_nuclear'],
                    d['var_cost_storage'],
                    d['var_cost_unmet_demand'],
                    
                    d['storage_charging_efficiency'],
                    
                    # mean of time series assumptions
                    
                    np.asscalar(sum(time_series['demand_series']))/num_time_periods,
                    np.asscalar(sum(time_series['solar_series']))/num_time_periods,
                    np.asscalar(sum(time_series['wind_series']))/num_time_periods,
                    
                    # scalar results
                    
                    d['capacity_natgas'].value,
                    d['capacity_solar'].value,
                    d['capacity_wind'].value,
                    d['capacity_nuclear'].value,
                    d['capacity_storage'].value,
                    d['system_cost'],
                    
                    # mean of time series results                
                                
                    np.asscalar(sum(d['dispatch_natgas'].value))/num_time_periods,
                    np.asscalar(sum(d['dispatch_solar'].value))/num_time_periods,
                    np.asscalar(sum(d['dispatch_wind'].value))/num_time_periods,
                    np.asscalar(sum(d['dispatch_nuclear'].value))/num_time_periods,
                    np.asscalar(sum(d['dispatch_to_storage'].value))/num_time_periods,
                    np.asscalar(sum(d['dispatch_from_storage'].value))/num_time_periods,
                    np.asscalar(sum(d['dispatch_curtailment'].value))/num_time_periods,
                    np.asscalar(sum(d['dispatch_unmet_demand'].value))/num_time_periods,
                    
                    np.asscalar(sum(d['energy_storage'].value))/num_time_periods
                    
             ]
            for d in combined_dic
            ]
            
    output_folder = file_info['output_folder']
    output_file_name = file_info['output_file_name']
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    with contextlib.closing(open(output_folder + "/" + output_file_name, 'wb')) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(scalar_names)
        writer.writerows(scalar_table)
        output_file.close()
        
    if verbose: 
        print 'file written: ' + file_info['output_file_name']
    
    return scalar_names,scalar_table
    
    return scalar_names,scalar_table

#------------------------------------------------------------------------------
def prepare_scalar_variables (
        file_info,
        time_series,
        assumption_list,
        result_list,
        verbose
        ):
    
    num_time_periods = result_list[0]['dispatch_natgas'].value.size
    res = {}

# the 'for dic in assumption_list' is just to make things the right size.
    res['demand'] = np.array([np.sum(time_series['demand_series'])/num_time_periods for dic in assumption_list])
    res['solar_capacity'] = np.array([np.sum(time_series['solar_series'])/num_time_periods for dic in assumption_list])
    res['wind_capacity'] = np.array([np.sum(time_series['solar_series'])/num_time_periods for dic in assumption_list])

# 5 fixed costs $/kW
    res['fix_cost_natgas'] = np.array([dic['fix_cost_natgas'] for dic in assumption_list])
    res['fix_cost_solar'] = np.array([dic['fix_cost_solar'] for dic in assumption_list])
    res['fix_cost_wind'] = np.array([dic['fix_cost_wind'] for dic in assumption_list])
    res['fix_cost_nuclear'] = np.array([dic['fix_cost_nuclear'] for dic in assumption_list])
    res['fix_cost_storage'] = np.array([dic['fix_cost_storage'] for dic in assumption_list])

# 6 variable costs $/kWh
    res['var_cost_natgas'] = np.array([dic['var_cost_natgas'] for dic in assumption_list])
    res['var_cost_solar'] = np.array([dic['var_cost_solar'] for dic in assumption_list])
    res['var_cost_wind'] = np.array([dic['var_cost_wind'] for dic in assumption_list])
    res['var_cost_nuclear'] = np.array([dic['var_cost_nuclear'] for dic in assumption_list])
    res['var_cost_storage'] = np.array([dic['var_cost_storage'] for dic in assumption_list])
    res['var_cost_unmet_demand'] = np.array([dic['var_cost_unmet_demand'] for dic in assumption_list])

#5 capacity kW
    res['capacity_natgas'] = np.array([dic['capacity_natgas'].value for dic in result_list])
    res['capacity_solar'] = np.array([dic['capacity_solar'].value for dic in result_list])
    res['capacity_wind'] = np.array([dic['capacity_wind'].value for dic in result_list])
    res['capacity_nuclear'] = np.array([dic['capacity_nuclear'].value for dic in result_list])
    res['capacity_storage'] = np.array([dic['capacity_storage'].value for dic in result_list])

# 7 dispatch amounts kWh/h    
    res['dispatch_natgas'] = np.array([np.sum(dic['dispatch_natgas'].value, axis=0)/num_time_periods for dic in result_list]).flatten()
    res['dispatch_solar'] = np.array([np.sum(dic['dispatch_solar'].value, axis=0)/num_time_periods for dic in result_list]).flatten()
    res['dispatch_wind'] = np.array([np.sum(dic['dispatch_wind'].value, axis=0)/num_time_periods for dic in result_list]).flatten()
    res['dispatch_nuclear'] = np.array([np.sum(dic['dispatch_nuclear'].value, axis=0)/num_time_periods for dic in result_list]).flatten()
    res['dispatch_to_storage'] = np.array([np.sum(dic['dispatch_to_storage'].value, axis=0)/num_time_periods for dic in result_list]).flatten()
    res['dispatch_from_storage'] = np.array([np.sum(dic['dispatch_from_storage'].value, axis=0)/num_time_periods for dic in result_list]).flatten()
    res['dispatch_unmet_demand'] = np.array([np.sum(dic['dispatch_unmet_demand'].value, axis=0)/num_time_periods for dic in result_list]).flatten()
    res['dispatch_curtailment'] = np.array([np.sum(dic['dispatch_curtailment'].value, axis=0)/num_time_periods for dic in result_list]).flatten()
    
    res['energy_storage'] = np.array([np.sum(dic['energy_storage'], axis=0)/num_time_periods for dic in result_list]).flatten()
    res['system_cost'] = np.array([dic['system_cost'] for dic in result_list])
    
    return res

#------------------------------------------------------------------------------
#------------------------------------------------ Plotting function -----------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# log x axis
def plot_line_logx(x,y):
    plt.subplots()
    plt.semilogx(x,y)
    #plt.title('semilogx')
    plt.grid(True)
    plt.show()
    
#------------------------------------------------------------------------------
def plot_line(x,y):
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.show()
    
def plot_line(x,y,filename):
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.savefig(filename)
    plt.show()

def plot_line(x,y,xminmax, yminmax, xlabel0, ylabel0,title,filename):
    fig, ax = plt.subplots()
    ax.set(xlabel=xlabel0,ylabel=ylabel0)
    ax.set_title(title)
    ax.set_xlim(xminmax)
    ax.set_ylim(yminmax)
    ax.plot(x,y)
    plt.savefig(filename)
    plt.show()

    
#------------------------------------------------------------------------------
def plot_stack(x,y):
    fig, ax = plt.subplots()
    ax.stackplot(x,y,
            baseline = 'zero')
    plt.show()
    
#------------------------------------------------------------------------------
def plot_stack_logx(x,y):
    fig, ax = plt.subplots()
    ax.set_xlim(min(x), max(x))
    ax.set_xscale('log')
    ax.stackplot(x,y,
            baseline = 'zero')
    plt.show()
    
#------------------------------------------------------------------------------
def plot_hist(vec,xminmax, yminmax, xlabel, ylabel,title,filename):
    n, bins, patches = plt.hist(vec, 50, normed=1, facecolor='green', alpha=0.5)

    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis([xminmax[0],xminmax[1], yminmax[0], yminmax[1]])
    plt.grid(True)
    
    plt.savefig(filename)
    plt.show()
        
def plot_cum_hist(vec,xminmax, yminmax, xlabel, ylabel,title):
    n, bins, patches = plt.hist(vec, 50, normed=1, facecolor='green', alpha=0.5, cumulative=True)

    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis([xminmax[0],xminmax[1], yminmax[0], yminmax[1]])
    plt.grid(True)
    
    plt.show()

#------------------------------------------------------------------------------
# contour plot with legend

def contour_with_legend(x_data,y_data,z_data):            
    origin = 'lower'
    
    X, Y = np.meshgrid(x_data, y_data)    
    nr, nc = z_data.shape
    
    # contourf will convert these to masked
 
    
    Z = np.ma.array(z_data)
    
    # We are using automatic selection of contour levels;
    # this is usually not such a good idea, because they don't
    # occur on nice boundaries, but we do it here for purposes
    # of illustration.
    CS = plt.contourf(X, Y, Z, 20, cmap=plt.cm.bone, origin=origin)
    
    # Note that in the following, we explicitly pass in a subset of
    # the contour levels used for the filled contours.  Alternatively,
    # We could pass in additional levels to provide extra resolution,
    # or leave out the levels kwarg to use all of the original levels.
    
    CS2 = plt.contour(CS, levels=CS.levels[::2], colors='r', origin=origin)
    
    plt.title('Nonsense')
    plt.xlabel('x_data')
    plt.ylabel('y_data')
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('verbosity coefficient')
    # Add the contour line levels to the colorbar
    cbar.add_lines(CS2)
    
    plt.figure()
    plt.show()   
    

# make plots of system cost vs battery cost
    
#------------------------------------------------------------------------------
def plot_system_cost_vs_battery_cost(
    res,
    verbose):
    
    plot_line_logx(res['fix_cost_storage'], res['system_cost'])
    

#------------------------------------------------------------------------------
def plot_cost_allocation_vs_battery_cost(res,verbose):
    
    cost_natgas = res['fix_cost_natgas']*res['capacity_natgas'] + res['var_cost_natgas']*res['dispatch_natgas']
    cost_solar = res['fix_cost_solar']*res['capacity_solar'] + res['var_cost_solar']*res['dispatch_solar']
    cost_wind = res['fix_cost_wind']*res['capacity_wind'] + res['var_cost_wind']*res['dispatch_wind']
    cost_nuclear = res['fix_cost_nuclear']*res['capacity_nuclear'] + res['var_cost_nuclear']*res['dispatch_nuclear']
    cost_storage = res['fix_cost_storage']*res['capacity_storage'] + res['var_cost_storage']*res['dispatch_from_storage']
    cost_unmet_demand = res['var_cost_unmet_demand']*res['dispatch_unmet_demand']
  
    # plot figure
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16.1803398875,10))
    x = res['fix_cost_storage']
    ax.set_xlim(min(x), max(x))
    ax.set_xscale('log')
#    plt.plot([],[],color='C1', label='natgas', linewidth=5)
#    plt.plot([],[],color='C2', label='solar', linewidth=5)
#    plt.plot([],[],color='C3', label='wind', linewidth=5)
#    plt.plot([],[],color='C4', label='nuclear', linewidth=5)
#    plt.plot([],[],color='C5', label='storage', linewidth=5)

    plt.xlabel('battery fixed cost ($/kW/h)', fontsize=24)
    plt.ylabel('contributon to system cost ($/kWh)', fontsize=24)
    plt.title('Contribution to system cost dependence on battery cost', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    pal = sns.color_palette("Set1")
    ax.stackplot(x,cost_natgas,cost_solar,cost_wind,cost_nuclear,cost_storage,cost_unmet_demand,
                 labels=['natgas','solar','wind','nuclear','storage','unmet_demand'],colors=pal,baseline = 'zero',
                 alpha=0.6)
    plt.legend(loc='lower right',fontsize=20)
    plt.savefig('foo.png', bbox_inches='tight')
    plt.show()
    
#------------------------------------------------------------------------------
def plot_generation_vs_battery_cost(res,file_info,verbose):
    
    disp_natgas =  res['dispatch_natgas']
    disp_solar =  res['dispatch_solar']
    disp_wind = res['dispatch_wind']
    disp_nuclear =  res['dispatch_nuclear']
    disp_storage = res['dispatch_from_storage'] -res['dispatch_to_storage']
    disp_unmet_demand = res['dispatch_unmet_demand']
    disp_curtailment = res['dispatch_curtailment']
    
    # plot figure
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16.1803398875,10))
    x = res['fix_cost_storage']
    ax.set_xlim(min(x), max(x))
    ax.set_xscale('log')
#    plt.plot([],[],color='C1', label='natgas', linewidth=5)
#    plt.plot([],[],color='C2', label='solar', linewidth=5)
#    plt.plot([],[],color='C3', label='wind', linewidth=5)
#    plt.plot([],[],color='C4', label='nuclear', linewidth=5)
#    plt.plot([],[],color='C5', label='storage', linewidth=5)

    plt.xlabel('battery fixed cost ($/kW/h)', fontsize=24)
    plt.ylabel(' contributon to generation (kWh/kWh)', fontsize=24)
    plt.title('Generation contribution dependence on battery cost', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    pal = sns.color_palette("Set1")
    ax.stackplot(x,disp_natgas,disp_solar,disp_wind,disp_nuclear,disp_storage,disp_unmet_demand,-disp_curtailment,
                 labels=['natgas','solar','wind','nuclear','storage','unmet_demand','curtailment'],colors=pal,baseline = 'zero',
                 alpha=0.6)
    plt.legend(loc='lower right',fontsize=20)
    file_prefix = file_info['output_folder']+'/'+file_info['base_case_switch']+'_'+file_info['case_switch']
    plt.savefig(file_prefix + '_generation_contributions.png', bbox_inches='tight')
    plt.show()
    
#------------------------------------------------------------------------------
def plot_fixed_cost_allocation_vs_battery_cost(res,verbose):
    
    cost_natgas = res['fix_cost_natgas']*res['capacity_natgas'] 
    cost_solar = res['fix_cost_solar']*res['capacity_solar']
    cost_wind = res['fix_cost_wind']*res['capacity_wind']
    cost_nuclear = res['fix_cost_nuclear']*res['capacity_nuclear']
    cost_storage = res['fix_cost_storage']*res['capacity_storage']
    cost_unmet_demand = 0*cost_natgas
  
    # plot figure
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16.1803398875,10))
    x = res['fix_cost_storage']
    ax.set_xlim(min(x), max(x))
    ax.set_xscale('log')
#    plt.plot([],[],color='C1', label='natgas', linewidth=5)
#    plt.plot([],[],color='C2', label='solar', linewidth=5)
#    plt.plot([],[],color='C3', label='wind', linewidth=5)
#    plt.plot([],[],color='C4', label='nuclear', linewidth=5)
#    plt.plot([],[],color='C5', label='storage', linewidth=5)

    plt.xlabel('battery fixed cost ($/kW/h)', fontsize=24)
    plt.ylabel('fixed cost contributon to system cost ($/kWh)', fontsize=24)
    plt.title('Fixed cost contribution to system cost dependence on battery cost', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    pal = sns.color_palette("Set1")
    ax.stackplot(x,cost_natgas,cost_solar,cost_wind,cost_nuclear,cost_storage,cost_unmet_demand,
                 labels=['natgas','solar','wind','nuclear','storage','unmet_demand'],colors=pal,baseline = 'zero',
                 alpha=0.6)
    plt.legend(loc='lower right',fontsize=20)
    plt.savefig('foo.png', bbox_inches='tight')
    plt.show()
    
#------------------------------------------------------------------------------
def plot_variable_cost_allocation_vs_battery_cost(res,verbose):
    
    cost_natgas =  res['var_cost_natgas']*res['dispatch_natgas']
    cost_solar =  res['var_cost_solar']*res['dispatch_solar']
    cost_wind = res['var_cost_wind']*res['dispatch_wind']
    cost_nuclear =  res['var_cost_nuclear']*res['dispatch_nuclear']
    cost_storage = res['var_cost_storage']*res['dispatch_from_storage']
    cost_unmet_demand = res['var_cost_unmet_demand']*res['dispatch_unmet_demand']
  
    # plot figure
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16.1803398875,10))
    x = res['fix_cost_storage']
    ax.set_xlim(min(x), max(x))
    ax.set_xscale('log')
#    plt.plot([],[],color='C1', label='natgas', linewidth=5)
#    plt.plot([],[],color='C2', label='solar', linewidth=5)
#    plt.plot([],[],color='C3', label='wind', linewidth=5)
#    plt.plot([],[],color='C4', label='nuclear', linewidth=5)
#    plt.plot([],[],color='C5', label='storage', linewidth=5)

    plt.xlabel('battery fixed cost ($/kW/h)', fontsize=24)
    plt.ylabel('variable cost contributon to system cost ($/kWh)', fontsize=24)
    plt.title('Variable cost contribution to system cost dependence on battery cost', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    pal = sns.color_palette("Set1")
    ax.stackplot(x,cost_natgas,cost_solar,cost_wind,cost_nuclear,cost_storage,cost_unmet_demand,
                 labels=['natgas','solar','wind','nuclear','storage','unmet_demand'],colors=pal,baseline = 'zero',
                 alpha=0.6)
    plt.legend(loc='lower right',fontsize=20)
    plt.savefig('foo.png', bbox_inches='tight')
    plt.show()
    
#------------------------------------------------------------------------------
def plot_costs_vs_battery_cost(res,file_info,verbose):
    
    fix_cost_natgas = res['fix_cost_natgas']*res['capacity_natgas']
    fix_cost_solar = res['fix_cost_solar']*res['capacity_solar']
    fix_cost_wind = res['fix_cost_wind']*res['capacity_wind'] 
    fix_cost_nuclear = res['fix_cost_nuclear']*res['capacity_nuclear'] 
    fix_cost_storage = res['fix_cost_storage']*res['capacity_storage']
    fix_cost_unmet_demand = 0 * fix_cost_natgas
    
    var_cost_natgas =  res['var_cost_natgas']*res['dispatch_natgas']
    var_cost_solar =  res['var_cost_solar']*res['dispatch_solar']
    var_cost_wind =  res['var_cost_wind']*res['dispatch_wind']
    var_cost_nuclear =  res['var_cost_nuclear']*res['dispatch_nuclear']
    var_cost_storage =  res['var_cost_storage']*res['dispatch_from_storage']
    var_cost_unmet_demand = res['var_cost_unmet_demand']*res['dispatch_unmet_demand']
    
    cost_natgas = res['fix_cost_natgas']*res['capacity_natgas'] + res['var_cost_natgas']*res['dispatch_natgas']
    cost_solar = res['fix_cost_solar']*res['capacity_solar'] + res['var_cost_solar']*res['dispatch_solar']
    cost_wind = res['fix_cost_wind']*res['capacity_wind'] + res['var_cost_wind']*res['dispatch_wind']
    cost_nuclear = res['fix_cost_nuclear']*res['capacity_nuclear'] + res['var_cost_nuclear']*res['dispatch_nuclear']
    cost_storage = res['fix_cost_storage']*res['capacity_storage'] + res['var_cost_storage']*res['dispatch_from_storage']
    cost_unmet_demand = res['var_cost_unmet_demand']*res['dispatch_unmet_demand']
  
    # plot figure
    # Three subplots, the axes array is 1-d
    fig, (ax0,ax1,ax2) = plt.subplots(3,figsize=(16.1803398875,3*10))
    plt.style.use('ggplot')
    x = res['fix_cost_storage']
    ax0.set_xlim(min(x), max(x))
    ax0.set_xscale('log')
    ax1.set_xlim(min(x), max(x))
    ax1.set_xscale('log')
    ax2.set_xlim(min(x), max(x))
    ax2.set_xscale('log')
#    plt.plot([],[],color='C1', label='natgas', linewidth=5)
#    plt.plot([],[],color='C2', label='solar', linewidth=5)
#    plt.plot([],[],color='C3', label='wind', linewidth=5)
#    plt.plot([],[],color='C4', label='nuclear', linewidth=5)
#    plt.plot([],[],color='C5', label='storage', linewidth=5)

    ax0.set_xlabel('battery fixed cost ($/kW/h)', fontsize=24)
    ax1.set_xlabel('battery fixed cost ($/kW/h)', fontsize=24)
    ax2.set_xlabel('battery fixed cost ($/kW/h)', fontsize=24)
    ax0.set_ylabel('contributon to system cost ($/kWh)', fontsize=24)
    ax1.set_ylabel('fixed cost contributon to system cost ($/kWh)', fontsize=24)
    ax2.set_ylabel('variable contributon to system cost ($/kWh)', fontsize=24)
    ax0.set_title('Contribution to system cost dependence on battery cost', fontsize=24)
    ax1.set_title('Fixed cost contribution to system cost dependence on battery cost', fontsize=24)
    ax2.set_title('Variable cost contribution to system cost dependence on battery cost', fontsize=24)
    ax0.tick_params(axis='both',labelsize=20)
    ax1.tick_params(axis='both',labelsize=20)
    ax2.tick_params(axis='both',labelsize=20)
    pal = sns.color_palette("Set1")
    ax0.stackplot(x,cost_natgas,cost_solar,cost_wind,cost_nuclear,cost_storage,cost_unmet_demand,
                 labels=['natgas','solar','wind','nuclear','storage','unmet_demand'],colors=pal,baseline = 'zero',
                 alpha=0.6)
    ax1.stackplot(x,fix_cost_natgas,fix_cost_solar,fix_cost_wind,fix_cost_nuclear,fix_cost_storage,fix_cost_unmet_demand,
                 labels=['natgas','solar','wind','nuclear','storage','unmet_demand'],colors=pal,baseline = 'zero',
                 alpha=0.6)
    ax2.stackplot(x,var_cost_natgas,var_cost_solar,var_cost_wind,var_cost_nuclear,var_cost_storage,var_cost_unmet_demand,
                 labels=['natgas','solar','wind','nuclear','storage','unmet_demand'],colors=pal,baseline = 'zero',
                 alpha=0.6)
    ax0.legend(loc='lower right',fontsize=20)
    ax1.legend(loc='lower right',fontsize=20)
    ax2.legend(loc='lower right',fontsize=20)
#    ax1.set_legend(loc='lower right',fontsize=20)
#    ax2.set_legend(loc='lower right',fontsize=20)
    file_prefix = file_info['output_folder']+'/'+file_info['base_case_switch']+'_'+file_info['case_switch']
    plt.savefig(file_prefix + '_cost_contributions.png', bbox_inches='tight')
    plt.show()
    
 
#------------------------------------------------------------------------------
#--------------------------------------- Battery use analysis -----------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# For each hour, compute: battery size needed to satisfy that hour, 
# and mean and maximum residence time of storage.
    
# first do for one battery  
# for result in result_list: battery_analysis(result)
 
def battery_analysis(assumption,result,verbose):
    dispatch_to_storage = result['dispatch_to_storage'].value
    dispatch_from_storage = result['dispatch_from_storage'].value
    energy_storage = result['energy_storage'].value # energy storage is at beginning of time step)
    storage_charging_efficiency = assumption['storage_charging_efficiency']
    
    num_time_periods = len(dispatch_to_storage)
    
    # create a lifo stack for use in analysis
    # first step is to get the initial condition for the stack.
    #  To do this, we cycle through the time periods once, without any
    #  calculation of statistics

    lifo_stack = []
    
    for idx in range(num_time_periods):
                    
        if dispatch_to_storage[idx] > 0:  # push on stack (with time moved up 1 cycle)
            lifo_stack.append([idx-num_time_periods,dispatch_to_storage[idx]*storage_charging_efficiency ])
                
        if dispatch_from_storage[idx] > 0:
            dispatch_remaining = dispatch_from_storage[idx]
            while dispatch_remaining > 0:
                #print len(lifo_stack),dispatch_from_storage[idx],dispatch_remaining
                if len(lifo_stack) != 0:
                    top_of_stack = lifo_stack.pop()
                    if top_of_stack[1] > dispatch_remaining:
                        # partial removal
                        new_top = top_of_stack
                        new_top[1] = top_of_stack[1] - dispatch_remaining
                        lifo_stack.append(new_top)
                        dispatch_remaining = 0
                    else:
                        dispatch_remaining = dispatch_remaining - top_of_stack[1]
                else:
                    dispatch_remaining = 0 # stop while loop if stack is empty
                    
    # Now we have the stack as an initial condition and can do it for real
    max_headroom = np.zeros(num_time_periods)
    mean_residence_time = np.zeros(num_time_periods)
    max_residence_time = np.zeros(num_time_periods)
    
    for idx in range(num_time_periods):
        
        
        max_head = 0
        mean_res = 0
        max_res = 0
        
        if dispatch_to_storage[idx] > 0:  # push on stack
            lifo_stack.append([idx,dispatch_to_storage[idx]*storage_charging_efficiency ])
                
        if dispatch_from_storage[idx] > 0:
            dispatch_remaining = dispatch_from_storage[idx]
            accum_time = 0
            while dispatch_remaining > 0:
                if lifo_stack != []:
                    top_of_stack = lifo_stack.pop()
                    if top_of_stack[1] > dispatch_remaining:
                        # partial removal
                        accum_time = accum_time + dispatch_remaining * (idx - top_of_stack[0])
                        new_top = top_of_stack
                        new_top[1] = top_of_stack[1] - dispatch_remaining
                        lifo_stack.append(new_top) # put back the remaining power at the old time
                        dispatch_remaining = 0
                    else: 
                        # full removal of top of stack
                        accum_time = accum_time + top_of_stack[1] * (idx - top_of_stack[0])
                        dispatch_remaining = dispatch_remaining - top_of_stack[1]
                else:
                    if verbose:
                        print ["problem. should not exhaust battery with dispatch_from_storage not satisfied."]
                        print 'idx = ',idx,' dispatch_remaining = ',dispatch_remaining
                    dispatch_remaining = 0 # stop while loop if stack is empty
            
            mean_res = accum_time / dispatch_from_storage[idx]
            max_res = idx - top_of_stack[0]
            # maximum headroom needed is the max of the storage between idx and top_of_stack[0]
            #    minus the amount of storage at time idx + 1
            energy_vec = np.concatenate([energy_storage,energy_storage,energy_storage])
            max_head = np.max(energy_vec[top_of_stack[0]+num_time_periods:idx+1+num_time_periods]) - energy_vec[idx + 1 + num_time_periods]
            #if verbose:
            #    print max_head, np.max(energy_vec[top_of_stack[0]+num_time_periods:idx+num_time_periods]),  energy_vec[idx  + num_time_periods],  energy_vec[idx + 1  + num_time_periods]
                   
        max_headroom[idx] = max_head
        mean_residence_time[idx] = mean_res
        max_residence_time[idx] = max_res
    
    return max_headroom,mean_residence_time,max_residence_time
 
def unique(vec):
    output = []
    for x in vec[:].tolist():
        if x not in output:
            output.append(x)
    return np.array(output)
#------------------------------------------------------------------------------
# Analyze max_headroom to get a list of d dispatch_from_storage / d max_headroom
def cycles_per_year(result, max_headroom, verbose):
    #dispatch_to_storage = result['dispatch_to_storage'].value
    dispatch_from_storage = np.ravel(result['dispatch_from_storage'].value)
    #energy_storage = result['energy_storage'].value # energy storage is at beginning of time step)
    #storage_charging_efficiency = assumption['storage_charging_efficiency']
    
    null = 0 * dispatch_from_storage
    hrt = np.transpose(np.array((max_headroom,dispatch_from_storage)))
    hrt1 = hrt[hrt[:,0].argsort()] # sort by max headroom
    # table is now <headroom, dispatch, cumulative_dispatch>
    
    hrt0_unique = np.sort(unique(hrt1[:,0])).tolist()
    output = []
    for headroom in hrt0_unique:
        subset = hrt1[hrt1[:,0] == headroom]
        record = [
                headroom,
                np.sum(subset[:,1]), # dispatch
                0., # margingal increase in headroom
                0., # cumulative dispatch
                0., #  increase in headroom / increase in dispatch
                0. #  increase in dispatch / increase in headroom
                ]
        output.append(record)
        
    output = np.array(output)
    output[1:,2]=output[1:,0]-output[:-1,0] # marginal increase in headroom
    output[:,3] = np.cumsum(output[:,1]) # take cumulative sum 
    output[1:,4] = output[1:,2]/output[1:,1] # increase in headroom per kWh delivered
    output[1:,5] = output[1:,1]/output[1:,2] # increase in kWh delivered per increase in headroom
    
    output100 = output[::200]
    output100[1:,1] = output100[1:,3] - output100[:-1,3] # dispatch
    output100[1:,2] = output100[1:,0] - output100[:-1,0] # marginal increase in headroom
    output100[1:,4] = output100[1:,2]/output100[1:,1] # increase in headroom per kWh delivered
    output100[1:,5] = output100[1:,1]/output100[1:,2] # increase in kWh delivered per increase in headroom

    headroom_table = output
    
    return headroom_table,output100

# recalculate dispatch_from_storage if there was a smaller battery
def dispatch_from_new(assumption,result,storage_capacity_new):
    dispatch_to_storage = result['dispatch_to_storage'].value
    dispatch_from_storage = np.ravel(result['dispatch_from_storage'].value)
    energy_storage = result['energy_storage'].value # energy storage is at beginning of time step)
    storage_charging_efficiency = assumption['storage_charging_efficiency']
    
    energy_storage_new = energy_storage
    num_time_periods = len(energy_storage)
    unmet_storage_demand = np.zeros(num_time_periods) 
    additional_curtailment = np.zeros(num_time_periods) 
    
    for idx in range(2 * num_time_periods):  # go around twice so battery is initialized properly
        idx_0 = idx % num_time_periods
        idx_1 = (idx + 1) % num_time_periods
        energy_storage_new[idx_1] = energy_storage_new[ idx_0 ] + storage_charging_efficiency * dispatch_to_storage[idx_0] - dispatch_from_storage[idx_0]
        if energy_storage_new[idx_1] < 0:
            unmet_storage_demand[idx_0] = -energy_storage_new[idx_1]
            energy_storage_new[idx_1] = 0
        if energy_storage_new[idx_1] > storage_capacity_new:
            additional_curtailment[idx_0] = energy_storage_new[idx_1] - storage_capacity_new
            energy_storage_new[idx_1] = storage_capacity_new
           
    return energy_storage_new,unmet_storage_demand,additional_curtailment
 
def iterate_cycles_per_year(assumption,result,storage_sizes):
    nsteps = len(storage_sizes)
    output = []
    for step in range(nsteps):
        storage_capacity_new = storage_sizes[step]
        es,ud,ac = dispatch_from_new(assumption,result,storage_capacity_new)
        output.append([storage_capacity_new,np.sum(ud)])
    output_array = np.array(output)
    return output_array

def list_iterate_cycles_per_year(assumption_list,result_list,min_store,max_store,step_store,verbose):
    nsteps = int(round(1 + (max_store-min_store)/step_store))
    capacity_storage = (np.array([dic['capacity_storage'].value for dic in result_list])).tolist()
    storage_sizes = (10.**(min_store + step_store*np.arange(nsteps))).tolist()
    for cs in capacity_storage:
        if cs not in storage_sizes:
            storage_sizes.append(cs)
    storage_sizes.sort()
    output_array = []
    nsteps = len(assumption_list)
    for idx in range(nsteps):
        if verbose:
            print idx,nsteps
        output_array.append(iterate_cycles_per_year(assumption_list[idx],result_list[idx],storage_sizes))
    return np.array(output_array)

#------------------------------------------------------------------------------
def interp_list_iterate_cycles(list_iterate_output):
    out_list = []
    for idx0 in range(len(list_iterate_output)):
        iterate_output = list_iterate_output[idx0,:,:]
        print idx0,iterate_output.shape
        out_item = []
        for idx in range(len(iterate_output)):
            print idx,len(iterate_output)
            if idx > 0:
                out_item.append([
                        iterate_output[idx-1,0], 
                        iterate_output[idx,0],
                        iterate_output[idx-1,1], 
                        iterate_output[idx,1], 
                        (iterate_output[idx-1,1]-iterate_output[idx,1])/(iterate_output[idx,0]-iterate_output[idx-1,0])
                        ])
        out_list.append(out_item)
    return np.array(out_list)           
    
    

#===============================================================================
#================================================== EXECUTION SECTION ==========
#===============================================================================

verbose = True

#google_drive = 'C:/Users/kcaldeira/Google Drive' # home
google_drive = 'C:/Users/Ken/Google Drive' # work

file_nuc = google_drive + '/simple energy system model/Kens version/Results/idealized_nuc_bat 20180219_110625/idealized_nuc_bat.pickle'
file_solar_wind = google_drive + '/simple energy system model/Kens version/Results/idealized_solar_wind_bat 20180219_105732/idealized_solar_wind_bat.pickle'
file_natgas = google_drive + '/simple energy system model/Kens version/Results/idealized_natgas_bat 20180219_104940/idealized_natgas_bat.pickle'

file_path_name = '/Users/Ken/Google Drive/simple energy system model/Kens version/Results/idealized_nate_spectrum 20180225_152620/idealized_nate_spectrum.pickle'
case_name = 'nuc_bat'
#file_path_name = file_solar_wind
#case_name = 'solar_wind_bat'
#num_case = 6

if verbose:
    print file_path_name

file_info, time_series, assumption_list, result_list = unpickle_raw_results(
        file_path_name,
        verbose
        )
res = prepare_scalar_variables(
        file_info,
        time_series,
        assumption_list,
        result_list,
        verbose
        )

#plot_line_logx(oa5[:,0],oa5[:,1])

res = list_iterate_cycles_per_year(assumption_list,result_list,-4,4,1,True)

xy_data_list = interp_list_iterate_cycles(res)
capacity_storage = (np.array([dic['capacity_storage'].value for dic in result_list])).tolist()
test_amts = (xy_data_list[0,:,1] + xy_data_list[0,:,1])/2.

contour_with_legend(test_amts[:-9],capacity_storage[:-8],xy_data_list[:-8,:-9,4])


#print  plot_generation_vs_battery_cost(res,file_info,verbose)
#print  plot_costs_vs_battery_cost(res,file_info,verbose)

#max_head,mean_time,max_time = battery_analysis(assumption_list[num_case],result_list[num_case], True)

#num_time_periods = len(result_list[num_case]['dispatch_to_storage'].value)

#ht,ht_smooth = cycles_per_year(result_list[num_case],max_head,True)

#capacity_storage = result_list[num_case]['capacity_storage'].value

#print plot_line(range(num_time_periods),result_list[num_case]['energy_storage'].value,[0,8760],[0,capacity_storage],'hour of year','energy storage (kWh/kW mean demand)',
#                'energy storage by hour of year',case_name+'_'+'energy_storage_line.png')
#
#print plot_line(range(num_time_periods),max_head,[0,8760],[0,capacity_storage],'hour of year','energy storage (kWh/kW mean demand)',
#                'energy storage by hour of year',case_name+'_'+'max_head_line.png')
#print plot_line(range(num_time_periods),np.sort(max_head),[0,8760],[0,capacity_storage],'hour of year','energy storage (kWh/kW mean demand)',
#                'energy storage by hour of year',case_name+'_'+'max_head_sort_line.png')
#
#print plot_hist(max_head,[0,1200 ],[0,0.03],'battery capacity (kWh/kW)','prob of being in bin','Frequency of battery capacity needed',case_name+'_'+'max_head_high.png')
#print plot_hist(max_head,[0,1200 ],[0,0.003],'battery capacity (kWh/kW)','prob of being in bin','Frequency of battery capacity needed',case_name+'_'+'max_head_low.png')

#print plot_line(range(num_time_periods),mean_time,case_name+'_'+'meantime_line.png')

#print plot_hist(mean_time,[0,8760],[0,0.003],'mean hours in battery','prob of being in bin','Frequency of battery time needed',case_name+'_'+'mean_time_low.png')
#print plot_hist(mean_time,[0,8760],[0,0.0003],'mean hours in battery','prob of being in bin','Frequency of battery time needed',case_name+'_'+'mean_time_tiny.png')

#print plot_cum_hist(max_head,[0,5],[0,1],'battery capacity (kWh/kW)','prob of being in bin','Frequency of battery capacity needed')
#print plot_cum_hist(mean_time,[0,1600],[0,1],'mean hours in battery','prob of being in bin','Frequency of battery time needed')
              
#print plot_line(ht[:,0],ht[:,4],[0,capacity_storage],[0,1],'battery size kWh/kW mean demand','battery capacity needed per additional kW battery dispatch',
#                'battery capacity/dispatch ratio',case_name+'_'+'battery__capacity_dispatch_ratio.png')              
#print plot_line(ht[:,0],ht[:,5],[0,capacity_storage],[0,max(ht[:,5])],'battery size kWh/kW mean demand','additional kWh battery dispatch\nper year per kWh battery capacity added',
#                'battery dispatch/capacity ratio',case_name+'_'+'battery_dispatch_capacity_ratio.png')
#print plot_line(ht_smooth[:,0],ht_smooth[:,4],[0,capacity_storage],[0,1],'battery size kWh/kW mean demand','battery capacity needed per additional kW battery dispatch',
#                'battery capacity/dispatch ratio',case_name+'_'+'battery__capacity_dispatch_ratio_smooth.png')              
#print plot_line(ht_smooth[:,0],ht_smooth[:,5],[0,capacity_storage],[0,max(ht_smooth[:,5])],'battery size kWh/kW mean demand','additional kWh battery dispatch\nper year per kWh battery capacity added',
#                'battery dispatch/capacity ratio',case_name+'_'+'battery_dispatch_capacity_ratio_smooth.png')