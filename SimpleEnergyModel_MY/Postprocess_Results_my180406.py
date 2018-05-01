"""
file name: Postprocess_Results.py
    unpickle and plot results

authors:
    Fan (original model)
    Ken (reorganized model)
    Lei (updated figure templates)
    Mengyao (nuclear vs. renewables)
    
version: _my180406
    simplified script for reading results and generating plots
    modified from _Lei180401 for preliminary analysis of nuclear vs. renewables 
    
to-dos: 
    (a) time-series data have 8640 hours instead of 8760 hours - why?
    (b) how to differentiate over-generation to curtailment or to storage? - currently double-counting generation to storage
    
"""

#%% import modules

import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle

# master plot settings
    # reference on font family: https://matplotlib.org/examples/api/font_family_rc.html
    # check later: subplot title font size > figure text size?
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['font.size'] = 8

## set rcParams back to default values
#    # reference: https://stackoverflow.com/questions/26413185/how-to-recover-matplotlib-defaults-after-setting-stylesheet
#mpl.rcParams.update(mpl.rcParamsDefault)

#%% function: unpickle results

def unpickle_raw_results(
        file_path_name,
        verbose
        ):
    
    with open(file_path_name, 'rb') as db:
       file_info, time_series, assumption_list, result_list = pickle.load(db)
    
    if verbose:
        print 'data unpickled from ' + file_path_name
    
    return file_info, time_series, assumption_list, result_list
    
#%% function: read and combine results from specified folder
    
def combine_results(file_path):

    results_all_cases = {}  # dictionary of combined assumptions and results of all cases
    verbose = True          # set to "true" to print file info
    
    # unpickle results in specified folder
    os.chdir(file_path)
    for file in glob.glob("*.pickle"):
        
        case_name = os.path.splitext(file)[0]   # case name of current file, e.g., ng_fixed_nuc
        # unpickle results
        file_info, time_series, assumption_list, result_list = unpickle_raw_results(
                    file_path + file,
                    verbose
                    )
    
        results = {}    # dictionary of combined assumptions and results for each case
    
        for i in range(len(assumption_list)):   # number of sets of assumptions, e.g., number of nuclear fixed costs
            
            # temp = {'variable name': arrays of assumptions or results}
            temp = {}   # temporary dictionary for storing results
    
            # time-varying demand (input data)
            temp['demand'] = time_series['demand_series']
            
            # fixed cost for each technology           
            temp['fix_cost_natgas'] = assumption_list[i]['fix_cost_natgas']
            temp['fix_cost_solar'] = assumption_list[i]['fix_cost_solar']
            temp['fix_cost_wind'] = assumption_list[i]['fix_cost_wind']
            temp['fix_cost_nuclear'] = assumption_list[i]['fix_cost_nuclear']    
            temp['fix_cost_storage'] = assumption_list[i]['fix_cost_storage'] 
            
            # variable cost for each technology (assumptions)
            temp['var_cost_natgas'] = assumption_list[i]['var_cost_natgas']
            temp['var_cost_solar'] = assumption_list[i]['var_cost_solar']
            temp['var_cost_wind'] = assumption_list[i]['var_cost_wind']            
            temp['var_cost_nuclear'] = assumption_list[i]['var_cost_nuclear']
            temp['var_cost_dispatch_to_storage'] = assumption_list[i]['var_cost_dispatch_to_storage']
            temp['var_cost_dispatch_from_storage'] = assumption_list[i]['var_cost_dispatch_from_storage']
            temp['var_cost_storage'] = assumption_list[i]['var_cost_storage']
    
            # capacity of each technology (results)
            temp['capacity_natgas'] = result_list[i]['capacity_natgas']
            temp['capacity_solar'] = result_list[i]['capacity_solar']
            temp['capacity_wind'] = result_list[i]['capacity_wind']
            temp['capacity_nuclear'] = result_list[i]['capacity_nuclear']
            temp['capacity_storage'] = result_list[i]['capacity_storage']
    
            # time-varying dispatch and curtailment of each technology (results)
            temp['dispatch_natgas'] = result_list[i]['dispatch_natgas']
            temp['dispatch_solar'] = result_list[i]['dispatch_solar']
            temp['dispatch_wind'] = result_list[i]['dispatch_wind']
            temp['dispatch_nuclear'] = result_list[i]['dispatch_nuclear']
            temp['dispatch_to_storage'] = result_list[i]['dispatch_to_storage']
            temp['dispatch_from_storage'] = result_list[i]['dispatch_from_storage']
            temp['energy_storage'] = result_list[i]['energy_storage']
            temp['curtailment_solar'] = result_list[i]['curtailment_solar']
            temp['curtailment_wind'] = result_list[i]['curtailment_wind']
            temp['curtailment_nuclear'] = result_list[i]['curtailment_nuclear']
            
            # system cost (results)
            temp['system_cost'] = result_list[i]['system_cost']
            
            # results = {'index of assumption set': {'variable name': arrays of assumptions or results} }
            results[i] = temp
                
        # results_all_cases = {'case name': {'index of assumption set': {'variable name': arrays of assumptions or results} } }
        results_all_cases[case_name] = results
        
    return results_all_cases
    
#%% function: organize specified variable into arrays, organize variable arrays from specified cases into dictionaries

def organize_results(results_all_cases, cases, var_name):

    var_all_cases = {}  # dictionary of variable arrays for all cases
    
    for case in cases:     # number of cases
        # organize specified variable into array for each case
        var = []    # variable array for each specified case    
        for i in range(len(results_all_cases[case])):  # number of sets of assumptions, e.g., number of nuclear fixed costs
            var = np.append(var, results_all_cases[case][i][var_name])
        # reshape variable array
            # number of rows = number of sets of assumptions
            # number of columns = 1 if not time series, = number of timesteps if time series
        var = np.squeeze(np.reshape(var, [len(results_all_cases[case]),-1]))
        # organize variable arrays for all cases into dictionaries
        var_all_cases[case] = var
    
    return var_all_cases    
    
#%% function: calculate total cost contributions and stack up results for single-variable plots
    
def single_var_results(file_path, cases):
    
    # -------------------------------------------------------------------------
    # assign variable names
    
    # read in and combine assumptions and results
    results_all_cases = combine_results(file_path)

    # fixed cost for each technology (assumptions)
    fix_cost_natgas = organize_results(results_all_cases, cases, var_name='fix_cost_natgas')
    fix_cost_solar = organize_results(results_all_cases, cases, var_name='fix_cost_solar')
    fix_cost_wind = organize_results(results_all_cases, cases, var_name='fix_cost_wind')
    fix_cost_nuclear = organize_results(results_all_cases, cases, var_name='fix_cost_nuclear')  # "x" in single-variable plot
    fix_cost_storage = organize_results(results_all_cases, cases, var_name='fix_cost_storage')
    
    # variable cost for each technology (assumptions)
    var_cost_natgas = organize_results(results_all_cases, cases, var_name='var_cost_natgas')
    var_cost_solar = organize_results(results_all_cases, cases, var_name='var_cost_solar')
    var_cost_wind = organize_results(results_all_cases, cases, var_name='var_cost_wind')
    var_cost_nuclear = organize_results(results_all_cases, cases, var_name='var_cost_nuclear')
    var_cost_storage = organize_results(results_all_cases, cases, var_name='var_cost_storage')
    var_cost_dispatch_to_storage = organize_results(results_all_cases, cases, var_name='var_cost_dispatch_to_storage')
    var_cost_dispatch_from_storage = organize_results(results_all_cases, cases, var_name='var_cost_dispatch_from_storage')
    
    # capacity of each technology (results)
    capacity_natgas = organize_results(results_all_cases, cases, var_name='capacity_natgas')
    capacity_solar = organize_results(results_all_cases, cases, var_name='capacity_solar')
    capacity_wind = organize_results(results_all_cases, cases, var_name='capacity_wind')
    capacity_nuclear = organize_results(results_all_cases, cases, var_name='capacity_nuclear')
    capacity_storage = organize_results(results_all_cases, cases, var_name='capacity_storage')
    
    # time-varying dispatch and curtailment of each technology (results)
    dispatch_natgas = organize_results(results_all_cases, cases, var_name='dispatch_natgas')
    dispatch_solar = organize_results(results_all_cases, cases, var_name='dispatch_solar')
    dispatch_wind = organize_results(results_all_cases, cases, var_name='dispatch_wind')
    dispatch_nuclear = organize_results(results_all_cases, cases, var_name='dispatch_nuclear')
    energy_storage = organize_results(results_all_cases, cases, var_name='energy_storage')
    dispatch_to_storage = organize_results(results_all_cases, cases, var_name='dispatch_to_storage')
    dispatch_from_storage = organize_results(results_all_cases, cases, var_name='dispatch_from_storage')
    curtailment_solar = organize_results(results_all_cases, cases, var_name='curtailment_solar')
    curtailment_wind = organize_results(results_all_cases, cases, var_name='curtailment_wind')
    curtailment_nuclear = organize_results(results_all_cases, cases, var_name='curtailment_nuclear')
        
    # -------------------------------------------------------------------------
    # process results: calculate costs, stack up results by technology
    
    capacity = {}   # stacked capacity contributions
    fix_cost = {}   # stacked fixed cost contributions to system cost
    var_cost = {}   # stacked variable cost contributions to system cost
    tot_cost = {}   # stacked total cost (fixed + variable) contributions to system cost
    
    for case in cases:
        
        # stack up capacity of each technology
        capacity[case] = np.vstack([capacity_natgas[case], 
                                    capacity_solar[case], 
                                    capacity_wind[case], 
                                    capacity_nuclear[case], 
                                    capacity_storage[case]
                                    ])
                                    
        # calculate and stack up fixed cost contribution of each technology to system cost
        fix_cost[case] = np.vstack([fix_cost_natgas[case] * capacity_natgas[case],
                                    fix_cost_solar[case] * capacity_solar[case],
                                    fix_cost_wind[case] * capacity_wind[case],
                                    fix_cost_nuclear[case] * capacity_nuclear[case],
                                    fix_cost_storage[case] * capacity_storage[case]
                                    ])
    
        # calculate and stack up variable cost contribution of each technology to system cost
        var_cost[case] = np.vstack([var_cost_natgas[case] * np.mean(dispatch_natgas[case], axis=1),
                                    var_cost_solar[case] * np.mean((dispatch_solar[case] + curtailment_solar[case]), axis=1),
                                    var_cost_wind[case] * np.mean((dispatch_wind[case] + curtailment_wind[case]), axis=1),
                                    var_cost_nuclear[case] * np.mean((dispatch_nuclear[case] + curtailment_nuclear[case]), axis=1),
                                    (var_cost_storage[case] * np.mean(energy_storage[case], axis=1) + 
                                    var_cost_dispatch_to_storage[case] * np.mean(dispatch_to_storage[case], axis=1) + 
                                    var_cost_dispatch_from_storage[case] * np.mean(dispatch_from_storage[case], axis=1))
                                    ])
                                    
        # calculate and stack up total cost contribution of each technology to system cost
        tot_cost[case] = fix_cost[case] + var_cost[case]
    
    # "x" = fix_cost_nuclear
    # "y" (stacked) = capcity, fix_cost, var_cost, tot_cost
    return fix_cost_nuclear, capacity, fix_cost, var_cost, tot_cost

#%% function: stack up results for time series plots

def time_series_results(file_path, cases, base_cost_idx):

    # -------------------------------------------------------------------------
    # assign variable names
    
    # read in and combine assumptions and results
    results_all_cases = combine_results(file_path)
    
    # time-varying demand (input) for specified cases
    demand = organize_results(results_all_cases, cases, var_name='demand')
    
    # time-varying dispatch and curtailment of each technology (results) for specified cases
    dispatch_natgas = organize_results(results_all_cases, cases, var_name='dispatch_natgas')
    dispatch_solar = organize_results(results_all_cases, cases, var_name='dispatch_solar')
    dispatch_wind = organize_results(results_all_cases, cases, var_name='dispatch_wind')
    dispatch_nuclear = organize_results(results_all_cases, cases, var_name='dispatch_nuclear')
    energy_storage = organize_results(results_all_cases, cases, var_name='energy_storage')
    dispatch_to_storage = organize_results(results_all_cases, cases, var_name='dispatch_to_storage')
    dispatch_from_storage = organize_results(results_all_cases, cases, var_name='dispatch_from_storage')
    curtailment_solar = organize_results(results_all_cases, cases, var_name='curtailment_solar')
    curtailment_wind = organize_results(results_all_cases, cases, var_name='curtailment_wind')
    curtailment_nuclear = organize_results(results_all_cases, cases, var_name='curtailment_nuclear')
    
    # -------------------------------------------------------------------------
    # process results: stack up results by technology and dispatch type
        # note: see definitions in Core_Model
    
    dispatch = {}       # dispatch = generation used, including dispatch from storage
    curtailment = {}    # curtailment = generation not used
    tot_dispatch = {}   # total dispatch = "useful" generation + energy flows into / out of storage + curtailment
    
    for case in cases:
        
        # dispatch from each technology = generation used to meet demand
        dispatch[case] = np.vstack([dispatch_natgas[case][base_cost_idx],
                                    dispatch_solar[case][base_cost_idx],
                                    dispatch_wind[case][base_cost_idx],
                                    dispatch_nuclear[case][base_cost_idx],
                                    dispatch_from_storage[case][base_cost_idx]
                                    ])

        # curtailment from solar, wind, and nuclear = amount generated but not used
        curtailment[case] = np.vstack([curtailment_solar[case][base_cost_idx],
                                      curtailment_wind[case][base_cost_idx],
                                      curtailment_nuclear[case][base_cost_idx]
                                      ])

        # total dispatch = dispatch + energy flows into / out of storage + curtailment
        tot_dispatch[case] = np.vstack([dispatch[case],
                                        dispatch_to_storage[case][base_cost_idx],
                                        curtailment[case]
                                        ])

    # demand and dispatch_to_storage are output for all cost assumptions
    # dispatch, curtailment, and tot_dispatch are output for specified baseline cost
    return demand, dispatch_to_storage, dispatch, curtailment, tot_dispatch


#%% function: calculate differences in system cost and reshape results for contour plots

def contour_results(file_path, fixed_cases, flex_cases, ng_cases):

    # -------------------------------------------------------------------------
    # assign variable names

    # read in and combine assumptions and results
    results_all_cases = combine_results(file_path)

    # nuclear fixed costs, nuclear variable costs, and system costs for all cases
    fix_cost_nuclear = organize_results(results_all_cases, (fixed_cases + flex_cases), var_name='fix_cost_nuclear')
    var_cost_nuclear = organize_results(results_all_cases, (fixed_cases + flex_cases), var_name='var_cost_nuclear')
    system_cost = organize_results(results_all_cases, (fixed_cases + flex_cases), var_name='system_cost')
    
    # -------------------------------------------------------------------------
    # process results: calculate differences in system cost, reshape arrays for contour plots
    # note: 
        # (a) programming approach from: 
            # https://stackoverflow.com/questions/21352129/matplotlib-creating-2d-arrays-from-1d-arrays-is-there-a-nicer-way
        # (b) case name (e.g., fixed_cases[0]) used for calling fixed and variable costs does not matter, since all cases are modeled using same cost assumptions for nuclear
        
    # number of colums in reshaped arrays = number of unique elements in either cost array
    cols = np.unique(var_cost_nuclear[fixed_cases[0]]).shape[0]
    
    # reshape array of nuclear fixed and variable costs
    fix_cost_nuclear_matrix = fix_cost_nuclear[fixed_cases[0]].reshape(-1,cols)   # "x" = nuclear fixed costs
    var_cost_nuclear_matrix = var_cost_nuclear[fixed_cases[0]].reshape(-1,cols)   # "y" = nuclear variable costs
    
    # calculate and reshape differences in system cost
    delta_system_cost = {}              # differences in system cost between cases with fixed and flexible nuclear
    delta_system_cost_matrix = {}       # reshaped arrays of differences in system cost
    rel_delta_system_cost = {}          # relative differences in system cost between cases with fixed and flexible nuclear
    rel_delta_system_cost_matrix = {}   # reshaped arrays of relative differences in system cost

    for i in range(len(ng_cases)):
        
        # calculate differences in system cost
        delta_system_cost[ng_cases[i]] = system_cost[fixed_cases[i]] - system_cost[flex_cases[i]]           # absolute differences
        rel_delta_system_cost[ng_cases[i]] = delta_system_cost[ng_cases[i]]/system_cost[fixed_cases[i]]     # differences relative to costs with fixed nuclear
    
        # reshape arrays of differences in system cost
        delta_system_cost_matrix[i] = delta_system_cost[ng_cases[i]].reshape(-1,cols)           # "z" = differences in system cost
        rel_delta_system_cost_matrix[i] = rel_delta_system_cost[ng_cases[i]].reshape(-1,cols)   # "z" = relative differences in system cost
    
    # reshaped arrays for contour plots
        # "x" (reshaped) = fix_cost_nuclear_matrix
        # "y" (reshaped) = var_cost_nuclear_matrix
        # "z" (reshaped) = delta_system_cost_matrix, rel_delta_system_cost_matrix
    return fix_cost_nuclear_matrix, var_cost_nuclear_matrix, delta_system_cost_matrix, rel_delta_system_cost_matrix

#%% function: single-variable stacked area plot (Figure 1)

def single_var_plot(
        x, y,
        xlabel_text, ylabel_text,
        cases, subplot_titles, technologies
        ):

    # set up figure layout and size
    nrows = 3       # number of rows of subplots = number of cases for natural gas (baseline, CCS, no natural gas)
    ncols = 2       # number of columns of subplots = number of cases for nuclear (fixed, flexible)
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(5,6.5))    
    case_num = 0    # number of cases, counter for indexing results                         
    
    # fill in each subplot
    for ng_num in range(nrows):         # cases for natural gas 
        for nuc_num in range(ncols):    # cases for nuclear
            # stacked area plot
            ax[ng_num,nuc_num].stackplot(x[cases[case_num]], y[cases[case_num]])
            # subplot title
            ax[ng_num,nuc_num].set_title(subplot_titles[case_num])
            # reverse x-axis
                # note to self: can't reverse x-axis?
#            ax[ng_num,nuc_num].invert_xaxis()
            # counter increases by one
            case_num += 1   
    
    # legend and axis labels
    fig.legend(labels=technologies, loc='lower left', bbox_to_anchor=(0.95,0.1), ncol=1)
    fig.text(0.5, 0.05, xlabel_text, horizontalalignment='center')
    fig.text(0.025, 0.58, ylabel_text, horizontalalignment='center', rotation='vertical')

    # adjust spacing within and around subplots
        # references: 
            # https://stackoverflow.com/questions/37558329/matplotlib-set-axis-tight-only-to-x-or-y-axis
            # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.subplots_adjust(hspace = 0.3)

    return fig
    
#%% function: time series plot (Figure 2)
    
def time_series_plot(
        y_stack, y_curve, 
        xlabel_text, ylabel_text, 
        subplot_titles, cases, technologies
        ):
    
    # set up figure layout and size
    fig, ax = plt.subplots(len(cases), 1, sharex=True, sharey=True, figsize=(4,3))
    
    # fill in each subplot
    for i in range(len(cases)):
    
        # stacked dispatch (tot_dispatch = output for specified baseline cost)
        ax[i].stackplot(range(len(y_stack[cases[i]][0])),y_stack[cases[i]])   # generation used to meet demand
        
        # demand curve (demand = output for all cost assumptions)
        ax[i].plot(range(len(y_curve[cases[i]][0])),y_curve[cases[i]][0], 'k')
        
        # subplot titles and x-axis settings        
        ax[i].set_title(subplot_titles[i])          # subplot titles
        ax[i].set_xlim(1440,1440+120)               # time period shown - can adjust        
        ax[len(cases)-1].set_xlabel(xlabel_text)    # x-axis label (lower center)

    # legend and y-axis label
    fig.legend(labels=technologies, loc='lower left', bbox_to_anchor=(1.05,0.1), ncol=1)
    fig.text(0.01, 0.64, ylabel_text, horizontalalignment='center', rotation='vertical')    # y-axis label (center left)

    # adjust spacing
    plt.tight_layout()
    
    return fig
    
#%% function: contour plot (Figure 3)
    
def contour_plot(
        x, y, z,
        xlabel_text, ylabel_text, cbar_label_text,
        ng_cases, subplot_titles
        ):
  
    # set up figure layout and size
    fig, ax = plt.subplots(1, len(ng_cases), sharex=True, sharey=True, figsize=(5.6,1.6))
    
    # fill in each subplot
    for i in range(len(ng_cases)):
        
        # note: 
            # nuclear fixed and variable costs in ¢/kWh - change scale if needed
            # differences in system cost in ¢/kWh or percentage
        cs = ax[i].contourf(x*100, y*100, z[i]*100, cmap='viridis')
        ax[i].set_title(subplot_titles[i])      # subplot titles
#        ax[i].set_xlim(6,10)
#        ax[i].set_ylim(0,0.02)
        
        # axis labels
        ax[0].set_ylabel(ylabel_text)  # y-axis label (center left)
        ax[1].set_xlabel(xlabel_text)  # x-axis label (lower center)
    
        # specify location and size of colorbar
        # references on colorbar formatting:
            # https://matplotlib.org/gallery/subplots_axes_and_figures/subplots_adjust.html#sphx-glr-gallery-subplots-axes-and-figures-subplots-adjust-py
            # https://matplotlib.org/examples/pylab_examples/contourf_demo.html
            # https://jakevdp.github.io/PythonDataScienceHandbook/04.07-customizing-colorbars.html
        # note to self: how to rescale colorbar?        
        cax = plt.axes([0.95, 0.1, 0.02, 0.8])
#        cbar = plt.colorbar(cs, cax=cax)    # absolute differences
        cbar = plt.colorbar(cs, cax=cax, format='%.0f%%')   # relative differences in percentage
        cbar.set_label(cbar_label_text)

    return fig

#%% generate single-variable plots

# specify folder containing results
file_path1 = 'D:/Mengyao @ Carnegie/research/models/model_my180406/Results/1 year/EIA costs/varying nuclear fixed cost/'

# case names for calling results for single-variable plots
cases1 = ['ng_fixed_nuc',
         'ng_flex_nuc',
         'ngccs_fixed_nuc',
         'ngccs_flex_nuc',
         'no_ng_fixed_nuc',
         'no_ng_flex_nuc']

# results for single-variable plots
    # "x" = fix_cost_nuclear
    # "y" (stacked) = capcity, fix_cost, var_cost, tot_cost
fix_cost_nuclear, capacity, fix_cost, var_cost, tot_cost = single_var_results(file_path1, cases1)

# x-axis and y-axis labels - check if match variables specified
xlabel_text1 = 'Nuclear fixed cost [$/kWh]'
ylabel_text1a = 'Normalized capacity [kW]'       # capacity shares vs. nuclear fixed cost
ylabel_text1b = 'System cost [$/kWh]'            # contribution to system cost vs. nuclear fixed cost

# subplot titles - check if match case names
subplot_titles1 = ['Baseline NG, fixed nuclear',
                  'Baseline NG, flexible nuclear',
                  'NG + CCS, fixed nuclear',
                  'NG + CCS, flexible nuclear',
                  'No NG, fixed nuclear',
                  'No NG, flexible nuclear']
                  
# technologies considered (= legend) - check if match variables specified
technologies1 = ['Natural gas', 'Solar', 'Wind', 'Nuclear', 'Storage']

# single-variable plot: capacity shares vs. nuclear fixed cost
fig1a = single_var_plot(fix_cost_nuclear, capacity, xlabel_text1, ylabel_text1a, cases1, subplot_titles1, technologies1)

# single-variable plot: system cost shares vs. nuclear fixed cost
fig1b = single_var_plot(fix_cost_nuclear, tot_cost, xlabel_text1, ylabel_text1b, cases1, subplot_titles1, technologies1)

## save figures - change title when needed
#fig1a.savefig('capacity shares' + '.png', dpi=300, bbox_inches='tight', pad_inches=0.2)    
#fig1b.savefig('system cost shares' + '.png', dpi=300, bbox_inches='tight', pad_inches=0.2)

#%% time series plot

## specify folder containing results
file_path2 = 'D:/Mengyao @ Carnegie/research/models/model_my180406/Results/1 year/EIA costs/varying nuclear fixed cost/'

# case names for calling results for demand and dispatch curves
cases2 = ['no_ng_fixed_nuc', 'no_ng_flex_nuc']

# technologies considered (= legend) - check if match variables specified
technologies2 = ['Demand', 
                 'Gen.: natural gas', 'Gen.: solar', 'Gen.: wind', 'Gen.: nuclear', 'Gen.: from storage',
                 'Gen.: to storage',
                 'Curt.: solar', 'Curt.: wind', 'Curt.: nuclear']

# subplot titles - check if match case names
subplot_titles2 = ['No NG, fixed nuclear', 'No NG, flexible nuclear']

# index of baseline fixed nuclear cost (value = $0.05/kWh)
base_cost_idx = 1   # second value in series

# results for time series plots     
    # demand and dispatch_to_storage are output for all cost assumptions
    # dispatch, curtailment, and tot_dispatch are output for specified baseline cost
demand, dispatch_to_storage, dispatch, curtailment, tot_dispatch = time_series_results(file_path2, cases2, base_cost_idx)

# x-axis and y-axis labels - check if match variables specified
xlabel_text2 = 'Hours'
ylabel_text2 = 'Normalized dispatch'

# time series plot: demand and total dispatch (= dispatch + curtailment + storage flows)
fig2 = time_series_plot(tot_dispatch, demand, 
                        xlabel_text2, ylabel_text2, 
                        subplot_titles2, cases2, technologies2)
                        
## save figures - change title when needed
#fig2.savefig('dispatch_no_NG' + '.png', dpi=300, bbox_inches='tight', pad_inches=0.2)    

#%% generate contour plot

# specify folder containing results
file_path3 = 'D:/Mengyao @ Carnegie/research/models/model_my180406/Results/1 year/EIA costs/varying nuclear fixed and variable costs/'

# case names with fixed (non-dispatchable) or flexible (dispatchable) nuclear
fixed_cases = ['ng_fixed_nuc', 'ngccs_fixed_nuc', 'no_ng_fixed_nuc']
flex_cases = ['ng_flex_nuc', 'ngccs_flex_nuc', 'no_ng_flex_nuc']

# case names for different natural gas scenarios
ng_cases = ['ng','ngccs','no_ng']  

# subplot titles - check if same order as case names
subplot_titles3 = ['Baseline NG', 'NG + CCS', 'No NG']

# x-axis, y-axis, and colorbar labels - check if match variables and units specified
xlabel_text3 = 'Nuclear fixed cost [' u'\xa2' '/kWh]'       # unit of ¢/kWh
ylabel_text3 = 'Nuclear variable cost [' u'\xa2' '/kWh]'    # unit of ¢/kWh
cbar_label_text = u'\u0394' ' System cost'    # in percentage or ¢/kWh

# results for contour plot
fix_cost_nuclear, var_cost_nuclear, delta_system_cost, rel_delta_system_cost = contour_results(file_path3, fixed_cases, flex_cases, ng_cases)

# contour plot: differences in system cost
fig3 = contour_plot(fix_cost_nuclear, var_cost_nuclear, rel_delta_system_cost, 
                    xlabel_text3, ylabel_text3, cbar_label_text, 
                    ng_cases, subplot_titles3)

# save figures - change title when needed
#fig3.savefig('relative value of dispatchability' + '.tif', dpi=300, bbox_inches='tight')  
    
#%% plot system cost shares for one case - temporary

#fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(3,2))   
#
## plot system cost shares vs. fixed nuclear cost
#ax.stackplot(fix_cost_nuclear['ngccs_flex_nuc'], tot_cost['ngccs_flex_nuc'])
#
## subplot title
#ax.set_title('NG + CCS, flexible nuclear')
#ax.set_xlabel('Nuclear fixed cost [$/kWh]')
#ax.set_ylabel('System cost [$/kWh]')
#ax.set_ylim(0,0.08)
#
## legend and axis labels
#fig.legend(labels=technologies, loc='center left', bbox_to_anchor=(1,0.6), ncol=1)

# save figure
#fig.savefig('system cost shares_ngcc flex nuc' + '.png', dpi=300, bbox_inches='tight', pad_inches=0.2)