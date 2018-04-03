"""
Post-processing
Created by Lei at 27 March, 2018
"""

# -----------------------------------------------------------------------------

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
#from matplotlib import style
#style.use('ggplot')

import pickle

#===============================================================================
#================================================= DEFINITION SECTION ==========
#===============================================================================

def unpickle_raw_results(
        file_path_name,
        verbose
        ):
    
    with open(file_path_name, 'rb') as db:
       file_info, time_series, assumption_list, result_list = pickle.load (db)
    
    if verbose:
        print 'data unpickled from '+file_path_name
    
    return file_info, time_series, assumption_list, result_list

def prepare_scalar_variables (
        file_info,
        time_series,
        assumption_list,
        result_list,
        verbose
        ):
    
    num_scenarios    = len(assumption_list)
    res = {}

    # put all scenarios data in one list res;
    for idx in range(num_scenarios):
        tmp = {}
        
        tmp['demand']         = np.array(np.squeeze(time_series['demand_series'])) #/ num_time_periods
        tmp['solar_capacity'] = np.array(np.squeeze(time_series['solar_series']))  #/ num_time_periods
        tmp['wind_capacity']  = np.array(np.squeeze(time_series['wind_series']))   #/ num_time_periods
        
        tmp['fix_cost_natgas']  = np.array(np.squeeze(assumption_list[idx]['fix_cost_natgas']))
        tmp['fix_cost_solar']   = np.array(np.squeeze(assumption_list[idx]['fix_cost_solar']))
        tmp['fix_cost_wind']    = np.array(np.squeeze(assumption_list[idx]['fix_cost_wind']))
        tmp['fix_cost_nuclear'] = np.array(np.squeeze(assumption_list[idx]['fix_cost_nuclear']))
        tmp['fix_cost_storage'] = np.array(np.squeeze(assumption_list[idx]['fix_cost_storage']))
        
        tmp['var_cost_natgas']        = np.array(np.squeeze(assumption_list[idx]['var_cost_natgas']))
        tmp['var_cost_solar']         = np.array(np.squeeze(assumption_list[idx]['var_cost_solar']))
        tmp['var_cost_wind']          = np.array(np.squeeze(assumption_list[idx]['var_cost_wind']))
        tmp['var_cost_nuclear']       = np.array(np.squeeze(assumption_list[idx]['var_cost_nuclear']))
        tmp['var_cost_storage']       = np.array(np.squeeze(assumption_list[idx]['var_cost_storage']))
        tmp['var_cost_dispatch_to_storage']    = np.array(np.squeeze(assumption_list[idx]['var_cost_dispatch_to_storage']))
        tmp['var_cost_dispatch_from_storage']  = np.array(np.squeeze(assumption_list[idx]['var_cost_dispatch_from_storage']))
        tmp['var_cost_unmet_demand']  = np.array(np.squeeze(assumption_list[idx]['var_cost_unmet_demand']))
        
        tmp['capacity_natgas']  = np.array(np.squeeze(result_list[idx]['capacity_natgas']))
        tmp['capacity_solar']   = np.array(np.squeeze(result_list[idx]['capacity_solar']))
        tmp['capacity_wind']    = np.array(np.squeeze(result_list[idx]['capacity_wind']))
        tmp['capacity_nuclear'] = np.array(np.squeeze(result_list[idx]['capacity_nuclear']))
        tmp['capacity_storage'] = np.array(np.squeeze(result_list[idx]['capacity_storage']))
        
        tmp['dispatch_natgas']        = np.array(np.squeeze(result_list[idx]['dispatch_natgas']))       #/ num_time_periods
        tmp['dispatch_solar']         = np.array(np.squeeze(result_list[idx]['dispatch_solar']))        #/ num_time_periods
        tmp['dispatch_wind']          = np.array(np.squeeze(result_list[idx]['dispatch_wind']))         #/ num_time_periods
        tmp['dispatch_nuclear']       = np.array(np.squeeze(result_list[idx]['dispatch_nuclear']))      #/ num_time_periods
        tmp['dispatch_to_storage']    = np.array(np.squeeze(result_list[idx]['dispatch_to_storage']))   #/ num_time_periods
        tmp['dispatch_from_storage']  = np.array(np.squeeze(result_list[idx]['dispatch_from_storage'])) #/ num_time_periods
        tmp['dispatch_unmet_demand']  = np.array(np.squeeze(result_list[idx]['dispatch_unmet_demand'])) #/ num_time_periods
        tmp['dispatch_curtailment']   = np.array(np.squeeze(result_list[idx]['dispatch_curtailment']))  #/ num_time_periods
        tmp['energy_storage']         = np.array(np.squeeze(result_list[idx]['energy_storage']))        #/ num_time_periods
        
        tmp['system_cost']    = np.array(np.squeeze(result_list[idx]['system_cost']))  
        tmp['storage_charging_efficiency']    = np.array(np.squeeze(assumption_list[idx]['storage_charging_efficiency']))

        res[idx] = tmp
    
    return res

#------------------------------------------------------------------------------
#------------------------------------------------ Plotting function -----------
#------------------------------------------------------------------------------   

def get_multicases_results(res, num_case, var, *avg_option):
    x = []
    for idx in range(num_case):
        tmp_var = res[idx][var]
        x.append(np.array(tmp_var))
    if avg_option:
        y = avg_series(x, 
                       num_case,
                       avg_option[0], 
                       avg_option[1], 
                       avg_option[2],
                       avg_option[3])
        return y
    else:
        return np.array(x)

def avg_series(var, num_case, beg_step, end_step, nstep, num_return):
    x = []
    y = []
    if num_case > 1:
        for idx in range(num_case):
            hor_mean = np.mean(var[idx][beg_step-1:end_step].reshape(-1,nstep),axis=1)
            ver_mean = np.mean(var[idx][beg_step-1:end_step].reshape(-1,nstep),axis=0)
            x.append(hor_mean)
            y.append(ver_mean)
    else:
        hor_mean = np.mean(var[beg_step-1:end_step].reshape(-1,nstep),axis=1)
        ver_mean = np.mean(var[beg_step-1:end_step].reshape(-1,nstep),axis=0)
        x.append(hor_mean)
        y.append(ver_mean)
    if num_return == 1:
        return np.array(x)
    if num_return == 2:
        return np.array(y)
    

def cal_cost(fix_cost, capacity,
             var_cost, dispatch,
             num_case, num_time_periods,
             *battery_dispatch):
    
    cost_fix = np.array(fix_cost * capacity)
    
    cost_var = np.zeros(num_case)
    for idx in range(num_case):
        if battery_dispatch:
            cost_var_tmp = var_cost[idx]       * np.sum(dispatch[idx]) + \
                           np.array(battery_dispatch[0][0]) * np.sum(np.array(battery_dispatch[0][1])) +\
                           np.array(battery_dispatch[2][1]) * np.sum(np.array(battery_dispatch[0][3]))
        else:
            cost_var_tmp = var_cost[idx] * np.sum(dispatch[idx]) 
        cost_var[idx] = cost_var_tmp
        
    cost_tot = cost_fix + cost_var
    return cost_fix, cost_var, cost_tot


# --------- stack plot1
"""
def plot_stack_single(x,y_ne,y_po,labels,colors,info, *battery_charge):
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1),(0,0))
    ax1.grid(False) #, color='k', linestyle='--', alpha=0.3)
    
    ax1.stackplot(x, y_ne, colors=colors[::-1], baseline = 'zero',alpha = 0.5)
    ax1.stackplot(x, y_po, labels=labels, colors=colors, baseline = 'zero',alpha = 0.5)
    if battery_charge:
        ax1.plot(x, np.array(battery_charge[0][0]),c='r', linewidth = '0.5', linestyle='-', label='charge')
        ax1.plot(x, np.array(battery_charge[0][1]),c='g', linewidth = '0.5', linestyle='-', label='discharge')
        ax1.fill_between(x, np.array(battery_charge[0][0]), np.array(battery_charge[0][1]), hatch='//', alpha=0, label='energy loss')
    y_line = np.zeros(y_po.shape[1])
    for idx in range(int(y_po.shape[0])):
        y_line = y_line + y_po[idx]
        ax1.plot(x, y_line, c='k', linewidth = 0.5)    

    ax1.set_xticks([0.1,0.08,0.06,0.05,0.04,0.035,0.03,0.025,0.02,0.015,0.01,0.005,0.00])
    ax1.set_xlim(0.10,0.00)
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    #plt.legend(loc="best")
    #plt.xlabel(info["xlabel"])
    #plt.ylabel(info["ylabel"])
    #plt.title(info["title"])
    plt.show()
    #plt.savefig(info["fig_name"]+'.png',dpi=300)
    plt.clf()
    
def plot_stack2(x,y_ne,y_po,labels,colors,demand,info):
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1),(0,0))
    ax1.grid(False) #True, color='k', linestyle='--', alpha=0.3)
    
    ax1.stackplot(x, y_ne, colors=colors, baseline = 'zero',alpha = 0.5)
    ax1.stackplot(x, y_po, labels=labels, colors=colors, baseline = 'zero',alpha = 0.5)
    ax1.plot(x, demand, c='k', linewidth = 2, linestyle = '-', label = 'demand')  
    total_gen = np.sum(y_po,axis=0)
    
    total_energy_gen = np.sum(y_po[:-1,:],axis=0)
    ax1.fill_between(x,demand,total_energy_gen, demand<total_energy_gen, hatch='//', alpha = 0.)
    

    y_line = np.zeros(y_po.shape[1])
    for idx in range(int(y_po.shape[0])):
        y_line = y_line + y_po[idx]
        ax1.plot(x, y_line, c='k', linewidth = 0.5)    
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
        
    #ax1.set_xlim(1,24)
    #ax1.set_ylim(-1,1)
   
    plt.legend(loc="best")
    plt.xlabel(info["xlabel"])
    plt.ylabel(info["ylabel"])
    plt.title(info["title"])
    plt.show()
    #plt.savefig(info["fig_name"]+'.p',dpi=300)
    plt.clf()
    
    
    
def plot_battery_cycle(x,dispatch_to,dispatch_from):
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1),(0,0))
    ax1.grid(False) #True, color='k', linestyle='--', alpha=0.3)
    
    new_line = dispatch_from - dispatch_to
    
    ax1.plot(x, dispatch_from*0., c='k')
    ax1.plot(x, new_line, c='k')
    ax1.fill_between(x,new_line,0.0, new_line>=0.0 ,facecolor="green")
    ax1.fill_between(x,new_line,0.0, new_line<=0.0 ,facecolor="red")    
    
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax1.set_xlim(1,24)

    plt.show()
    #plt.savefig(info["fig_name"]+'.png',dpi=300)
    plt.clf() 
  
"""
    




def plot_multi_panels1(ax,case):
    ax.grid(True, color='k', linestyle='--', alpha=0.2)
    ax.set_axis_bgcolor('white')
    
    #ax.stackplot(case[0], case[1], colors=case[4][::-1], baseline = 'zero', alpha = 0.5)
    ax.stackplot(case[0], case[1], colors=case[4], baseline = 'zero', alpha = 0.5)
    ax.stackplot(case[0], case[2], labels=case[3], colors=case[4],  baseline = 'zero', alpha = 0.5)
    if len(case) == 7:
        #print case[5]["title"], 'dl-------------------'
        #axv = ax.twinx()
        ax.plot(case[0], np.array(case[6][0]),c='r', linewidth = '1', linestyle='-', label='charge')
        ax.plot(case[0], np.array(case[6][1]),c='g', linewidth = '1', linestyle='-', label='discharge')
        ax.fill_between(case[0], np.array(case[6][0]), np.array(case[6][1]), facecolor='black', alpha=0.2, label='energy loss')
    y_line = np.zeros(case[2].shape[1])
    for idx in range(int(case[2].shape[0])):
        y_line = y_line + case[2][idx]
        ax.plot(case[0], y_line, c='k', linewidth = 0.5)
        
    #ax.set_xticks([0.1,0.08,0.06,0.05,0.04,0.03,0.02,0.01,0.00])
    #ax.set_xlim(0.10,0.00)
    ax.set_xlim(case[0][-1],case[0][0])
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax.set_xlabel(case[5]["xlabel"],fontsize=9)
    ax.set_title(case[5]["title"],fontsize=9)   
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    leg = ax.legend(loc='center left', ncol=1, 
                    bbox_to_anchor=(1, 0.5), prop={'size': 5})
    leg.get_frame().set_alpha(0.4)
    
def plot_stack_multi1(case1,case2,case3,case4, case_name):
    fig, axes = plt.subplots(2,2)
    fig.subplots_adjust(top=1, left=0.0, right=1, hspace=0.5, wspace=0.35)
    ((ax1, ax2), (ax3, ax4)) = axes
    
    plot_multi_panels1(ax1,case1)
    plot_multi_panels1(ax2,case2)
    plot_multi_panels1(ax3,case3)
    plot_multi_panels1(ax4,case4)
    
    plt.setp(ax1.get_xticklabels(), size=7)
    plt.setp(ax2.get_xticklabels(), size=7)
    plt.setp(ax3.get_xticklabels(), size=7)
    plt.setp(ax4.get_xticklabels(), size=7)
    
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    
    plt.setp(ax1.get_yticklabels(), size=7)
    plt.setp(ax2.get_yticklabels(), size=7)
    plt.setp(ax3.get_yticklabels(), size=7)
    plt.setp(ax4.get_yticklabels(), size=7)

    #plt.show()
    plt.savefig(case_name+'_MC.pdf',dpi=200,bbox_inches='tight',transparent=True)
    plt.clf() 

def stack_plot1(
        res,
        num_case,
        case_name,
        multipanel):
    
    # --- get Raw Data ---
    num_time_periods = len(res[0]['demand'])

    solar_series      = get_multicases_results(res, num_case , 'solar_capacity')   / num_time_periods
    wind_series       = get_multicases_results(res, num_case , 'wind_capacity')    / num_time_periods

    capacity_natgas   = get_multicases_results(res, num_case , 'capacity_natgas')
    capacity_solar    = get_multicases_results(res, num_case , 'capacity_solar')
    capacity_wind     = get_multicases_results(res, num_case , 'capacity_wind')
    capacity_nuclear  = get_multicases_results(res, num_case , 'capacity_nuclear')
    capacity_storage  = get_multicases_results(res, num_case , 'capacity_storage')
    
    fix_cost_natgas  = get_multicases_results(res, num_case, 'fix_cost_natgas')
    fix_cost_solar   = get_multicases_results(res, num_case, 'fix_cost_solar')
    fix_cost_wind    = get_multicases_results(res, num_case, 'fix_cost_wind')
    fix_cost_nuclear = get_multicases_results(res, num_case, 'fix_cost_nuclear')
    fix_cost_storage = get_multicases_results(res, num_case, 'fix_cost_storage')
    
    var_cost_natgas  = get_multicases_results(res, num_case, 'var_cost_natgas')
    var_cost_solar   = get_multicases_results(res, num_case, 'var_cost_solar')
    var_cost_wind    = get_multicases_results(res, num_case, 'var_cost_wind')
    var_cost_nuclear = get_multicases_results(res, num_case, 'var_cost_nuclear')
    var_cost_storage = get_multicases_results(res, num_case, 'var_cost_storage') 
    var_cost_dispatch_to_storage   = get_multicases_results(res, num_case, 'var_cost_dispatch_to_storage') 
    var_cost_dispatch_from_storage = get_multicases_results(res, num_case, 'var_cost_dispatch_from_storage') 
    
    dispatch_natgas       = get_multicases_results(res, num_case, 'dispatch_natgas')        / num_time_periods
    dispatch_solar        = get_multicases_results(res, num_case, 'dispatch_solar')         / num_time_periods
    dispatch_wind         = get_multicases_results(res, num_case, 'dispatch_wind')          / num_time_periods
    dispatch_nuclear      = get_multicases_results(res, num_case, 'dispatch_nuclear')       / num_time_periods
    dispatch_to_storage   = get_multicases_results(res, num_case, 'dispatch_to_storage')    / num_time_periods
    dispatch_from_storage = get_multicases_results(res, num_case, 'dispatch_from_storage')  / num_time_periods
    energy_storage        = get_multicases_results(res, num_case, 'energy_storage')         / num_time_periods

    # --- global setting ---
    order_list = fix_cost_nuclear.argsort()  
    xaxis = fix_cost_nuclear[order_list]
    
    # -plot1: capacity-
    yaxis_capacity_ne = np.zeros(num_case)
    yaxis_capacity_po = np.vstack([capacity_natgas[order_list], 
                                   capacity_solar[order_list], 
                                   capacity_wind[order_list],
                                   capacity_nuclear[order_list],
                                   capacity_storage[order_list]])
    labels_capacity = ["natgas", "solar", "wind", "nuclear", "storage"]
    colors_capacity = [color_natgas[1], color_solar[1], color_wind[1], color_nuclear[1], color_storage[1]]
    info_capacity = {
            "title": "Capacity mix\n(kW)",
            "xlabel": "Fixed Cost Nuclear ($/kW/h)",
            "ylabel": "Capacity (kW)",
            "fig_name": "Capacity_mix"}    

    # -plot2: total dispatch 
    dispatch_tot_natgas  = np.sum(dispatch_natgas,axis=1)
    dispatch_tot_solar   = np.sum(dispatch_solar,axis=1)
    dispatch_tot_wind    = np.sum(dispatch_wind,axis=1)
    dispatch_tot_nuclear = np.sum(dispatch_nuclear,axis=1)
    dispatch_tot_to_storage   = np.sum(dispatch_to_storage,axis=1)
    dispatch_tot_from_storage = np.sum(dispatch_from_storage,axis=1)
    
    curtail_tot_natgas  = capacity_natgas - dispatch_tot_natgas
    curtail_tot_solar   = capacity_solar * np.sum(solar_series,axis=1) - dispatch_tot_solar
    curtail_tot_wind    = capacity_wind  * np.sum(wind_series,axis=1)  - dispatch_tot_wind
    curtail_tot_nuclear = capacity_nuclear - dispatch_tot_nuclear    
            
    yaxis_dispatch_ne = np.vstack([curtail_tot_natgas[order_list]   * (-1),
                                   curtail_tot_solar[order_list]    * (-1),
                                   curtail_tot_wind[order_list]     * (-1),
                                   curtail_tot_nuclear[order_list]  * (-1)
                                   ])        
    yaxis_dispatch_po = np.vstack([dispatch_tot_natgas[order_list], 
                                   dispatch_tot_solar[order_list], 
                                   dispatch_tot_wind[order_list],
                                   dispatch_tot_nuclear[order_list]])
    battery_charge = np.array([dispatch_tot_to_storage, dispatch_tot_from_storage])
   
    
    labels_dispatch = ["natgas", "solar", "wind", "nuclear"]
    colors_dispatch = [color_natgas[1], color_solar[1], color_wind[1], color_nuclear[1]]    
    info_dispatch = {
            "title": "Total dispatched energy\n(kWh)",
            "xlabel": "Fixed Cost Nuclear ($/kW/h)",
            "ylabel": "Total dispatch (KWh)",
            "fig_name": "Total_dispatch_mix"} 

    
    # -plot3: system_cost
    cost_natgas  = cal_cost(fix_cost_natgas,  capacity_natgas,  var_cost_natgas,  dispatch_natgas,  num_case, num_time_periods)
    cost_solar   = cal_cost(fix_cost_solar,   capacity_solar,   var_cost_solar,   dispatch_solar,   num_case, num_time_periods)
    cost_wind    = cal_cost(fix_cost_wind,    capacity_wind,    var_cost_wind,    dispatch_wind,    num_case, num_time_periods)
    cost_nuclear = cal_cost(fix_cost_nuclear, capacity_nuclear, var_cost_nuclear, dispatch_nuclear, num_case, num_time_periods)
    cost_storage = cal_cost(fix_cost_storage, capacity_storage, var_cost_storage, energy_storage,num_case, num_time_periods, 
                            var_cost_dispatch_to_storage,  dispatch_to_storage,
                            var_cost_dispatch_from_storage,dispatch_from_storage)  # now dispatch_to/from is free    
    
    yaxis_cost_ne = np.zeros(num_case)
    yaxis_cost1_po = np.vstack([cost_natgas[2][order_list], 
                                cost_solar[2][order_list], 
                                cost_wind[2][order_list],
                                cost_nuclear[2][order_list],
                                cost_storage[2][order_list]])
    labels_cost1 = ["natgas", "solar", "wind", "nuclear", "storage"]
    colors_cost1 = [color_natgas[1], color_solar[1], color_wind[1], color_nuclear[1], color_storage[1]]
    info_cost1 = {
            "title": "System cost\n($/kW/h)",
            "xlabel": "Fixed Cost Nuclear ($/kW/h)",
            "ylabel": "System cost ($/kW/h)",
            "fig_name": "System_cost_total"} 
    
    # -plot4: system_cost
    yaxis_cost2_po = np.vstack([cost_natgas[0][order_list],
                                cost_natgas[1][order_list],
                                cost_solar[0][order_list],
                                cost_solar[1][order_list],
                                cost_wind[0][order_list],
                                cost_wind[1][order_list],
                                cost_nuclear[0][order_list],
                                cost_nuclear[1][order_list],
                                cost_storage[0][order_list],
                                cost_storage[1][order_list]]) 
    labels_cost2 = ["natgas_fix",  'natgas_var', 
                    "solar_fix",   'solar_var', 
                    "wind_fix",    'wind_var', 
                    "nuclear_fix", 'nuclear_var', 
                    "storage_fix", 'storage_var',
                    ]
    colors_cost2 = [color_natgas[1],  color_natgas[0],
                    color_solar[1],   color_solar[0], 
                    color_wind[1],    color_wind[0],
                    color_nuclear[1], color_nuclear[0], 
                    color_storage[1], color_storage[0]
                    ]
    info_cost2 = {
            "title": "System cost\n($/kW/h)",
            "xlabel": "Fixed Cost Nuclear ($/kW/h)",
            "ylabel": "System cost ($/kW/h)",
            "fig_name": "System_cost_seperate"} 
    
    plot_case1 = [xaxis, yaxis_capacity_ne, yaxis_capacity_po, labels_capacity, colors_capacity, info_capacity]
    plot_case2 = [xaxis, yaxis_dispatch_ne, yaxis_dispatch_po, labels_dispatch, colors_dispatch, info_dispatch, battery_charge] 
    plot_case3 = [xaxis, yaxis_cost_ne, yaxis_cost1_po, labels_cost1, colors_cost1, info_cost1]
    plot_case4 = [xaxis, yaxis_cost_ne, yaxis_cost2_po, labels_cost2, colors_cost2, info_cost2]   
    
    if multipanel:
        plot_stack_multi1(plot_case1, plot_case2, plot_case3, plot_case4, case_name)
    else:
        print 'please use multipanel = True!'
        #plot_stack_single(xaxis,yaxis_capacity_ne,yaxis_capacity_po, labels_capacity, colors_capacity,info_capacity)
        #plot_stack_single(xaxis,yaxis_dispatch_ne,yaxis_dispatch_po, labels_dispatch, colors_dispatch,info_dispatch, battery_charge)
        #plot_stack_single(xaxis,yaxis_cost_ne,    yaxis_cost1_po,    labels_cost1,    colors_cost1,   info_cost1)
        #plot_stack_single(xaxis,yaxis_cost_ne,    yaxis_cost2_po,    labels_cost2,    colors_cost2,   info_cost2)




# --------- stack plot2
def plot_multi_panels2(ax,case):
    ax.grid(True, color='k', linestyle='--', alpha=0.2)
    ax.set_axis_bgcolor('white')
    
    ax.stackplot(case[0], case[1], colors=case[4], baseline = 'zero', alpha = 0.5)
    ax.stackplot(case[0], case[2], labels=case[3], colors=case[4],  baseline = 'zero', alpha = 0.5)
    ax.plot(case[0], case[5], c='k', linewidth = 1.5, linestyle = '-', label = 'demand')  
    total_energy_gen = np.sum(case[2][:-1,:],axis=0)
    ax.fill_between(case[0],case[5],total_energy_gen, case[5]<total_energy_gen, alpha = 0.0)        
    
    y_line = np.zeros(case[2].shape[1])
    for idx in range(int(case[2].shape[0])):
        y_line = y_line + case[2][idx]
        ax.plot(case[0], y_line, c='grey', linewidth = 0.5)
    y_line = np.zeros(case[1].shape[1])
    for idx in range(int(case[1].shape[0])):
        y_line = y_line + case[1][idx]
        ax.plot(case[0], y_line, c='grey', linewidth = 0.5)
        
        
    ax.set_xlim(case[0][0],case[0][-1])
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax.set_xlabel(case[6]["xlabel"],fontsize=9)
    ax.set_title(case[6]["title"],fontsize=9)   
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    leg = ax.legend(loc='center left', ncol=1, 
                    bbox_to_anchor=(1, 0.5), prop={'size': 5})
    leg.get_frame().set_alpha(0.4)
    
def plot_stack_multi2(case1,case2,case3, case_name):
    fig = plt.figure()
    fig.subplots_adjust(top=1, left=0.0, right=1, hspace=0.7, wspace=0.35)
    
    ax1 = plt.subplot2grid((2,2),(0,0),rowspan=1, colspan=2)
    plot_multi_panels2(ax1,case1)

    ax2 = plt.subplot2grid((2,2),(1,0),rowspan=1, colspan=1)
    plot_multi_panels2(ax2,case2)
    
    ax3 = plt.subplot2grid((2,2),(1,1),rowspan=1, colspan=1,sharey=ax2)
    plot_multi_panels2(ax3,case3)


    plt.setp(ax1.get_xticklabels(), size=7)
    plt.setp(ax2.get_xticklabels(), size=7)
    plt.setp(ax3.get_xticklabels(), size=7)
    
    plt.setp(ax1.get_yticklabels(), size=7)
    plt.setp(ax2.get_yticklabels(), size=7)
    plt.setp(ax3.get_yticklabels(), size=7)

    #plt.show()
    plt.savefig(case_name+'_TP.pdf',dpi=200,bbox_inches='tight',transparent=True)
    plt.clf()    
    
    
def stack_plot2(
        res,
        num_case,
        case_name,
        multipanel,
        *select_case):
    
    # --- data preparation ---
    num_time_periods = len(res[0]['demand'])
    
    find_case_idx = False
    if select_case:
        var1 = get_multicases_results(res, num_case , select_case[0][0])
        var2 = get_multicases_results(res, num_case , select_case[0][1])
        for idx in range(num_case):
            if var1[idx] == select_case[1][0] and var2[idx] == select_case[1][1]:
                find_case_idx = True
                case_idx = idx
                break
                
        if find_case_idx: 
            print 'Find case index:', case_idx
        else:
            print 'Error: no such case'
            sys.exit(0)
        
    if find_case_idx == False:
        case_idx = 0
    
    capacity_natgas   = get_multicases_results(res, num_case , 'capacity_natgas')[case_idx]
    how_many_case = int(capacity_natgas.size)
    if how_many_case > 1:
        print "too many case for time path plot"
        sys.exit(0)
    
    capacity_solar    = get_multicases_results(res, num_case , 'capacity_solar')[case_idx]
    capacity_wind     = get_multicases_results(res, num_case , 'capacity_wind')[case_idx]
    capacity_nuclear  = get_multicases_results(res, num_case , 'capacity_nuclear')[case_idx]
    
    demand_yr = get_multicases_results(res, num_case , 'demand'   ,1,num_time_periods,24,1)[case_idx]
    demand_day1 = get_multicases_results(res, num_case , 'demand'   ,3601,4320,24,2)[case_idx]
    demand_day2 = get_multicases_results(res, num_case , 'demand'   ,7921,8640,24,2)[case_idx]    
    
    solar_series_yr = get_multicases_results(res, num_case , 'solar_capacity'   ,1,num_time_periods,24,1)[case_idx]
    solar_series_day1 = get_multicases_results(res, num_case , 'solar_capacity' ,3601,4320,24,2)[case_idx]
    solar_series_day2 = get_multicases_results(res, num_case , 'solar_capacity' ,7921,8640,24,2)[case_idx]
    
    wind_series_yr  = get_multicases_results(res, num_case , 'wind_capacity'   ,1,num_time_periods,24,1)[case_idx]
    wind_series_day1  = get_multicases_results(res, num_case , 'wind_capacity' ,3601,4320,24,2)[case_idx]
    wind_series_day2  = get_multicases_results(res, num_case , 'wind_capacity' ,7921,8640,24,2)[case_idx]
    
    dispatch_natgas_yr  = get_multicases_results(res, num_case,      'dispatch_natgas',      1,num_time_periods,24,1)[case_idx]
    dispatch_solar_yr   = get_multicases_results(res, num_case,      'dispatch_solar',       1,num_time_periods,24,1)[case_idx]     
    dispatch_wind_yr    = get_multicases_results(res, num_case,      'dispatch_wind',        1,num_time_periods,24,1)[case_idx]          
    dispatch_nuclear_yr = get_multicases_results(res, num_case,      'dispatch_nuclear',     1,num_time_periods,24,1)[case_idx]  
    dispatch_from_storage_yr = get_multicases_results(res, num_case, 'dispatch_from_storage',1,num_time_periods,24,1)[case_idx]
    
    dispatch_natgas_day1  = get_multicases_results(res, num_case,      'dispatch_natgas',      3601,4320,24,2)[case_idx]     
    dispatch_solar_day1   = get_multicases_results(res, num_case,      'dispatch_solar',       3601,4320,24,2)[case_idx]     
    dispatch_wind_day1    = get_multicases_results(res, num_case,      'dispatch_wind',        3601,4320,24,2)[case_idx]          
    dispatch_nuclear_day1 = get_multicases_results(res, num_case,      'dispatch_nuclear',     3601,4320,24,2)[case_idx]  
    dispatch_from_storage_day1 = get_multicases_results(res, num_case, 'dispatch_from_storage',3601,4320,24,2)[case_idx]    
    
    dispatch_natgas_day2  = get_multicases_results(res, num_case,      'dispatch_natgas',      7921,8640,24,2)[case_idx]     
    dispatch_solar_day2   = get_multicases_results(res, num_case,      'dispatch_solar',       7921,8640,24,2)[case_idx]     
    dispatch_wind_day2    = get_multicases_results(res, num_case,      'dispatch_wind',        7921,8640,24,2)[case_idx]          
    dispatch_nuclear_day2 = get_multicases_results(res, num_case,      'dispatch_nuclear',     7921,8640,24,2)[case_idx]  
    dispatch_from_storage_day2 = get_multicases_results(res, num_case, 'dispatch_from_storage',7921,8640,24,2)[case_idx] 
    
    curtail_natgas_yr  = capacity_natgas                    - dispatch_natgas_yr
    curtail_solar_yr   = capacity_solar   * solar_series_yr - dispatch_solar_yr
    curtail_wind_yr    = capacity_wind    * wind_series_yr  - dispatch_wind_yr
    curtail_nuclear_yr = capacity_nuclear                   - dispatch_nuclear_yr

    curtail_natgas_day1  = capacity_natgas                      - dispatch_natgas_day1
    curtail_solar_day1   = capacity_solar   * solar_series_day1 - dispatch_solar_day1
    curtail_wind_day1    = capacity_wind    * wind_series_day1  - dispatch_wind_day1
    curtail_nuclear_day1 = capacity_nuclear                     - dispatch_nuclear_day1

    curtail_natgas_day2  = capacity_natgas                      - dispatch_natgas_day2
    curtail_solar_day2   = capacity_solar   * solar_series_day2 - dispatch_solar_day2
    curtail_wind_day2    = capacity_wind    * wind_series_day2  - dispatch_wind_day2
    curtail_nuclear_day2 = capacity_nuclear                     - dispatch_nuclear_day2

    # Now plot
    xaxis_yr = np.arange(360)+1
    yaxis_yr_ne = np.vstack([curtail_natgas_yr*(-1),
                             curtail_solar_yr*(-1),
                             curtail_wind_yr*(-1),
                             curtail_nuclear_yr*(-1),
                             curtail_natgas_yr*0.0
                             ])
    yaxis_yr_po = np.vstack([dispatch_natgas_yr,
                             dispatch_solar_yr,
                             dispatch_wind_yr,
                             dispatch_nuclear_yr,
                             dispatch_from_storage_yr
                             ]) 
        
    opccinfo1 = select_case[0][0]+'='+str(select_case[1][0])
    opccinfo2 = select_case[0][1]+'='+str(select_case[1][1])
    
    labels = ["natgas", "solar", "wind", "nuclear","discharge"]
    colors = [color_natgas[1], color_solar[1], color_wind[1], color_nuclear[1], color_storage[1]]    
    info_yr = {
            "title": "Daily-average per hour dispatch (kWh)\n(For central case:  " + opccinfo1+';  '+opccinfo2+')',
            "xlabel": "time step (day)",
            "ylabel": "",
            "fig_name": "dispatch_case"}
    
    
    xaxis_day = np.arange(24)+1
    yaxis_day1_ne = np.vstack([curtail_natgas_day1*(-1),
                               curtail_solar_day1*(-1),
                               curtail_wind_day1*(-1),
                               curtail_nuclear_day1*(-1),
                               curtail_natgas_day1*0.0
                               ])
    yaxis_day1_po = np.vstack([dispatch_natgas_day1,
                               dispatch_solar_day1,
                               dispatch_wind_day1,
                               dispatch_nuclear_day1,
                               dispatch_from_storage_day1
                               ]) 
    info_day1 = {
            "title": "Hourly-average per hour dispatch (kWh)\n(June)",
            "xlabel": "time step (hour)",
            "ylabel": "",
            "fig_name": "dispatch_case"}   
    
    yaxis_day2_ne = np.vstack([curtail_natgas_day2*(-1),
                               curtail_solar_day2*(-1),
                               curtail_wind_day2*(-1),
                               curtail_nuclear_day2*(-1),
                               curtail_natgas_day2*0.0
                               ])
    yaxis_day2_po = np.vstack([dispatch_natgas_day2,
                               dispatch_solar_day2,
                               dispatch_wind_day2,
                               dispatch_nuclear_day2,
                               dispatch_from_storage_day2
                               ]) 
    
    info_day2 = {
            "title": "Hourly-average per hour dispatch (kWh)\n(December)",
            "xlabel": "time step (hour)",
            "ylabel": "",
            "fig_name": "dispatch_case"}  
    
    if multipanel:
        plot_case1 = [xaxis_yr, yaxis_yr_ne,yaxis_yr_po,labels, colors, demand_yr, info_yr]
        plot_case2 = [xaxis_day, yaxis_day1_ne,yaxis_day1_po,labels, colors, demand_day1, info_day1]
        plot_case3 = [xaxis_day, yaxis_day2_ne,yaxis_day2_po,labels, colors, demand_day2, info_day2]
        plot_stack_multi2(plot_case1,plot_case2,plot_case3,case_name)
    else:
        print 'please use multipanel = True!'
        #plot_stack2(xaxis_yr,yaxis_yr_ne,yaxis_yr_po,labels, colors, demand_yr, info_yr)
        #plot_stack2(xaxis_day,yaxis_day1_ne,yaxis_day1_po,labels, colors, demand_day1, info_day1)
        #plot_stack2(xaxis_day,yaxis_day2_ne,yaxis_day2_po,labels, colors, demand_day2, info_day1)

    
    
# --------- contour plot
    
def plot_contour(x,y,z,levels,case_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    cs1 = ax.contourf(x,y,z,levels=levels,
                      cmap='PuBu_r',
                      extend='both')
    cs2 = ax.contour(x,y,z,levels=levels[::4],
                     colors='k',
                     linewidths=0.5, 
                     alpha=1.)
    ax.clabel(cs2, inline=1, fontsize=5)
    
    plt.colorbar(cs1, ticks=levels[::2], orientation='vertical')    
    ax.set_title('system_cost\n($)')
    ax.set_xlabel('Fixed cost nuclear\n($/kW/h)')
    ax.set_ylabel('Var cost nuclear\n($/kW/h)') 
    ax.set_xlim(x.max(),x.min())
    
    #plt.show()
    plt.savefig(case_name + '_contour.pdf',dpi=200,bbox_inches='tight',transparent=True)
    plt.clf()    

def create_contour_axes(x,y,z):
    
    def find_z(x_need, y_need):
        tot_idx = len(x)
        for idx in range(tot_idx):
            if x[idx] == x_need and y[idx] == y_need:
                find_z = z[idx]
        return find_z
    
    x_uni = np.unique(x)
    y_uni = np.unique(y)
    z2 = np.ones([ len(x_uni), len(y_uni) ])* (-9999)
    for idx_x in range( len(x_uni) ):
        for idx_y in range( len(y_uni) ):
            x_need = x_uni[idx_x]
            y_need = y_uni[idx_y]
            z2[idx_x, idx_y] = find_z(x_need,y_need)
    z3 = np.ma.masked_values(z2, -9999)
    
    return x_uni, y_uni, z3

def contour_plot(res,num_case,case_name):
    fix_cost_nuclear = get_multicases_results(res, num_case, 'fix_cost_nuclear')
    var_cost_nuclear = get_multicases_results(res, num_case, 'var_cost_nuclear')
    system_cost      = get_multicases_results(res, num_case, 'system_cost')
    x,y,z = create_contour_axes(fix_cost_nuclear, var_cost_nuclear, system_cost)
    levels = np.linspace(z.min(), z.max(), 20)   
    plot_contour(x,y,z,levels,case_name)





# --------- battery plot
    
def battery_TP(xaxis, mean_residence_time, max_residence_time, max_headroom):
    
    y1 = np.squeeze(avg_series(mean_residence_time, 1, 1,8640,24,1))
    y2 = np.squeeze(avg_series(max_residence_time,  1, 1,8640,24,1))
    y3 = np.squeeze(avg_series(max_headroom,        1, 1,8640,24,1))
    
    fig = plt.figure()
    fig.subplots_adjust(top=1, left=0.0, right=1, hspace=0.7, wspace=0.35)

    ax1 = plt.subplot2grid((2,1),(0,0),rowspan=1, colspan=1)
    ax1v = ax1.twinx()
    ln1 = ax1.stackplot(xaxis, y1, colors ='g', baseline = 'zero', alpha=0.5, labels=['Mean residence time'])
    ln2 = ax1.plot(xaxis,      y2, c = 'green', alpha=0.5,    label='Max energy storage (kWh/kW)')
    ln3 = ax1v.plot(xaxis,     y3, c = 'red',   alpha=0.5,    label='Max headroom ')
    
    lns = ln1+ln2+ln3
    labs = [l.get_label() for l in lns]
    leg = ax1.legend(lns, labs, loc='center left', ncol=1, 
                     bbox_to_anchor=(1.07, 0.5), prop={'size': 5})
    leg.get_frame().set_alpha(0.4)
    #for label in ax1.xaxis.get_ticklabels():
    #    label.set_rotation(45)
    
    ax1.set_title('(Left) battery storage required to satisfy demand at each hour hour\n'+\
                  '(Right) maximum headroom required to satisfy demand at each hour hour',
                  fontsize = 10)
    ax1.set_xlabel('time step (day)')
    plt.setp(ax1.get_xticklabels(), size=7)
    plt.setp(ax1.get_yticklabels(), size=7, color='green')
    plt.setp(ax1v.get_yticklabels(), size=7, color='red')
    
    # ---
    array_to_draw = y1
    ax2 = plt.subplot2grid((2,1),(1,0),rowspan=1, colspan=1)
    weights = np.ones_like(array_to_draw)/float(len(array_to_draw))
    ax2.hist(array_to_draw, 50, weights=weights, label = 'Frequency distribution of\nmean residence time')
    leg = ax2.legend(loc='center left', ncol=1, 
                     bbox_to_anchor=(1.07, 0.5), prop={'size': 5})
    
    ax2.set_title('Frequency of battery storage for demand at a particular hour')
    ax2.set_xlabel('Battery storage (kWh/kW)')
    plt.setp(ax2.get_xticklabels(), size=7)
    plt.setp(ax2.get_yticklabels(), size=7)
    
    # ---
    #plt.show()
    plt.savefig(case_name+'_Battery.pdf',dpi=200,bbox_inches='tight',transparent=True)
    plt.clf()
    
def battery_calculation(
        num_time_periods,
        dispatch_to_storage,
        dispatch_from_storage,
        energy_storage,
        storage_charging_efficiency
        ):
    
    start_point = 0.
    for idx in range(num_time_periods):
        if energy_storage[idx] == 0:
            start_point = idx

    lifo_stack = []
    tmp = 0.
    
    for idx in range(num_time_periods-start_point):
        idx = idx + start_point
        tmp = tmp + dispatch_to_storage[idx] - dispatch_from_storage[idx]
              
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
                        new_top = np.copy(top_of_stack)
                        new_top[1] = new_top[1] - dispatch_remaining
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
                        new_top = np.copy(top_of_stack)
                        new_top[1] = new_top[1] - dispatch_remaining
                        lifo_stack.append(new_top) # put back the remaining power at the old time
                        dispatch_remaining = 0
                    else: 
                        # full removal of top of stack
                        accum_time = accum_time + top_of_stack[1] * (idx - top_of_stack[0])
                        dispatch_remaining = dispatch_remaining - top_of_stack[1]
                else:
                    dispatch_remaining = 0 # stop while loop if stack is empty
            
            mean_res = accum_time / dispatch_from_storage[idx]
            max_res = idx - top_of_stack[0]
            # maximum headroom needed is the max of the storage between idx and top_of_stack[0]
            #    minus the amount of storage at time idx + 1
            energy_vec = np.concatenate([energy_storage,energy_storage,energy_storage])
            max_head = np.max(energy_vec[int(top_of_stack[0]+num_time_periods):int(idx + 1+num_time_periods)]) - energy_vec[int(idx + 1 + num_time_periods)]   # dl-->could be negative?
            
        max_headroom[idx] = max_head
        mean_residence_time[idx] = mean_res
        max_residence_time[idx] = max_res
    
    return max_headroom,mean_residence_time,max_residence_time
    
def battery_plot(res,
                 num_case,
                 case_name,
                 multipanels,
                 *select_case):
    
    # --- multi case plot
    num_time_periods = len(res[0]['demand'])
    
    find_case_idx = False
    if select_case:
        var1 = get_multicases_results(res, num_case , select_case[0][0])
        var2 = get_multicases_results(res, num_case , select_case[0][1])
        for idx in range(num_case):
            if var1[idx] == select_case[1][0] and var2[idx] == select_case[1][1]:
                find_case_idx = True
                case_idx = idx
                break
                
        if find_case_idx: 
            print 'Find case index:', case_idx
        else:
            print 'Error: no such case'
            sys.exit(0)
        
    if find_case_idx == False:
        case_idx = 0
    
    dispatch_to_storage         = get_multicases_results(res, num_case, 'dispatch_to_storage')[case_idx]
    dispatch_from_storage       = get_multicases_results(res, num_case, 'dispatch_from_storage')[case_idx]
    energy_storage              = get_multicases_results(res, num_case, 'energy_storage')[case_idx]
    storage_charging_efficiency = get_multicases_results(res, num_case, 'storage_charging_efficiency')[case_idx]
    
    max_headroom, mean_residence_time, max_residence_time = battery_calculation(num_time_periods,
                                                                                dispatch_to_storage,
                                                                                dispatch_from_storage,
                                                                                energy_storage,
                                                                                storage_charging_efficiency)
    
    xaxis = np.arange(360)+1
    battery_TP(xaxis,mean_residence_time,max_residence_time,max_headroom)
    
    
        


#===============================================================================
#================================================== EXECUTION SECTION ==========
#===============================================================================

#------------- make change here
# set color 
color_natgas  = {0:"red",    1:"tomato"}
color_solar   = {0:"orange", 1:"wheat"}
color_wind    = {0:"blue",   1:"skyblue"}
color_nuclear = {0:"green",  1:"limegreen"}
color_storage = {0:"m",      1:"orchid"}

verbose = True    # actually no use here
file_path_stack = '/Users/leiduan/Desktop/File/phd/phd_7/CIS_work/Energy_optimize_model/WORK/Results' + \
                  '/Mengyao_data/one_year_simulations/'
            
file_path_contour = '/Users/leiduan/Desktop/File/phd/phd_7/CIS_work/Energy_optimize_model/WORK/Results' + \
                    '/Mengyao_data/one_year_simulations_contour/'
            
switch = 'battery'     # stack(x2) or contour(x1) or both(x3)
multipanel = True    # please set to True for now
scenario_name = 'no_ng_fixed_nuc.pickle'  # Which scenario do you want? set to scenario_name or all for all scenarios

# central case info.for time patch stack, only two constrains for now
# only one central case can be produced for each scenario
# if you need more constrains, make change in "stack_plot2"
select_case1 = ['fix_cost_nuclear', 'var_cost_nuclear']
select_case2 = [0.05 ,  0.001]





#########  make changes above, not below #########

if switch == 'stack' or switch == 'battery':
    file_path = file_path_stack
    file_list = os.listdir(file_path)
elif switch == 'contour':
    file_path = file_path_contour
    file_list = os.listdir(file_path)
else:
        print
        print 
        print 'Error: not support plot'
        sys.exit(0)
    
for file in file_list:
    
    case_name = file
    if scenario_name == 'all' or case_name == scenario_name:
        print 'deal with case:', case_name
        
        file_info, time_series, assumption_list, result_list = unpickle_raw_results(
                file_path + case_name,
                verbose
                )
        
        res = prepare_scalar_variables(
                file_info,
                time_series,
                assumption_list,
                result_list,
                verbose
                )
        
        num_case = len(res)
        num_time_periods = len(res[0]["demand"])
        
        if switch == 'stack':
            stack_plot1(res, num_case, case_name,multipanel)
            stack_plot2(res, num_case, case_name,multipanel, select_case1, select_case2)
        elif switch == 'contour':
            contour_plot(res,num_case, case_name)
        elif switch == 'battery':
            battery_plot(res,num_case,case_name, multipanel, select_case1, select_case2)
        elif switch == 'both':
            #stack_plot1(res, num_case, case_name,multipanel)
            stack_plot2(res, num_case, case_name,multipanel, select_case1, select_case2)
            contour_plot(res,num_case, case_name)
        else:
            print 'not supported plot'

