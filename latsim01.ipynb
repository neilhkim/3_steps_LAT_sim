#%%
# import multiprocessing
# import tqdm
# from pickle import FALSE, TRUE
# import matplotlib.pyplot as plt
from cmath import exp
import random
import numpy as np
# import scipy.stats as st
# import numba
# import biocircuits

# Plotting modules
import bokeh.io
import bokeh.plotting
bokeh.io.output_notebook()

from bokeh.models import Band, ColumnDataSource
import pandas as pd

# Line profiler (can install with conda install line_profiler)
# %load_ext line_profiler


def boundtime_dep_propensity(lifetimes, kbind, kunbind):
    # Three scenarios - 0) new pMHC binds, 1) bound pMHC unbinds, 2) bound pMHC forms LAT
    nBoundPMhc = lifetimes.size
    propensities = np.array([]) 
    propensities = np.append(propensities, kbind)  # Scenario 0)
    propensities = np.append(propensities, kunbind * nBoundPMhc) # Scenario 1)
    if lifetimes.size > 0: # Scenario 2)
        for t in np.nditer(lifetimes):
            kN = 0.1
            # N = 1
            kt = np.square(kN) * t * np.exp(-2*kN*t) # This is a test in the time being. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # kt = 0.1 * t
            propensities = np.append(propensities, kt)
    return propensities

# Specify parameters for calculation
time_points = np.linspace(0, 50, 101)
kN = 0.1
y = [np.square(kN) * t * np.exp(-2*kN*t) for t in time_points] 
plotkt = bokeh.plotting.figure(plot_width=300,plot_height=200,
                               x_axis_label='dwell time',y_axis_label='nucleation rate')
plotkt.line(time_points, y,        line_width=3, color='gray', line_join='bevel')
bokeh.io.show(plotkt)


kbind = 1
kunbind = 1 
args = (kbind, kunbind)

nCell = 100
# np.random.seed(42) # Seed random number generator for reproducibility

# Initialize output array
samples = np.empty((nCell, len(time_points), 2), dtype=int)

pMhcLifeLogs = np.array([])
# Run the calculations
for i in range(nCell): #tqdm.notebook.tqdm(range(size)):
    samples[i,:,:], temp = gillespie_ssa(boundtime_dep_propensity, time_points, args=args)
    pMhcLifeLogs = np.append(pMhcLifeLogs, temp)
    pMhcLifeLogs = np.reshape(pMhcLifeLogs, (-1,len(time_points)))

plot2 = bokeh.plotting.figure(plot_width=300,plot_height=200,
                               x_axis_label='dwell time',y_axis_label='number of pMHC', 
                               y_axis_type="log")
finalValidTimeIdx = np.argwhere(np.sum(~np.isnan(pMhcLifeLogs), axis=0)==0).min() 
y = pMhcLifeLogs[:,:finalValidTimeIdx]
x = time_points[:finalValidTimeIdx]
plot2.line(x, sum(~np.isnan(y)),        line_width=3, color='pink', line_join='bevel')
bokeh.io.show(plot2)

# Set up plots
countmsg = 'N = %d' % len(pMhcLifeLogs)
plots = [bokeh.plotting.figure(plot_width=300,plot_height=200,x_axis_label='time',
                                y_axis_label='number of bound pMHC', title=countmsg),
         bokeh.plotting.figure(plot_width=300,plot_height=200,x_axis_label='time',
                                y_axis_label='number of LAT condensates', title=countmsg)]
# Plot trajectories and mean
for i in [0, 1]: # 0 is nBoundPMhc and 1 is nLat
    for x in samples[:,:,i]: # x is each trial sequence
        plots[i].line(time_points, x, line_width=0.6,     alpha=0.6, line_join='bevel')
    plots[i].line(time_points, samples[:,:,i].mean(axis=0),    
                    line_width=3, color='orange', line_join='bevel')
# Link axes
plots[0].x_range = plots[1].x_range
bokeh.io.show(bokeh.layouts.gridplot(plots, ncols=2))

plot3 = bokeh.plotting.figure(plot_width=300,plot_height=200,
                                x_axis_label='dwell time',y_axis_label='prob(LAT)')
finalValidTimeIdx = np.argwhere(np.sum(~np.isnan(pMhcLifeLogs), axis=0)<1).min() 
yy = pMhcLifeLogs[:,:finalValidTimeIdx]
x = time_points[:finalValidTimeIdx]

for y in yy[:,:]: # pMHC, record | x: per-pMHC
    plot3.line(x, y, line_width=0.6,             alpha=0.6, line_join='bevel')

plot3.line(x, np.nanmean(yy[:,:], axis=0),       line_width=3, color='green', line_join='bevel')

# create the coordinates for the errorbars
sem = np.nanstd(yy[:,:], axis=0) / np.sqrt(sum(~np.isnan(yy)))
err_xs = []
err_ys = []

for x, y, yerr in zip(x, np.nanmean(yy[:,:], axis=0), sem):
    err_xs.append((x, x))
    err_ys.append((y - yerr, y + yerr))

# plot them
plot3.multi_line(err_xs, err_ys, color='green')
plot3.x_range.start = time_points[0]
plot3.x_range.end = time_points[-1]
bokeh.io.show(plot3)

plot1 = plot3
finalValidTimeIdx = np.argwhere(np.sum(~np.isnan(pMhcLifeLogs), axis=0)<10).min() 
plot1.x_range.start = time_points[0]
plot1.x_range.end = time_points[finalValidTimeIdx-2]
bokeh.io.show(plot1)



def common_member(a, b):   
    a_set = set(a)
    b_set = set(b)
     
    # check length
    if len(a_set.intersection(b_set)) > 0:
        return a_set.intersection(b_set)
    else:
        return []



def sample_discrete(probs):
    # Generate random number
    q = np.random.rand()
    # Find index
    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return i - 1


def gillespie_draw(propensity_func, lifetimes, args=()):
    # Compute propensities
    calc_propensities = propensity_func(lifetimes, *args)
    # Sum of propensities
    props_sum = calc_propensities.sum()
    # Compute next time
    time = np.random.exponential(1.0 / props_sum)
    # Compute discrete probabilities of each reaction
    rxn_probs = calc_propensities / props_sum
    # Draw reaction from this distribution
    rxn = sample_discrete(rxn_probs)
    return rxn, time

def gillespie_ssa(propensity_func, time_points, args=()):
    # Initialize output
    nSpecies = 2
    pop_out = np.empty((len(time_points), nSpecies), dtype=int)
    nBindUnbindEvents = 2

    # Initialize and perform simulation
    i_time = 1
    i = 0
    t = time_points[0]
    nTimepoints = time_points.size
    boundPMhcIndices = np.array([])
    lifeLog = np.array([])
    '''
    lifeLog[0].size == time_points.size 
    (Examples)
    lifeLog[i] (non-productive life) : [0, 0, 0, 0, 0, 0, np.nan, np.nan, ..., np.nan] : {bound, unbind}
    lifeLog[j] (producctive life)    : [0, 0, 0, 0, 1, 1,  np.nan, np.nan, ... np.nan] : {bound, produce, unbind}
    '''
    # pMhcLifeLogs = np.reshape(pMhcLifeLogs, (-1,len(time_points)))

    pMhcLifeTimes = np.array([])
    pMhcLifeTimes = np.reshape(pMhcLifeTimes, (-1,2))
    # unbindingTimes = np.array([])
    # productionTimes = np.array([])
    nBoundPMhc = 0
    nLat = 0
    pop_out[0,:] = np.array([0, 0])
    nPMhcTot = int(0)

    boundPMhcIndices = []
    productivePMhcIndices = []

    while i < len(time_points):
        while t < time_points[i_time]:

            population_previous = np.array([nBoundPMhc,nLat])

            # draw the event and time step
            event, dt = gillespie_draw(propensity_func, pMhcLifeTimes[:,1], args) # event number also indicates the index of the pMhc that results in LAT condensation.

            if not isinstance(event, int):
                print('event is not integer')
                quit()    

            # Process which event happened.
            if event == 0: # new binding
                pMhcLifeTimes = np.append(pMhcLifeTimes, [nPMhcTot, 0])
                pMhcLifeTimes = np.reshape(pMhcLifeTimes, (-1,2))
                entry = np.empty(nTimepoints)*np.nan
                entry[0] = 0
                lifeLog = np.append(lifeLog, entry)
                lifeLog = np.reshape(lifeLog, (-1,nTimepoints))
                boundPMhcIndices.append(int(nPMhcTot))
                nPMhcTot += 1

            elif event == 1: # unbinding
                random_index = random.randrange(len(boundPMhcIndices))
                poppedPMhcId = int(boundPMhcIndices.pop(random_index))
                # randIdex = random.randrange(pMhcLifeTimes.size)
                # unbindingTimes = np.append(unbindingTimes, pMhcLifeTimes[randIdex])
                i = np.argwhere(pMhcLifeTimes[:,0]==poppedPMhcId)
                if len(i) > 1:
                    print('popped more than 1?')
                    quit()   
                pMhcLifeTimes = np.delete(pMhcLifeTimes, i, 0)                    

            else: # LAT condensation
                nLat += 1
                # prod_index = event - 2
                # productivePMhcIndices.append(prod_index)
                prodIdxAmongLifeTimes = event - 2
                # lifetimesIndex = event - 2 # event number 2, 3, ... means LAT condensate producing pMHC's index is 0, 1, ...
                productivePMhcIndices.append(int(pMhcLifeTimes[prodIdxAmongLifeTimes,0]))
                # print(productivePMhcIndices)
                # productionTimes = np.append(productionTimes, pMhcLifeTimes[prodPMhcIdx])
                pMhcLifeTimes = np.delete(pMhcLifeTimes,prodIdxAmongLifeTimes, axis=0) 

            nBoundPMhc = len(boundPMhcIndices)

            # Increment time
            pMhcLifeTimes[:,1] += dt
            t += dt

        # Update the index
        i = np.searchsorted(time_points > t, True)
        
        # Update the population
        pop_out[i_time:min(i,len(time_points))] = population_previous
        bNotP_indices = [item for item in boundPMhcIndices if item not in productivePMhcIndices]
        bAndP_indices = [item for item in boundPMhcIndices if item in productivePMhcIndices]
        bNotP_indices = [int(x) for x in bNotP_indices]
        bAndP_indices = [int(x) for x in bAndP_indices]

        for x in bNotP_indices:
            earlistNanPosition = np.where(np.isnan(lifeLog[x]))[0][0]
            lifeLog[x,earlistNanPosition:earlistNanPosition+i-i_time] = int(0)

        for x in bAndP_indices:
            earlistNanPosition = np.where(np.isnan(lifeLog[x]))[0][0]
            lifeLog[x,earlistNanPosition:earlistNanPosition+i-i_time] = int(1)

        # Increment index
        i_time = i

    return pop_out, lifeLog
    
# %%
