
import sys
  
# setting path
sys.path.append('../')
  
# importing
from src.main import * 

# Simulating a response curve over ligand densities using speed enhancing heuristics (example):
run(
    kunbind_list = [1, .1], # The known kunbind of self ligand ~1/s and foreign ligand ~0.1/s  
    ligand_densities = [.001, .01, .1, 1, 10, 30, 100], # unit: molecules per um2
    lat_formation_propensity_fn_str = '0.3 * stats.gamma.pdf(t-3, a=2, scale=14)',
    huge_nlat_earlyfinish_zeroifnot_used = 10,
    plot_activation_titration_curve = True,
    early_assume_100p = True,
    )

# Showing all the options you can use:
# run(
#     kunbind_list = [10, 1, .1, .01, .001], # The known kunbind of MCC is 0.042/s and T102S is 0.11/s
#     ligand_densities = [.001, .01, .1, 1, 10, 30, 100], # unit: molecules per um2
#     n_experiment_per_ligdensity = [3,3,3,3,2,2,2],
#     lat_formation_propensity_fn_str = '0.3 * stats.gamma.pdf(t-3, a=2, scale=14)',
#     plot_lat_propensity = True,
#     n_cell = 10, 
#     endtime = 120, 
#     n_timepoints = 100,
#     ec50_cellactivating_latnumber = 3,
#     huge_nlat_earlyfinish_zeroifnot_used = 10,
#     plot_time_traces = True,
#     plot_activation_titration_curve = True,
#     display_progressbar = True, # works on online notebooks
#     outfname_ifnotnull_write = '/content/drive/My Drive/Colab/1s_const',
#     early_assume_100p = True,
#     plat_threshold_zeroifnot_used = 0,
#     )
