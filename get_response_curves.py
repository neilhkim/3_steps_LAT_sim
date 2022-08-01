# Draw response curves
run(
      ligand_densities = [0.01, .03, .1, .3, .7, 1], 
      kunbind_list = [0.11], #[0.011, 0.042, 0.11], 
      nCell=30, 
      n_P_act_estimation=3,
    #   nLATthreshold = 3, 
      endtime=200, 
      plotResponseCurve=True, 
      probabilistic_activation=True,
      early_termination_option=False,
    )
