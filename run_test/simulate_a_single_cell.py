
import sys
  
# setting path
sys.path.append('../')
  
# importing
from src.main import * 

# Simulating just a single T-cell with default values + plot_time_traces option (example):
run(    
    plot_time_traces = True,     
    )
