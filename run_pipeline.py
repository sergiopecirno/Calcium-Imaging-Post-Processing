import numpy as np
import numpy.matlib 
import eval_dir 
import os
import h5py
import matplotlib.pyplot as plt
import sys 
from Calcium import Calcium
from termcolor import colored
import time 
from scipy.ndimage import gaussian_filter
import warnings
import Utils
import random 
from tqdm import tqdm

from multiprocessing import Pool, cpu_count # push cpu 

warnings.filterwarnings("ignore", category=RuntimeWarning)
# Extract file paths to matlab files 
base = os.path.expanduser("~")
home_dir = os.path.join(base,'Keinath Lab Dropbox')
is_mat = 'MatlabData' 
mat_data_path = eval_dir.find_folder_os(home_dir,is_mat)[0]
# destination = 'OpenField' # Which project 
destination = 'OpenField' # Which project 
do_mouse = 'SAP174221' # Which Mouse 
# do_mouse = 'CA1174216'
mat_data_path = os.path.join(mat_data_path,destination)
mouse_path = eval_dir.find_folder_os(mat_data_path,do_mouse)[0] # Path to this mouses - raw data 

p,p_ends = eval_dir.get_file_paths(mouse_path,'.mat','name') # Extract file paths - as well as file ends, sort option = name * date of recording

# Create Folder for this mouse # Will return if already exists 
processed_folder = eval_dir.mk_dir('processed',location = 'current') 

save_to = os.path.join(processed_folder,do_mouse,destination)
save_to = eval_dir.mk_dir(save_to) # Creates folder with mouse_ID "do_mouse"

i = 0 # n-th day to analyze 
print(colored(f"Running Calcium Imaging Analysis On:  {p[i]}... \n", "green"))
data = Calcium(p[i])
data.get_data_from_mat() # Extract essential data from matlab file - needed to get folder label
session_folder = eval_dir.mk_dir(data.recording_date,save_to) # Creates folder with recording date and time in 'processed'
data.get_save_folder(session_folder) # run this to innit save folder 
data.get_figure_folder

if data.check_exist():
    print(colored(f"<<<<< DATA HAS ALREADY BEEN ANALYZED: {p[i]} SKIPPED >>>>> \n","red"))
    print(colored(f"LOADING DATA... \n","red"))
    pkl_path = data.data_save_file
    data.load_object() # re-writes current data file 
    print(colored(f"KEYS EXIST IN: {pkl_path}... \n","red"))
    for key, value in data.__dict__.items():
        print(colored(f"{key}","red"))
    print('\n')
else: 
    # Binarize trace - sometimes we want to do this - but beware of losing essential information 
    # Converts all non-zero values to 1 
    # data.trace = Utils.binarize_mat(data.trace)
    # Environment specs 
    env_size = np.array([np.max(data.position[0,:]).astype(int),np.max(data.position[1,:]).astype(int)])
    bins = 15 
    smoothing = 1
    data.set_params(bins,env_size,smoothing)
    # Path and Occupancy
    data.save_path(fig_extend = '.pdf') # export path 
    data.grab_occupancy(export_fig = True) # compute discrete state occupancy 
    data.get_sample_statistics(export_figs = True) # Computes average linear and rotational velocity, as well as total amount of sampling 

    ### Rate Maps ### 
    # Whole recording rate maps 
    data.get_maps() # This functions uses matrix manipulation and iteration through linear index states 
    # data.get_maps2() # This function computes maps by iterating through pairs of states 
    data.get_split_maps() # Split Half Rate Maps 
    # # Split map correlations
    data.split_map_corr(export_fig=True) # This function computes correlations using matrix manip and indexing 
    # # Export a spike map overlayed on path figure with corresponding rate map 
    # # data.high_corr_example(export_fig=True,fig_extend='.pdf',view_smoothed = False)

    ### Permute for % Place Cells ### 
    data.permute_p(export_fig1=True,export_fig2=True,override=False)

    ### Naive spatial information ### 
    data.get_spatial_information()

    ### Compute Raw Transition Counts ### 
    data.get_state_transitions()
    ### Compute Successor Representation ### 
    data.analytical_successor_representation()

    # data.save_object()