import numpy as np 
import scipy.ndimage 
from termcolor import colored
import time

# def sub2ind(array_shape, rows, cols): # convert to linear indices 
    # return (rows*array_shape[1] + cols).astype(int)

def sub2ind(array_shape, rows, cols): # convert to linear indices 
    return (rows + cols * array_shape[0]).astype(int)    

def gaussian_filter(for_smooth,sigma=1):
    """
    Gaussian Filter 
    for_smooth - input to be smothed
    sigma - standard deviation for smoothing default = 1
    """
    return scipy.ndimage.gaussian_filter(for_smooth,sigma) #smoothed maps 

def filter_maps(maps,sigma=1,method = 'gaussian',show_run = False):
    """
    Function returns gaussian filtered maps 
    """
    if method != 'gaussian':
        raise ValueError(colored("Method must be gaussian","blue"))
    filter_start = time.time()
    smoothed_maps = np.zeros(np.shape(maps))
    invalid_idx = np.isnan(maps[:,:,0])
    for i in range(len(maps[0,0,:])):
        m = maps[:,:,i]
        m[invalid_idx] = 0
        m = gaussian_filter(m,sigma)
        m[invalid_idx] = np.nan
        smoothed_maps[:,:,i] = m
    filter_end = time.time()
    if show_run:
        print(colored(f"Filtering rate maps... \n","yellow"))
        print(colored(f"Run time to filter maps: {round(filter_end-filter_start,3)} seconds... \n","yellow"))
    return smoothed_maps

def linear_position(position,env_size,bins):
    scale = env_size/bins # for dividing position data to discretize
    discrete_position = np.floor(position/scale)+1 # discretize to == bins 
    discrete_position = np.clip(discrete_position, 1, bins).astype(int) # handles all edge cases
    array_shape = [bins[0],bins[1]] 
    rows = discrete_position[0,:]-1
    cols = discrete_position[1,:]-1
    return sub2ind(array_shape, rows, cols) # Extract discrete position as linear indices 

def column_split(input,split = 2):
    """
    This function splits array values columnwise into default 2: 
    Split - number of divisions 
    Removes end points to ensure equal split 
    """
    if len(np.shape(input))==1:
        input = input.reshape(1,-1)

    D, T = input.shape
    trimmed_T = T - (T % split)
    input_trimmed = input[:, :trimmed_T]
    chunks = np.array_split(input_trimmed, split, axis=1)
    return np.stack(chunks, axis=-1)

def ensure_bin_env(bins,env_size):
    """
    Ensures Bins and Env_size are proper dims/format 
    """
    # Handle cases where env_size and bins are not 1 or 2 scalar values 
    if np.isscalar(bins):
        bins = [bins] # handle cases where bins is a scalar input 
    if np.ndim(bins) > 1:
        raise ValueError(colored("Bin array must be 1-dimensional... \n","yellow"))
    else:
            if np.shape(bins)[0] == 1:
                bins = (np.zeros((2,1)) + bins).astype(int)
            elif np.shape(bins)[0] == 2: 
                bins = bins.reshape((2,1)).astype(int) # always ensure column vector 
            else: 
                raise ValueError(colored("Bins must be <= 2 values... \n","yellow"))
    # env_size = self.global_env_size
    if np.isscalar(env_size):
        env_size = [env_size] # handle cases where bins is a scalar input 
    if np.ndim(env_size) > 1:
        raise ValueError(colored("Environment size array must be 1-dimensional... \n","yellow"))
    else:
            if np.shape(env_size)[0] == 1:
                env_size = (np.zeros((2,1)) + env_size).astype(int) # initialize to column vector 
            elif np.shape(env_size)[0] == 2: 
                env_size = env_size.reshape((2,1)).astype(int) # always ensure column vector 
            else: 
                raise ValueError(colored("Environment Size must be <= 2 values... \n","yellow"))

    return bins, env_size

def binarize_mat(arr):
    """
    This function converts all non-zero indices in a numpy array to 1 
    Returns boolean array - less memory requirements 
    """
    idx = arr != 0
    arr[idx] = 1
    return arr.astype(bool)

def grab_split_half(t,lin_split,bins):
    """
    This functions computes un-smoothed split halfs given inputs:
    t - trace, 2 dimensional array shape(cells x time)
    lin_split, 1 dimensinoal array shape( x time)
    """
    bins, _ = ensure_bin_env(bins,1)
    states = np.arange(np.prod(np.array(bins).astype(int)))
    num_neurons = np.shape(t)[0]
    maps = np.zeros((np.prod(bins).astype(int),num_neurons*2)) # init empty maps 
    t_split = column_split(t)
    for s in states:
        maps[s,:np.shape(t)[0]] = np.mean(t_split[:,(lin_split[:,:,0]==s).flatten(),0],axis=1)
        maps[s,np.shape(t)[0]:] = np.mean(t_split[:,(lin_split[:,:,1]==s).flatten(),1],axis=1)
    idx = np.any(np.isnan(maps),1) # Inlcude only if pixels have sampling in both halves, divide by zero gens nans
    corr_mat = np.corrcoef(maps[~idx,:],rowvar=False) # Correlation Matrix, excluding unsampled pixels 

    is_match = ((num_neurons*2)*np.arange(1,num_neurons))+num_neurons+np.arange(2,num_neurons+1)
    is_match = np.hstack((num_neurons+1,is_match))-1
    return corr_mat.T.ravel()[is_match]

def norm_arr(arr):
    return arr/np.sum(arr)

def get_mfr(t):
    return np.mean(t,1)

def get_linear_velocity(p,fps=30,tau=1):
    """
    Compute Linear Velocity in cm/s 
    p is position array [2 x time]
    Fps is 30 - our sampling rate 
    Tau - 1 frame look ahead 
    """
    tm1 = p
    tm2 = np.roll(tm1,-tau)[:,:-tau] # Duplicate array - shifted by tau (delete wrap around)
    tm1 = tm1[:,:-tau] # Delete wrap around 
    dis =  np.sqrt((tm2[0,:] - tm1[0,:])**2 + (tm2[1,:] - tm1[1,:])**2)
    vel = dis*fps
    return vel, dis 
    

def get_rotational_velocity(p,fps=30,tau=1):
    """
    Computes rotational velocity in cm/s 
    p is position array [2 x time]
    Fps is 30 - our sampling rate 
    Tau - 1 frame look ahead 
    """
    tm1 = p
    tm2 = np.roll(tm1,-tau)[:,:-tau] # Duplicate array - shifted by tau (delete wrap around)
    tm1 = tm1[:,:-tau] # Delete wrap around 
    x = tm2[0,:] - tm1[0,:] 
    y = tm2[1,:] - tm1[1,:]
    w = np.arctan2(y,x) # Compute change in heading angle 
    return np.diff(w)*fps 
    