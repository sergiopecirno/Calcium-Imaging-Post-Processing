import numpy as np 
import eval_dir
import h5py
from termcolor import colored
from scipy.ndimage import gaussian_filter
import time
import Utils # related general functions / tools
import scipy
import matplotlib.pyplot as plt 
import os 
import random
from tqdm import tqdm 
import numpy.matlib
import pickle

class Calcium:
    """
    This class contains the primary calcium imaging analysis 
        init
            file_path - path to .mat file 
            save_folder - location for data storage 
            save_file - pkl file to store data (run after save_folder)
            figure_folder - child of save_folder for figures
            position - [2 x time] corrdinates of position at each time step 
            lin_pos - position vectorized
            trace - [num_neurons x time] trace vectors for each cell 
            sampling - occupancy matrix
            num_neurons - number of neurons recorded
            recording_date - date of recording 
            env_size - vector of environment size type:int 
            bins - number of discrete states 
            smoothing_std - standard deviation for rate map smoothing 
            linear_velocity - linear velocity at each time step 
            mean_linear_velocity - Average linear velocity 
            median_linear_velocity - Median linear velocity 
            stdev_linear_velocity - standard deviation linear velocity 
            rotational_velocity - rotational velocity at each time step 
            mean_rotational_velocity - Average Rotational Velocity 
            stdev_rotational_velocity - Standard deviation rotational velocity 
            distance_sampled - Amount of sampling 
            distance_per_minute - Sampling rate per minute (redundant with linear velocity)
            unsmoothed_maps - rate maps no smoothing 
            smoothed_maps - rate maps with smoothing, "gaussian" 
            first_half_maps - rate maps from first half 
            second_half_maps - rate maps from second half 
            save_folder - path where figs exported to
            split_half - split half correlations for each cell 
            p_vals - p_value for each cells split-half ()
            is_cell - which cells to include in analysis 
            percent_stable - % of place cells with stable rate maps 
            self.raw_transitions - state transition counts 
            self.transition_probability - state transition normalized
            self.M - succesor representation from self.transition_probability  
    """
    def __init__(self,file_path):
        self.file_path = file_path
        self.save_folder = None
        self.data_save_file = None
        self.figure_folder = None
        self.position = None
        self.lin_pos =  None 
        self.trace = None 
        self.sampling = None
        self.num_neurons = None
        self.recording_date = None
        self.env_size = None
        self.global_env_size = None
        self.bins = None
        self.smoothing_std = None
        self.linear_velocity = None 
        self.mean_linear_velocity = None
        self.median_linear_velocity = None
        self.stdev_linear_velocity = None
        self.rotational_velocity = None
        self.mean_rotational_velocity = None
        self.stdev_rotational_velocity = None
        self.distance_sampled = None
        self.distance_per_minute = None 
        self.unsmoothed_maps = None
        self.smoothed_maps = None
        self.first_half_maps = None
        self.second_half_maps = None
        self.save_folder = None
        self.split_half = None
        self.p_vals = None 
        self.is_cell = None
        self.percent_stable = None
        self.spatial_information = None
        self.raw_transitions = None 
        self.transition_probability = None
        self.M = None 


        
    def get_data_from_mat(self):
        # Extract data 
        print(colored(f"Reading MATLAB calcium imaging data file... \n","yellow"))
        for_get = 'processed'
        file = h5py.File(self.file_path,'r') # loads matlab file 
        file_keys = list(file.keys())
        print(colored(f"Available keys in LEVEL:HIGH calcium imaging data file include: {file_keys} ...\n","yellow"))    
        self.num_neurons = int(file['calcium']['numNeurons'][:]) # extract number of neurons

        self.recording_date = file['properties']['trial'][:] # Extract date and time of recording 
        self.recording_date = self.recording_date.squeeze() # 1 Dimensional Array 
        self.recording_date = self.recording_date.astype(np.uint16).tobytes() # Convert from u2 to interpretable txt 
        self.recording_date = self.recording_date .decode('utf-16-le') 
        is_slash = self.recording_date.index('/')
        self.recording_date = list(self.recording_date)
        self.recording_date[is_slash] = '-'
        self.recording_date = ''.join(self.recording_date)

        print(colored(f"Recording date and time EXTRACTED and wrote to self.recording_date ... \n","yellow"))

        self.global_env_size = file['environment']['size'][:].flatten().astype(int)
        # print(self.env_size)
        processed = file[for_get] # extracts processed struct 
        processed_keys = list(processed.keys()) # structs that exist in MATLAB file 
        print(colored(f"Available keys in LEVEL:Processed include: {processed_keys} ... \n","yellow")) # list keys 
        pos_key = 'p' # position key 
        trace_key = 'trace' # trace key 
        if pos_key in processed_keys and trace_key in processed_keys:
            self.position = processed['p'][:] # position data 
            self.trace = processed['trace'][:] # trace data 
            ## Transpose for computes 
            self.position = self.position.T 
            self.trace = self.trace.T
            print(colored(f"Position data written to self.position... \n Trace data written to self.trace... \n ","yellow"))
        else:
            raise KeyError("Position and/or Trace data does not exist ")

    def get_save_folder(self,path):
        """
        This create a folder for the days recording
        """
        self.save_folder = path
        self.data_save_file = os.path.join(self.save_folder,'data.pkl') # file name for saving 

    def get_figure_folder(self):
        """
        This create a folder for the days recording
        """
        if self.save_folder is None:
            raise ValueError('SAVE FOLDER DOES NOT EXIST: NEED TO CALL get_save_folder()...')
        fig_folder = os.path.join(self.save_folder,'fig')
        self.figure_folder = eval_dir.mk_dir(fig_folder)
        
    def check_exist(self):
        """
        This method checks if the data has already written to a file 
        """
        if self.save_folder is None:
            raise ValueError('BLAH')
        filename = os.path.join(self.save_folder,'data.pkl')
        return os.path.exists(filename) 
        

    def set_params(self,bins,env_size,smoothing):
        self.bins,self.env_size = Utils.ensure_bin_env(bins,env_size) # set bins and environment size parameters 
        self.smoothing_std = smoothing # standard deviation for rate map smoothing 

    def save_path(self,fig_extend = '.pdf'):
        """
        Write path to pdf figure 
        env_size - size of environment 
        fig_extend - default as .pdf (vectorized for illustrator)
        """
        if self.env_size is None:
            raise ValueError("Self.env_size doesn't exist... Need to call self.set_params first... \n")
        if self.figure_folder is None:
            raise ValueError('SAVE FOLDER DOES NOT EXIST: NEED TO CALL get_save_folder()...')
        if not os.path.exists(os.path.join(self.figure_folder,('path'+fig_extend))):
            print(colored(f"Plotting Path... \n","yellow")) # list keys 
            plt.plot(self.position[0,:],self.position[1,:],color = 'black')
            plt.axis([0, self.env_size[0], 1, self.env_size[1]])  
            plt.ylabel('Y-Cord')
            plt.xlabel('X-Cord')
            plt.title('')
            if self.figure_folder is None:
                raise ValueError("SAVE FOLDER DOES NOT EXIST... Need to call self.get_figure_folder first... \n")   
            where = os.path.join(self.figure_folder,('path'+fig_extend))
            plt.axis('equal')
            plt.savefig(where)
            plt.close()
            print(colored(f"Path saved to: {where} ... \n","cyan")) 
        else: 
            print(colored(f"<<<<<<<<< SAVE PATH SKIPPED >>>>>>>>> \n Already saved to: {os.path.join(self.figure_folder,('path'+fig_extend))}... \n","cyan"))

    def get_sample_statistics(self,fig_extend = '.pdf',export_figs = False):
        """
        This method computes sample statistics for animal behavior. Computes linear and rotational velocity as well as total amount of sampling. 
        fig_extend - file extension for figure
        export_figs - if true writes sample statistics to figure 
        """
        if self.position is None:
            ValueError("self.position doesn't exist... Need to call Calcium.get_data_from_mat first... \n") 
        print(colored("Computing Sampling Statistics... \n","yellow"))
        self.rotational_velocity = Utils.get_rotational_velocity(self.position) # Computes rotational velocity 
        self.mean_rotational_velocity = np.nanmean(self.rotational_velocity) # Mean rotational 
        self.stdev_rotational_velocity = np.nanstd(self.rotational_velocity) # Stdev rotational 
        self.linear_velocity, distance = Utils.get_linear_velocity(self.position) # Computes rotational velocity and distance stepped in each frame 
        self.mean_linear_velocity = np.nanmean(self.linear_velocity) # Mean Linear velocity 
        self.median_linear_velocity = np.nanmedian(self.linear_velocity)
        self.stdev_linear_velocity = np.nanstd(self.linear_velocity) # Standard deviation linear 
        self.distance_sampled = np.nansum(distance) # total distance sampled during recording 
        self.distance_per_minute = self.distance_sampled / (len(distance) / 30 / 60) # 30 is our sampling rate 
        print(colored(f"Sample Statistics Computed...\n Linear Velocity Mean: {round(self.mean_linear_velocity,3)}, Standard Deviation: {round(self.stdev_linear_velocity)}... \n Rotational Velocity Mean: {round(self.mean_rotational_velocity,3)}, Standard Deviation: {round(self.stdev_rotational_velocity)}... \n","yellow"))
        print(colored(f"Sample Statistics Written to self.rotational_velocity, self.mean_rotational_velocity, self.stdev_rotational_velocity, self.linear_velocity, self.mean_linear_velocity, self.stdev_linear_velocity, self.distance_sampled,self.distance_per_minute \n","yellow"))
        if export_figs:   
            if self.figure_folder is None:
                raise ValueError("SAVE FOLDER DOES NOT EXIST... Need to call self.get_figure_folder first... \n")
            else: 
                print(colored(f"Generating Rotational Velocity Histogram... \n","yellow"))
                where = os.path.join(self.figure_folder,('rotational_velocity'+fig_extend)) # saves in this mouse/date recording folder 
                plt.hist(self.rotational_velocity,bins=100,color=[0,0,1])
                plt.title(f"Rotational Velocity (cm/second), Mean: {round(self.mean_rotational_velocity,3)}, Stdev: {round(self.stdev_rotational_velocity,3)}")
                plt.ylabel('Counts')
                plt.xlabel('cm/second')
                plt.savefig(where)
                plt.close()
                print(colored(f"Rotational Velocity Histogram saved to: {where} ... \n","cyan"))
                print(colored(f"Generating Linear Velocity Histogram... \n","yellow"))
                where = os.path.join(self.figure_folder,('linear_velocity'+fig_extend)) # saves in this mouse/date recording folder 
                plt.hist(self.linear_velocity,bins=100,color=[0,0,1])
                plt.title(f"Linear Velocity (cm/second), Mean: {round(self.mean_linear_velocity,3)}, Stdev: {round(self.stdev_linear_velocity,3)}")
                plt.ylabel('Counts')
                plt.xlabel('cm/second')
                plt.savefig(where)
                plt.close()
                print(colored(f"Linear Velocity Histogram saved to: {where} ... \n","cyan")) 
                
    def grab_occupancy(self,export_fig=True,fig_extend=".pdf"):
        """
        This method computes pixelwise sampling - this is already integrated into the rate map method
        But if you need a sampling matrix this will compute and generate an export.pdf file 
        """
        if self.bins is None:
            raise ValueError("Self.bins doesn't exist... Need to call self.set_params first... \n")
        if self.env_size is None:
            raise ValueError("Self.env_size doesn't exist... Need to call self.set_params first... \n")
        if self.position is None:
            ValueError("self.position doesn't exist... Need to call Calcium.get_data_from_mat first... \n") 
        print(colored(f"Generating sampling matrix... \n","yellow"))
        linear = Utils.linear_position(self.position,self.env_size,self.bins) # returns discrete lienar position
        # self.sampling = np.array([int((linear==i).sum(0)) for i in range(min(linear),max(linear)+1)]).reshape((self.bins.flatten()[0],self.bins.flatten()[1])) # compute pixel wise sampling and reshape 
        self.sampling = np.array([(linear == i).sum() for i in range(np.prod(self.bins))]).reshape(self.bins.flatten()[0], self.bins.flatten()[1])
        plt.imshow(self.sampling,origin='lower') # transpose because python linear indexing moves columnwise first (MATLAB is row wise)
        plt.axis('equal')
        plt.axis('off')
        print(colored(f"Sampling matrix generated... Written to self.sampling... \n","yellow"))
        if export_fig:
            if self.figure_folder is None:
                raise ValueError("SAVE FOLDER DOES NOT EXIST... Need to call self.get_figure_folder first... \n")
            else:
                where = os.path.join(self.figure_folder,('occupancy_matrix'+fig_extend)) # saves in this mouse/date recording folder 
                plt.savefig(where)
                print(colored(f"Occupancy matrix saved to: {where} ... \n","cyan")) 
        plt.close()
        
    def get_maps(self):
        """ 
        This function is one method for generating place cell rate maps. 
        env_size - size of evironment as 1 dimensional array (1 or 2 values) 
        bins - number of discrete bins for rate maps as 1 dimensional array (1 or 2 values)
        smoothing - standard deviation for gaussian filter  
        """
        if self.bins is None:
            raise ValueError("Self.bins doesn't exist... Need to call self.set_params first... \n")
        if self.env_size is None:
            raise ValueError("Self.env_size doesn't exist... Need to call self.set_params first... \n")
        if self.smoothing_std is None:
            raise ValueError("Self.smoothing_std doesn't exist... Need to call self.set_params first... \n")
        print(colored(f"Generating rate maps... \n","yellow")) # print statement 
        linear = Utils.linear_position(self.position,self.env_size,self.bins) # returns discrete lienar position  
        self.unsmoothed_maps = np.zeros((self.num_neurons,self.bins.flatten()[0]*self.bins.flatten()[1]))
        start = time.time() # time to compute - end 
        for s in range(np.size(self.unsmoothed_maps[0,:])): # iterate across pixels to compute MFR 
            self.unsmoothed_maps[:,s] = np.mean(self.trace[:,linear == s],axis=1) # mean columnwise 
        self.unsmoothed_maps  = np.reshape(self.unsmoothed_maps.T,(self.bins.flatten()[0],self.bins.flatten()[1],self.num_neurons)) # must transpose for proper indexing 
        self.smoothed_maps = Utils.filter_maps(self.unsmoothed_maps.copy(),self.smoothing_std,method='gaussian',show_run = True) # apply gaussian filter. In Utils 
        end = time.time() # time to compute - end 
        print(colored(f"Rate maps generated, INIT TO self.unsmoothed_maps / self.smoothed_maps ... \n","yellow")) # print statement 
        print(colored(f"Time to generate {self.num_neurons} rate maps, with bin size = ({self.bins.flatten()[0]} x {self.bins.flatten()[1]}): {round(end-start,3)} seconds... \n","yellow")) # print statement 

    def get_maps2(self):
        """ 
        This function is one method for generating place cells - the primary difference between 
        get_maps1 and 2 are iteration method. get_maps1 is significantly faster. Run either and compare output 
        """
        if self.bins is None:
            raise ValueError("Self.bins doesn't exist... Need to call self.set_params first... \n")
        if self.env_size is None:
            raise ValueError("Self.env_size doesn't exist... Need to call self.set_params first... \n")
        if self.smoothing_std is None:
            raise ValueError("Self.smoothing_std doesn't exist... Need to call self.set_params first... \n")
        scale = self.env_size/self.bins # for dividing position data to discretize
        discrete_position = np.floor(self.position/scale)+1 # discretize to == bins 
        discrete_position = np.clip(discrete_position, 1, self.bins).astype(int) # handles all edge cases
        # Create combinations of all states to loop through
        state_combs1 = np.matlib.repmat(np.arange(1,self.bins.flatten()[0]+1),1,self.bins.flatten()[1]) # x-states 
        state_combs2 = np.repeat(np.arange(1,self.bins.flatten()[1]+1),self.bins.flatten()[0]) # y - states 
        state_combs = np.concatenate((
            state_combs1.reshape(-1, 1),
            state_combs2.reshape(-1, 1)
        ), axis=1).T # Concatenate combinations  

        # Initilize empty rate maps array 
        self.unsmoothed_maps = np.zeros((self.bins.flatten()[0],self.bins.flatten()[1],self.num_neurons))
        # print(np.shape(state_combs))
        pc_start_time = time.time()
        for s in range(len(state_combs[0,:])-1):  
            # Loop through states and compute averate activity of each bin 
            tfirst_half_maps = discrete_position[0,:] == state_combs[0,s] # check x bin 
            tsecond_half_maps = discrete_position[1,:] == state_combs[1,s] # check y bin
            is_in = np.concatenate((tfirst_half_maps.reshape(-1,1),tsecond_half_maps.reshape(-1,1)),axis=1)
            is_in = np.all(is_in,axis=1) # occupancy index 
            get_av = self.trace[:,is_in] 
            get_av = np.nanmean(self.trace[:,is_in],axis=1)
            self.unsmoothed_maps[state_combs[0,s]-1,state_combs[1,s]-1,:] = get_av
        self.smoothed_maps = Utils.filter_maps(self.unsmoothed_maps.copy(),self.smoothing_std,method='gaussian',show_run = True) # apply gaussian filter. In Utils 
        pc_end_time = time.time()
        print(colored(f"Rate maps generated, INIT TO self.unsmoothed_maps / self.smoothed_maps ... \n","yellow")) # print statement 
        print(colored(f"Time to generate {self.num_neurons} rate maps, with bin size = ({self.bins.flatten()[0]} x {self.bins.flatten()[1]}): {round(pc_end_time-pc_start_time,2)} seconds... \n","yellow")) # print statement 

    def get_split_maps(self):
        """
        This method returns split maps - first half and second half rate maps 
        """
        if self.bins is None:
            raise ValueError("Self.bins doesn't exist... Need to call self.set_params first... \n")
        if self.env_size is None:
            raise ValueError("Self.env_size doesn't exist... Need to call self.set_params first... \n")
        if self.position is None:
            ValueError("self.position doesn't exist... Need to call Calcium.get_data_from_mat first... \n") 
        start = time.time()
        print(colored(f"Generating split-maps... \n","yellow"))
        t = Utils.column_split(self.trace) # Default split trace in half 
        p = Utils.column_split(self.position) # Default split position in half 
        lin = Utils.linear_position(p[:,:,0],self.env_size,self.bins) # returns discrete linear position  for first half 
        lin1 = Utils.linear_position(p[:,:,1],self.env_size,self.bins) # for second half 
        self.first_half_maps = np.zeros((self.num_neurons,self.bins.flatten()[0]*self.bins.flatten()[1])) # first half init 
        self.second_half_maps = np.zeros(np.shape(self.first_half_maps)) # second half init
        for s in range(np.size(self.first_half_maps[0,:])):
            self.first_half_maps[:,s] = np.mean(t[:,lin == s,0],axis=1) # mean columnwise of first half 
            self.second_half_maps[:,s] = np.mean(t[:,lin1 == s,1],axis=1) # mean columnwise of second half 
        self.first_half_maps  = np.reshape(self.first_half_maps.T,(self.bins.flatten()[0],self.bins.flatten()[1],self.num_neurons)) # must transpose for proper indexing 
        self.first_half_maps_smoothed = Utils.filter_maps(self.first_half_maps.copy(),self.smoothing_std,method='gaussian') # smooth first half maps 
        self.second_half_maps  = np.reshape(self.second_half_maps.T,(self.bins.flatten()[0],self.bins.flatten()[1],self.num_neurons)) # must transpose for proper indexing 
        self.second_half_maps_smoothed = Utils.filter_maps(self.second_half_maps.copy(),self.smoothing_std,method='gaussian') # smooth second half maps 
        end = time.time()
        print(colored(f"Split maps generated, INIT TO self.first_half_maps /self.first_half_maps_smoothed AND self.second_half_maps / self.second_half_maps_smoothed... \n","yellow")) 
        print(colored(f"Time to generate {self.num_neurons*2} maps, with bin size = ({self.bins.flatten()[0]} x {self.bins.flatten()[1]}): {round(end-start,3)} seconds... \n","yellow"))         

    def split_map_corr(self,export_fig = False,fig_extend ='.pdf'):
        """
        COMPARE TIME w/ SPLIT_MAP_CORR2
        Function to compute split half correlation on cell maps from first half and second half 
        This method uses matrix manipulation - concatenate correlations as matrix then indexing only self-cell-pair corerlations using linear indexing 
        Default for export figure is false - true if want to save / overwrite current figure 
        Default for fig_extend is to save as .pdf (vectorized for illustrator) 
        """
        if self.first_half_maps_smoothed is None or self.second_half_maps_smoothed is None:
            raise ValueError("SPLIT MAPS NEVER CREATED: PLEASE RUN self.get_split_maps(...) FIRST... \n")
        else:
            m1 = self.first_half_maps_smoothed # extract smoothed first half maps 
            m2 = self.second_half_maps_smoothed # extract smoothed second half maps
        print(colored(f"Computing split-map Correlations... \n","yellow"))
        start = time.time()
        inv_idx = np.any(np.isnan(np.stack((m1.copy()[:,:,0].flatten(),m2.copy()[:,:,0].flatten()))),axis=0)
        l1 = m1.flatten()
        tmp1 = l1.reshape((len(inv_idx),self.num_neurons)).T
        l2 = m2.flatten()
        tmp2 = l2.reshape((len(inv_idx),self.num_neurons)).T
        catted = np.vstack((tmp1[:,~inv_idx],tmp2[:,~inv_idx]))
        corr_mat = np.corrcoef(catted)
        idx = ((self.num_neurons*2)*np.arange(1,self.num_neurons))+self.num_neurons + np.arange(2,self.num_neurons+1)
        idx = np.hstack((self.num_neurons+1,idx))-1
        self.split_half = corr_mat.T.ravel()[idx]
        end = time.time()
        print(colored(f"Split map correlations, INIT TO self.split_half... \n","yellow")) 
        print(colored(f"Time to generate correlations: {round(end-start,3)} seconds... \n","yellow"))         
        colors = ["cyan"]
        bp = plt.boxplot(self.split_half,patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylim([-1, 1])
        plt.title(f"Split Map Correlations, n: {self.num_neurons}, mean: {round(np.mean(self.split_half),3)}",fontweight='bold')
        if export_fig:
            if not self.figure_folder:
                raise ValueError("SAVE FOLDER DOES NOT EXIST... \n")
            else:
                where = os.path.join(self.figure_folder,('split_half'+fig_extend)) # saves in this mouse/date recording folder 
                plt.savefig(where)
                print(colored(f"Correlations saved to: {where} ... \n","cyan")) 
        plt.close()

    def split_map_corr2(self,export_fig = False,fig_extend ='.pdf'):
        """
        Function to compute split half correlation on cell maps from first half and second half 
        This method uses an iterative version - iterating through each cells f1/f2 rate maps to compute split half correlations
        Default for export figure is false - true if want to save / overwrite current figure 
        Default for fig_extend is to save as .pdf (vectorized for illustrator) 
        """
        if self.first_half_maps_smoothed is None or self.second_half_maps_smoothed is None:
            raise ValueError("SPLIT MAPS NEVER CREATED: PLEASE RUN self.get_split_maps(...) FIRST... \n")
        else:
            m1 = self.first_half_maps_smoothed # extract smoothed first half maps 
            m2 = self.second_half_maps_smoothed # extract smoothed second half maps 
        print(colored(f"Computing split-map Correlations... \n","yellow"))
        start = time.time()
        inv_idx = np.any(np.isnan(np.stack((m1.copy()[:,:,0].flatten(),m2.copy()[:,:,0].flatten()))),axis=0) # index unvisited map states, where NaN
        self.split_half = np.zeros((self.num_neurons,1)) # init empty split half 
        for neur in range(self.num_neurons): # iterate across all neurons 
            tm1 = m1[:,:,neur].flatten()
            tm2 = m2[:,:,neur].flatten()
            self.split_half[neur] = np.corrcoef(tm1[~inv_idx],tm2[~inv_idx])[0,1]
        end = time.time()
        print(colored(f"Split map correlations, INIT TO self.split_half... \n","yellow")) 
        print(colored(f"Time to generate correlations: {round(end-start,3)} seconds... \n","yellow"))  
        colors = ["black"]
        bp = plt.boxplot(self.split_half,patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylim([-1, 1])
        plt.title(f"Split Map Correlations, n: {self.num_neurons}, mean: {round(np.mean(self.split_half),3)}",fontweight='bold')
        if export_fig:
            if not self.figure_folder:
                raise ValueError("SAVE FOLDER DOES NOT EXIST... \n")
            else:
                where = os.path.join(self.figure_folder,('split_half'+fig_extend)) # saves in this mouse/date recording folder 
                plt.savefig(where)
                print(colored(f"Correlations saved to: {where} ... \n","cyan")) 
        plt.close()

    def high_corr_example(self,export_fig=True,fig_extend='.pdf',view_smoothed=False):
        """
        This method generates an example of a cells spike plot overlayed onto the path of the mouse - it also generates the corresponding rate map 
        export_fig - default save figure 
        """
        if self.position is None or self.trace is None:
            raise ValueError("NEED TO CALL Calcium.get_mat_data() FROM PIPE LINE FRIST self.position OR self.trace DO NOT EXIST... \n")
        elif self.unsmoothed_maps is None or self.smoothed_maps is None: 
            raise ValueError("NEED TO CALL Calcium.get_maps() FROM PIPE LINE FIRST self.unsmoothed_maps OR self.smoothed_maps DOES NOT EXIST... ")
        neur = np.where(self.split_half == max(self.split_half))[0] # take most stable cell 
        plt.close() # close any open figures 
        plt.figure(figsize=(20, 10)) # set figure size 
        plt.subplot(1,2,1) # first subploy 
        plt.plot(self.position[0,:],self.position[1,:],color='black',alpha = 0.7) # plot path 
        spikes = [[float(self.position[0,i]),float(self.position[1,i])] for i in range(len(self.position[0,:])) if self.trace[neur,i] == 1] # index spikes 
        for i in range(len(spikes)):
            fired = spikes[i]
            plt.plot(fired[0],fired[1],marker='o',mec = 'blue',mfc = 'blue',markersize=5) # plot each spike location 
        plt.axis('equal')
        if view_smoothed:
            m = self.smoothed_maps
        else:
            m = self.unsmoothed_maps 
        plt.subplot(1,2,2)
        plt.imshow(m[:,:,neur],origin='lower')
        plt.axis('equal')
        plt.colorbar()
        if export_fig:
            if not self.figure_folder:
                raise ValueError("SAVE FOLDER DOES NOT EXIST... \n")
            else:
                where = os.path.join(self.figure_folder,('spike_plot_rate_map_examp'+fig_extend)) # saves in this mouse/date recording folder 
                plt.savefig(where)
                print(colored(f"Neuron Spiking Figure saved to: {where} ... \n","cyan")) 
        plt.close()

    def trace_split_half(self):
        """
        This method computes un-smoothed split halfs given inputs:
        """
        if self.bins is None:
            raise ValueError("Self.bins doesn't exist... Need to call self.set_params first... \n")
        if self.env_size is None:
            raise ValueError("Self.env_size doesn't exist... Need to call self.set_params first... \n")
        if self.smoothing_std is None:
            raise ValueError("Self.smoothing_std doesn't exist... Need to call self.set_params first... \n")
        if self.position is None: 
            raise ValueError("NEED TO CALL Calcium.get_mat_data() FROM PIPE LINE FRIST self.position DO NOT EXIST... \n")
        lin_pos = Utils.linear_position(self.position,self.env_size,self.bins) # convert position to discrete linear indices 
        lin_split = Utils.column_split(lin_pos) # split linear position equally into halves
        states = np.arange(np.prod(np.array(self.bins).astype(int))) # Linear States 
        maps = np.zeros((np.prod(self.bins).astype(int),self.num_neurons*2)) # init empty maps 
        t_split = Utils.column_split(self.trace) # split trace equally into halves 
        for s in states: # Iterate through states 
            maps[s,:self.num_neurons] = np.mean(t_split[:,(lin_split[:,:,0]==s).flatten(),0],axis=1) # 1:num_neurons columns 
            maps[s,self.num_neurons:] = np.mean(t_split[:,(lin_split[:,:,1]==s).flatten(),1],axis=1) # num_neurons:end columns 
        idx = np.any(np.isnan(maps),1) # Inlcude only if pixels have sampling in both halves, divide by zero gens nans
        corr_mat = np.corrcoef(maps[~idx,:],rowvar=False) # Correlation Matrix, excluding unsampled pixels 
        is_match = ((self.num_neurons*2)*np.arange(1,self.num_neurons))+self.num_neurons+np.arange(2,self.num_neurons+1)
        is_match = np.hstack((self.num_neurons+1,is_match))-1
        return corr_mat.T.ravel()[is_match] # return split half 

    def permute_p(self,iters=500,export_fig1=True,export_fig2=True,fig_extend='.pdf',override=False):
        """
        This method used permutation to generate a shuffle  null distribution to grab p_values
        bins - number of discrete states 
        env_size - size of the environment 
        iters - number of shuffled distributions, default 500 
        fig_extend - figure extension, default pdf
        export_fig1 - write hist fig to pdf, default true 
        export_fig2 - write cum proportion to pdf, default true
        override - rerun-needed when extracting p-values, default false
        """
        if self.bins is None:
            raise ValueError("Self.bins doesn't exist... Need to call self.set_params first... \n")
        if self.env_size is None:
            raise ValueError("Self.env_size doesn't exist... Need to call self.set_params first... \n")
        if self.smoothing_std is None:
            raise ValueError("Self.smoothing_std doesn't exist... Need to call self.set_params first... \n")
        if override:
            run_permutation = True
        elif (
            not os.path.exists(os.path.join(self.figure_folder, 'split_half_null_hist' + fig_extend)) or 
            not os.path.exists(os.path.join(self.figure_folder, 'cumulative_proportion' + fig_extend))
        ):
            run_permutation = True
        else:
            run_permutation = False
        if run_permutation:
            print(colored(f"Computing Actual Split-Half Correlation... \n", "yellow"))
            actual = self.trace_split_half() # Copmute actual Split-Half 
            lin_pos = Utils.linear_position(self.position,self.env_size,self.bins) # convert position to discrete linear indices 
            lin_split = Utils.column_split(lin_pos) # split linear position equally into halves 
            states = np.arange(np.prod(np.array(self.bins))) # Discrete states 
            null = np.zeros((self.num_neurons,iters)) # init empty null distribution array
            maps = np.zeros((np.prod(self.bins),self.num_neurons*2)) # init empty maps 
            is_match = ((self.num_neurons*2)*np.arange(1,self.num_neurons))+self.num_neurons + np.arange(2,self.num_neurons+1) # off diagonal indices for split-half 
            is_match = np.hstack((self.num_neurons+1,is_match))-1 # not last 
            start = time.time() # Grab iteration start time 
            print(colored(f"Computing Split Half Null: {iters} Iterations \n",'yellow'))
            for iter in tqdm(range(iters)): # Iterate Across Iteratinos 
                shift_len = np.random.randint(np.shape(self.trace)[1]//3,(np.shape(self.trace)[1]//3)*2) # chose circular shift length 
                tmp_t = np.roll(self.trace,-shift_len,axis=1) # Shift trace by -shift len (equivalent to a wrap around)
                t_split = Utils.column_split(tmp_t) # Split trace equally 
                for s in states: # Iterate over states 
                    maps[s,:self.num_neurons] = np.mean(t_split[:,(lin_split[:,:,0]==s).flatten(),0],axis=1) # 1:num_neurons columns 
                    maps[s,self.num_neurons:] = np.mean(t_split[:,(lin_split[:,:,1]==s).flatten(),1],axis=1) # num_neurons:end columns 
                idx = np.any(np.isnan(maps),1) # Ifnore any NaN (Result of divide by zero)
                corr_mat = np.corrcoef(maps[~idx,:],rowvar=False) # Correlation Matrix
                null[:,iter] = corr_mat.T.ravel()[is_match] # INIT to null
            end = time.time()
            print(colored(f"\n Run Time for {iters} Iterations: {round(end-start,3)} Seconds... \n",'yellow'))
            self.p_vals = np.sum(null > np.matlib.repmat(actual.reshape(-1,1),1,iters),1)/iters # Compute p_vals 
            self.is_cell = self.p_vals<=.05 # What cells have a p_val less than .05 
            self.percent_stable = (np.sum(self.is_cell)/self.num_neurons) * 100 # Percentage of stable cells 
            print(colored(f"Percent Place Cells: {self.percent_stable}%... \n"))
            print(colored(f"p_values INIT TO self.p_valus ... \n","yellow")) 
            print(colored(f"Percent Stable INIT to self.percent_stable ... \n","yellow")) 
            ### Compute Cumulative Proportion ### 
            cum_counts, _ = np.histogram(np.sort(self.p_vals), bins=np.arange(0,1,.001)) 
            cum_counts = (np.cumsum(cum_counts)/np.sum(cum_counts))*100
            cum_counts = np.insert(cum_counts,0,0)
            if export_fig1:
                print(colored(f"Generating Permutated Histogram... \n","yellow"))
                show_dist = 100 
                if iters < show_dist:
                    show_dist = iters
                for iter in range(show_dist-1): # only show 100 distributions or == iters if iters is ess 
                    plt.hist(null[:,iter],bins=30,color=[.7,.7,.7])
                    plt.xlim([-1, 1])
                plt.hist(actual,bins = 30,color=[0,0,1]) 
                plt.ylabel('Bin Counts')
                plt.xlabel('Correlation (Pearsons, sr)')
                plt.title('Permuted / Actual Split Half Correlations')
                if not self.figure_folder:
                    plt.close()
                    raise ValueError("SAVE FOLDER DOES NOT EXIST... \n")
                else:
                    where = os.path.join(self.figure_folder,('split_half_null_hist'+fig_extend)) # saves in this mouse/date recording folder 
                    plt.savefig(where)
                    print(colored(f"Permutated Distribution saved to: {where} ... \n","cyan")) 
                plt.close()
            if export_fig2:
                print(colored(f"Generating Cumulative Proportion... \n","yellow"))
                plt.plot(np.arange(0,1,.001),cum_counts,color = [1, 0, 0])
                plt.axvline(0.05, linestyle='--', color='gray', linewidth=1)  # <-- Add this line
                plt.ylim([0,100])
                plt.xlim([-.01,1])
                plt.ylabel('%')
                plt.xlabel('P-Value')
                plt.title('Cumulative Proportion % Place Cells')
                if not self.figure_folder:
                    plt.close()
                    raise ValueError("SAVE FOLDER DOES NOT EXIST... \n")
                else:
                    where = os.path.join(self.figure_folder,('cumulative_proportion'+fig_extend)) # saves in this mouse/date recording folder 
                    plt.savefig(where)
                    print(colored(f"Cumulative Proportion saved to: {where} ... \n","cyan")) 
                plt.close()
        else: 
            print(colored(f"""<<<<<<<<< PERMUTE SPLIT HALF SKIPPED >>>>>>>>> \n 
            {os.path.join(self.figure_folder,('cumulative_proportion'+fig_extend))} \n 
            AND \n 
            {os.path.join(self.figure_folder,('split_half_null_hist'+fig_extend))} ALREADY EXIST... \n""","cyan")) 

    def get_spatial_information(self,fig_extend = '.pdf',export_fig=True):
        """
        This method returns a place cells Naive spatial information (expressed in bit/spike)
        fig_extend - file extion for figure, default pdf 
        export_fig - default true, if true exports figure 
        """
        if self.bins is None:
            raise ValueError("Self.bins doesn't exist... Need to call self.set_params first... \n")
        if self.env_size is None:
            raise ValueError("Self.env_size doesn't exist... Need to call self.set_params first... \n")
        if self.sampling is None:
            raise ValueError("self.sampling doesn't exist... Need to call Calcium.grab_occupancy first... \n") 
        else:
            print(colored(f"Computing Spatial Information... \n", "yellow"))
            normed_occupancy = Utils.norm_arr(self.sampling).reshape(1,-1) # Compute normalized pixel occupancy 
            linear = Utils.linear_position(self.position,self.env_size,self.bins) # returns discrete lienar position  
            r = np.zeros((self.num_neurons,self.bins.flatten()[0]*self.bins.flatten()[1])) # firing rate of each neuron in each pixel 
            start = time.time() # time to compute - end 
            for s in range(np.size(r[0,:])): # iterate across pixels to compute MFR 
                r[:,s] = np.mean(self.trace[:,linear == s],axis=1) # mean columnwise 
            mfr = np.mean(self.trace,axis=1)
            firing_ratio = (r / np.matlib.repmat(mfr.reshape(-1,1),1,self.bins.flatten()[0]*self.bins.flatten()[1])) # Firing in each bin relative to mean firing rate 
            firing_ratio[np.isnan(firing_ratio)] = 0 # Handles divide by zero (NaN)
            information = firing_ratio * np.log2(firing_ratio) # Information term 
            self.spatial_information = np.nansum(np.matlib.repmat(normed_occupancy.reshape(1,-1),self.num_neurons,1) * information,axis=1)
            end = time.time()
            print(colored(f"\n Run Time: {round(end-start,3)} Seconds... \n",'yellow'))
            print(colored(f"Spatial information computed... Written to self.spatial_information... \n","yellow"))
        if export_fig:   
            if self.figure_folder is None:
                raise ValueError("SAVE FOLDER DOES NOT EXIST... Need to call self.get_figure_folder first... \n")
            else: 
                print(colored(f"Generating Spatial Information Histogram... \n","yellow"))
                where = os.path.join(self.figure_folder,('spatial_information'+fig_extend)) # saves in this mouse/date recording folder 
                plt.hist(self.spatial_information,bins=100,color=[0,0,1])
                plt.title(f"Spatial Information (Bits/Spike), Mean: {round(np.nanmean(self.spatial_information),3)}")
                plt.ylabel('Counts')
                plt.xlabel('Bits/Spike')
                plt.savefig(where)
                plt.close()
                print(colored(f"Spatial Information Velocity Histogram saved to: {where} ... \n","cyan")) 
        

    def get_state_transitions(self,tau=1):
        """
        This method computes discrete state-state transitions at varying time scales
        tau - time scale to compute transitions in frames, default = 1 frame 
        """
        if self.bins is None:
            raise ValueError("Self.bins doesn't exist... Need to call self.set_params first... \n")
        if self.env_size is None:
            raise ValueError("Self.env_size doesn't exist... Need to call self.set_params first... \n")
        print(colored(f"Computing State State Transitions... \n", "yellow"))
        tm1 = Utils.linear_position(self.position,self.env_size,self.bins) # Convert positions to linear indices 
        tm2 = np.roll(tm1,-tau)[:-tau] # Duplicate array - shifted by tau (delete wrap around)
        tm1 = tm1[:-tau] # Delete wrap around 
        transitions = np.arange(0,(np.prod(self.bins)**2)+1) # Possible transition 0-states**2
        lin_step = Utils.sub2ind((np.prod(self.bins),np.prod(self.bins)),tm1,tm2) # compute state-state transitions
        counts,_ = np.histogram(lin_step,transitions) # Count linear state-state transition
        self.raw_transitions = counts.reshape(np.prod(self.bins),np.prod(self.bins)) # raw transition counts 
        self.transition_probability = self.raw_transitions / numpy.matlib.repmat( np.sum(self.raw_transitions,axis=1).reshape(-1,1),1,np.prod(self.bins)) # Normalize Transition matrix
        print(colored(f"State Transitions Computed... Raw transition counts written to self.raw_transitions AND Normalized transitions written to self.transition probability... \n","yellow"))

    def analytical_successor_representation(self,gamma = .95):
        """
        This method computes the successor representation from transition proability matrix. A predictive 
        representation of state-state transtitions at various time scales.
        gamma - discounting factor (0-1), default .95
        """
        if self.raw_transitions is None or self.transition_probability is None: 
            self.get_state_transitions() 
            print(colored("State Transitions never computed... METHOD HAS BEEN CALLED \n ","yellow"))
        print(colored("Computing Successor Representation... \n ","yellow"))
        is_eye = np.identity(np.shape(self.raw_transitions)[0]) # Make Identity Matrix 
        A = is_eye - (gamma * self.transition_probability) # Analytical solution to SR 
        self.M = np.linalg.inv(A) # Inverse to finish analytical computation 
        print(colored(f"Successor representation computed... written to self.M ... \n","yellow"))

    def save_object(self):
        """
        This method writes all the data stored in self to a python readable file 
        """
        if self.data_save_file is None:
            raise ValueError("Save File does not exist... ")
        with open(self.data_save_file, 'wb') as f:
            pickle.dump(self.__dict__, f)
            print(colored(f"DATA SAVED TO: {self.data_save_file}... \n","cyan"))

    def load_object(self):
        """
        This method loads file if already written 
        """
        if self.data_save_file is None:
            raise ValueError("Save File does not exist... ")
        with open(self.data_save_file , 'rb') as f:
            loaded_dict = pickle.load(f)
            self.__dict__.update(loaded_dict)  
            self.data_save_file = os.path.join(self.save_folder, 'data.pkl')  
        print(colored(f"DATA Loaded FROM: {self.data_save_file}... \n","cyan"))

