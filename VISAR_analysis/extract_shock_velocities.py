
import matplotlib
# matplotlib.use('qtagg')   ###ENABLE ONLY FOR THE INTERACTIVE PLOT. THEN YOU HAVE TO RESTART THE KERNEL
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats, interpolate
from pathlib import Path
import pandas as pd
import os 
import re
import subprocess
from multiprocessing import get_context
from matplotlib.widgets import Slider

################################################################################################################################################################################################################################################

class process_txt:    

    def __init__(self,path_txt,path_xlsx):
        self.path_txt = path_txt
        self.path_xlsx = path_xlsx
        self.data = pd.read_csv(self.path_xlsx,header = None )
        self.visar2_index = self.data[self.data[0].str.contains('#VISAR 2', na=False)].index[0]

        self.data_visar1 = self.data.iloc[:self.visar2_index].reset_index(drop=True)
        self.data_visar2 = self.data.iloc[self.visar2_index:].reset_index(drop=True)

        self.index_for_values_visar1 = self.data_visar1[~self.data_visar1[0].str.startswith('#', na=False)].index[0]
        self.index_for_values_visar2 = self.data_visar2[~self.data_visar2[0].str.startswith('#', na=False)].index[0]

    class _VISAR1:
        def __init__(self, df,index_for_values):
            self.df = df
            self.index_for_values = index_for_values
        def offset(self):
            return float(self.df.iloc[1].str.strip()[0].split(':')[1].strip().split(' ')[1].replace('(','').replace(')',''.replace('(','').replace(')','')))
        def sensitivity(self):
            return float(self.df.iloc[2].str.strip()[0].split(':')[1])
        def slit(self):
            return float(self.df.iloc[3].str.strip()[0].split(':')[1])
        def interfringe(self):
            return float(self.df.iloc[4].str.strip()[0].split(':')[1].split(" ")[1])
        def angle(self):
            return float(self.df.iloc[4].str.strip()[0].split(':')[1].split(" ")[2])
        def hamamatsu_parameters(self):
            return np.asarray([float(par) for par in self.df.iloc[7].str.strip()[0].split(':')[1].strip(' Hamamatsu ').split()])
        def t0(self):
            return self.df.iloc[8].str.strip()[0].split(':')[1].strip(' ').split()[0]
        def delay(self):
            return self.df.iloc[8].str.strip()[0].split(':')[1].strip(' ').split()[1]
        def center(self):
            return self.df.iloc[8].str.strip()[0].split(':')[1].strip(' ').split()[1]
        def magnification(self):
            return self.df.iloc[8].str.strip()[0].split(':')[1].strip(' ').split()[0]
        def time_of_jumps(self):
            return np.asarray([float(self.df.iloc[10].str.strip()[0].split(':')[1].split(';')[i].strip().split()[0]) for i in range(len(self.df.iloc[10].str.strip()[0].split(':')[1].split(';')))])
        def number_of_jumps(self):
            return np.asarray([float(self.df.iloc[10].str.strip()[0].split(':')[1].split(';')[i].strip().split()[1]) for i in range(len(self.df.iloc[10].str.strip()[0].split(':')[1].split(';')))])
        def refractive_index(self):
            return np.asarray([float(self.df.iloc[10].str.strip()[0].split(':')[1].split(';')[i].strip().split()[2]) for i in range(len(self.df.iloc[10].str.strip()[0].split(':')[1].split(';')))])
        def time(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[0].to_numpy()
        def velocity(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[1].to_numpy()
        def error_velocity(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[2].to_numpy()
        def reflectivity(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[3].to_numpy()
        def error_reflectivity(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[4].to_numpy()
        def quality(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[5].to_numpy()
        def pixel(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[6].to_numpy()
        def shift(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[7].to_numpy()
        def error_shift(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[8].to_numpy()
        def refint(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[9].to_numpy()
        def shotint(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[10].to_numpy()
        def refcontr(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[11].to_numpy()
        def shotcontr(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[12].to_numpy()
      
    class _VISAR2:
        def __init__(self, df,index_for_values):
            self.index_for_values = index_for_values
            self.df = df
        def offset(self):
            return float(self.df.iloc[1].str.strip()[0].split(':')[1].strip().split(' ')[1].replace('(','').replace(')',''.replace('(','').replace(')','')))
        def sensitivity(self):
            return float(self.df.iloc[2].str.strip()[0].split(':')[1])
        def slit(self):
            return float(self.df.iloc[3].str.strip()[0].split(':')[1])
        def interfringe(self):
            return float(self.df.iloc[4].str.strip()[0].split(':')[1].split(" ")[1])
        def angle(self):
            return float(self.df.iloc[4].str.strip()[0].split(':')[1].split(" ")[2])
        def hamamatsu_parameters(self):
            return np.asarray([float(par) for par in self.df.iloc[7].str.strip()[0].split(':')[1].strip(' Hamamatsu ').split()])
        def t0(self):
            return self.df.iloc[8].str.strip()[0].split(':')[1].strip(' ').split()[0]
        def delay(self):
            return self.df.iloc[8].str.strip()[0].split(':')[1].strip(' ').split()[1]
        def center(self):
            return self.df.iloc[8].str.strip()[0].split(':')[1].strip(' ').split()[1]
        def magnification(self):
            return self.df.iloc[8].str.strip()[0].split(':')[1].strip(' ').split()[0]
        def time_of_jumps(self):
            return np.asarray([float(self.df.iloc[10].str.strip()[0].split(':')[1].split(';')[i].strip().split()[0]) for i in range(len(self.df.iloc[10].str.strip()[0].split(':')[1].split(';')))])
        def number_of_jumps(self):
            return np.asarray([float(self.df.iloc[10].str.strip()[0].split(':')[1].split(';')[i].strip().split()[1]) for i in range(len(self.df.iloc[10].str.strip()[0].split(':')[1].split(';')))])
        def refractive_index(self):
            return np.asarray([float(self.df.iloc[10].str.strip()[0].split(':')[1].split(';')[i].strip().split()[2]) for i in range(len(self.df.iloc[10].str.strip()[0].split(':')[1].split(';')))])
        def time(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[0].to_numpy()
        def velocity(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[1].to_numpy()
        def error_velocity(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[2].to_numpy()
        def reflectivity(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[3].to_numpy()
        def error_reflectivity(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[4].to_numpy()
        def quality(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[5].to_numpy()
        def pixel(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[6].to_numpy()
        def shift(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[7].to_numpy()
        def error_shift(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[8].to_numpy()
        def refint(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[9].to_numpy()
        def shotint(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[10].to_numpy()
        def refcontr(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[11].to_numpy()
        def shotcontr(self):
            return pd.DataFrame([list(map(float,row[0].split())) for row in self.df.iloc[self.index_for_values:].values])[12].to_numpy()
         

    @property
    def V1(self):
        return self._VISAR1(self.data_visar1,self.index_for_values_visar1)
    @property
    def V2(self):
        return self._VISAR2(self.data_visar2,self.index_for_values_visar2)
    
    def plot_velocity(self):
        plt.plot(self.V1.time(),self.V1.velocity(),alpha = 0.8)
        plt.plot(self.V2.time(),self.V2.velocity(), alpha = 0.8)
        plt.axvline(x = self.V1.time_of_jumps()[1], color = 'k', ls= '--')

################################################################################################################################################################################################################################################

    @property 
    def calculate_shock_velocity_reference_material(self,num_iter = 10 ):
        ### Find the index which minimizes the difference of time array with our jump time choice.
        def index_for_minimum_v1(): 
            return np.argmin(abs(self.V1.time()-self.V1.time_of_jumps()[1]))
        def index_for_minimum_v2(): 
            return np.argmin(abs(self.V2.time()-self.V2.time_of_jumps()[1]))
        ### Make slice of the arrays, according to interpolation region
        def slice(arr,ii,ni,refparmin,refparmax): # ni is the number of cells to move for region choice
            return arr[(ii-refparmin*ni):(ii-refparmax*ni)]
        
        #Keep it for the final cut 
        def fitting_polynomial(x,y,err,x_val):
            parameters, covariance_matrix = np.polyfit(x,y,1,w=1/err,cov = True)
            function = np.poly1d(parameters)
            parameters_sample = np.random.multivariate_normal(parameters,covariance_matrix, 1000)
            lerr_cov = np.asarray([np.std([np.poly1d(c)(xx) for c in parameters_sample]) for xx in x])
            err_val = np.std(np.array([np.poly1d(c)(x_val) for c in parameters_sample]))
            return function(x_val), err_val #value and error
        ### Calculate average shock velocity
        def weigthed_average(x,s): #x and s should be arrays
            x_bar, w_bar = np.sum(x*s**(-2))/np.sum(s**(-2)) ,np.sqrt(np.sum(s**(-2))**(-1))
            return x_bar,w_bar
        
        #### ANALYZE THE slice.xlsx FILE#####
        number_of_shot = int(re.search(r'_(\d+)',self.path_txt).group(1))
        slice_data_frame = pd.read_excel(self.path_xlsx)
        refparmin = slice_data_frame[slice_data_frame[slice_data_frame.columns[0]]==number_of_shot]['refparmin'].values[0]
        refparmax = slice_data_frame[slice_data_frame[slice_data_frame.columns[0]]==number_of_shot]['refparmax'].values[0]
        ni = slice_data_frame[slice_data_frame[slice_data_frame.columns[0]]==number_of_shot]['refni'].values[0]
        

        time_jump, visar_jitter = self.V1.time_of_jumps()[1],1e-2  #[ns]
        time_jump_monte = np.random.normal(time_jump, visar_jitter, num_iter)   
        vel_list, vel_err_list = [], []   

        for time_jump_i in time_jump_monte:
            t1, t2 = slice(self.V1.time(),index_for_minimum_v1(),ni,refparmin,refparmax), slice(self.V2.time(),index_for_minimum_v2(),ni,refparmin,refparmax)
            v1, v2 = slice(self.V1.velocity(),index_for_minimum_v1(),ni,refparmin,refparmax), slice(self.V2.velocity(),index_for_minimum_v2(),ni,refparmin,refparmax)
            errv1 , errv2 =  slice(self.V1.error_velocity(),index_for_minimum_v1(),ni,refparmin,refparmax), slice(self.V2.error_velocity(),index_for_minimum_v2(),ni,refparmin,refparmax)
            vv1 = fitting_polynomial(t1,v1,errv1,time_jump_i)
            vv2 = fitting_polynomial(t2,v2,errv2,time_jump_i)
            average = weigthed_average(np.asarray([vv1[0],vv2[0]]),np.asarray([vv1[1],vv2[1]]))
            vel_list.append(average[0])
            vel_err_list.append(average[1])
        final_result = weigthed_average(np.asarray(vel_list),np.asarray(vel_err_list))
        return  final_result[0], final_result[1], number_of_shot #value and error
################################################################################################################################################################################################################################################

    @property 
    def calculate_shock_velocity_sample_material(self,num_iter = 10 ):
        ### Find the index which minimizes the difference of time array with our jump time choice.
        def index_for_minimum_v1(): 
            return np.argmin(abs(self.V1.time()-self.V1.time_of_jumps()[1]))
        def index_for_minimum_v2(): 
            return np.argmin(abs(self.V2.time()-self.V2.time_of_jumps()[1]))
        ### Make slice of the arrays, according to interpolation region
        def slice(arr,ii,ni,samparmin,samparmax): # ni is the number of cells to move for region choice
            return arr[(ii+samparmin*ni):(ii+samparmax*ni)]
        
        #Keep it for the final cut 
        def fitting_polynomial(x,y,err,x_val):
            parameters, covariance_matrix = np.polyfit(x,y,1,w=1/err,cov = True)
            function = np.poly1d(parameters)
            parameters_sample = np.random.multivariate_normal(parameters,covariance_matrix, 1000)
            lerr_cov = np.asarray([np.std([np.poly1d(c)(xx) for c in parameters_sample]) for xx in x])
            err_val = np.std(np.array([np.poly1d(c)(x_val) for c in parameters_sample]))
            return function(x_val), err_val #value and error
        ### Calculate average shock velocity
        def weigthed_average(x,s): #x and s should be arrays
            x_bar, w_bar = np.sum(x*s**(-2))/np.sum(s**(-2)) ,np.sqrt(np.sum(s**(-2))**(-1))
            return x_bar,w_bar

        #### ANALYZE THE slice.xlsx FILE#####
        number_of_shot = int(re.search(r'_(\d+)',self.path_txt).group(1))
        slice_data_frame = pd.read_excel(self.path_xlsx)
        samparmin = slice_data_frame[slice_data_frame[slice_data_frame.columns[0]]==number_of_shot]['samparmin'].values[0]
        samparmax = slice_data_frame[slice_data_frame[slice_data_frame.columns[0]]==number_of_shot]['samparmax'].values[0]
        ni = slice_data_frame[slice_data_frame[slice_data_frame.columns[0]]==number_of_shot]['samni'].values[0]

        time_jump, visar_jitter = self.V1.time_of_jumps()[1],1e-2  #[ns]
        time_jump_monte = np.random.normal(time_jump, visar_jitter, num_iter)   
        vel_list, vel_err_list = [], []   

        for time_jump_i in time_jump_monte:
            t1, t2 = slice(self.V1.time(),index_for_minimum_v1(),ni,samparmin,samparmax), slice(self.V2.time(),index_for_minimum_v2(),ni,samparmin,samparmax)
            v1, v2 = slice(self.V1.velocity(),index_for_minimum_v1(),ni,samparmin,samparmax), slice(self.V2.velocity(),index_for_minimum_v2(),ni,samparmin,samparmax)
            errv1 , errv2 =  slice(self.V1.error_velocity(),index_for_minimum_v1(),ni,samparmin,samparmax), slice(self.V2.error_velocity(),index_for_minimum_v2(),ni,samparmin,samparmax)
            vv1 = fitting_polynomial(t1,v1,errv1,time_jump_i)
            vv2 = fitting_polynomial(t2,v2,errv2,time_jump_i)
            average = weigthed_average(np.asarray([vv1[0],vv2[0]]),np.asarray([vv1[1],vv2[1]]))
            vel_list.append(average[0])
            vel_err_list.append(average[1])
        final_result = weigthed_average(np.asarray(vel_list),np.asarray(vel_err_list))
        return  final_result[0], final_result[1], number_of_shot #value and error
    
 ################################################################################################################################################################################################################################################
  
    def manual_slicing(self):

        ### Find the index which minimizes the difference of time array with our jump time choice.
        def index_for_minimum_v1(): 
            return np.argmin(abs(self.V1.time()-self.V1.time_of_jumps()[1]))
        def index_for_minimum_v2(): 
            return np.argmin(abs(self.V2.time()-self.V2.time_of_jumps()[1]))
   
        ### Fitting polynomial
        def fitting_polynomial_left(x,y,err):
            parameters, covariance_matrix = np.polyfit(x,y,1,w=1/err,cov = True)
            function = np.poly1d(parameters)
            parameters_sample = np.random.multivariate_normal(parameters,covariance_matrix, 100)
            lerr_cov = np.asarray([np.std([np.poly1d(c)(xx) for c in parameters_sample]) for xx in x])
            return function, lerr_cov, parameters_sample
        def fitting_polynomial_right(x,y,err):
            parameters, covariance_matrix = np.polyfit(x,y,1,w=1/err,cov = True)
            function = np.poly1d(parameters)
            parameters_sample = np.random.multivariate_normal(parameters,covariance_matrix, 100)
            lerr_cov = np.asarray([np.std([np.poly1d(c)(xx) for c in parameters_sample]) for xx in x ])
            return function, lerr_cov, parameters_sample
        ### Make slice of the arrays, according to interpolation region
        def slicel(arr,ii,ni,parmin,parmax): # ni is the number of cells to move for region choice
            return arr[(ii-parmin*ni):(ii-parmax*ni)]
        def slicer(arr,ii,ni,parmin,parmax): # ni is the number of cells to move for region choice
            return arr[(ii+parmin*ni):(ii+parmax*ni)]
        ### Define parameters
        ########## slicing on left and right ##########
        nil = 2 
        parminl = 30
        parmaxl = 5
        nir = 2
        parminr = 10
        parmaxr = 80
        time_jump = self.V1.time_of_jumps()[1]  

        ################################################################################ 
        ###LEFT###
        t1l, t2l = slicel(self.V1.time(),index_for_minimum_v1(),nil,parminl,parmaxl), slicel(self.V2.time(),index_for_minimum_v2(),nil,parminl,parmaxl)
        v1l, v2l = slicel(self.V1.velocity(),index_for_minimum_v1(),nil,parminl,parmaxl), slicel(self.V2.velocity(),index_for_minimum_v2(),nil,parminl,parmaxl)
        errv1l , errv2l =  slicel(self.V1.error_velocity(),index_for_minimum_v1(),nil,parminl,parmaxl), slicel(self.V2.error_velocity(),index_for_minimum_v2(),nil,parminl,parmaxl)

        ####RIGHT###
        t1r, t2r = slicer(self.V1.time(),index_for_minimum_v1(),nir,parminr,parmaxr), slicer(self.V2.time(),index_for_minimum_v2(),nir,parminr,parmaxr)
        v1r, v2r = slicer(self.V1.velocity(),index_for_minimum_v1(),nir,parminr,parmaxr), slicer(self.V2.velocity(),index_for_minimum_v2(),nir,parminr,parmaxr)
        errv1r , errv2r =  slicer(self.V1.error_velocity(),index_for_minimum_v1(),nir,parminr,parmaxr), slicer(self.V2.error_velocity(),index_for_minimum_v2(),nir,parminr,parmaxr)
        

        ###### left side (reference)####################
        vv1l = fitting_polynomial_left(t1l,v1l,errv1l)
        vv2l = fitting_polynomial_left(t2l,v2l,errv2l)
        ###### right side (sample)####################
        vv1r = fitting_polynomial_right(t1r,v1r,errv1r)
        vv2r = fitting_polynomial_right(t2r,v2r,errv2r)
        ################################################################################


        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        axes[0].axvline(x = self.V1.time()[index_for_minimum_v1()-parmaxl*nil], color = 'red', ls= '--')
        axes[0].axvline(x = self.V1.time()[index_for_minimum_v1()-parminl*nil], color = 'red', ls= '--')
        axes[0].plot(self.V1.time(),self.V1.velocity(),alpha = 0.8,label='V1')
        axes[0].plot(self.V2.time(),self.V2.velocity(), alpha = 0.8,label='V2')
        axes[0].axvline(x = self.V1.time_of_jumps()[1], color = 'k', ls= '--')
        axes[0].axvline(x = self.V1.time()[index_for_minimum_v1()+parminr*nir], color = 'magenta', ls= '--')
        axes[0].axvline(x = self.V1.time()[index_for_minimum_v1()+parmaxr*nir], color = 'magenta', ls= '--')
        axes[0].set_title('VISAR data')

        axes[0].set_xlabel('Time [ns]')
        axes[0].set_ylabel('Velocity [μm/ns]')
        axes[0].legend()

        axes[1].errorbar(t1l,v1l,errv1l,label = 'V1')
        axes[1].errorbar(t2l,v2l,errv2l,label = 'V2')
        axes[1].errorbar(t1l,vv1l[0](np.asarray(t1l)),vv1l[1],color = 'k',label = 'V1-fit')
        axes[1].errorbar(t2l,vv2l[0](np.asarray(t2l)),vv2l[1],label = 'V2-fit')
        axes[1].set_title('Reference material (left)')
        axes[1].set_xlabel('Time [ns]')
        axes[1].set_ylabel('Velocity [μm/ns]')
        axes[1].legend()

        axes[2].errorbar(t1r,v1r,errv1r,label = 'V1')
        axes[2].errorbar(t2r,v2r,errv2r,label = 'V2')
        axes[2].errorbar(t1r,vv1r[0](np.asarray(t1r)),vv1r[1],color = 'k',label = 'V1-fit')
        axes[2].errorbar(t2r,vv2r[0](np.asarray(t2r)),vv2r[1],label = 'V2-fit')
        axes[2].set_title('Sample material (right)')
        axes[2].set_xlabel('Time [ns]')
        axes[2].set_ylabel('Velocity [μm/ns]')
        axes[2].legend()
################################################################################################################################################################################################################################################

    def interactive_slicing(self):

        def index_for_minimum_v1(): 
            return np.argmin(abs(self.V1.time() - self.V1.time_of_jumps()[1]))

        def index_for_minimum_v2(): 
            return np.argmin(abs(self.V2.time() - self.V2.time_of_jumps()[1]))
        
        ### Fitting polynomial
        def fitting_polynomial_left(x,y,err):
            parameters, covariance_matrix = np.polyfit(x,y,1,w=1/err,cov = True)
            function = np.poly1d(parameters)
            parameters_sample = np.random.multivariate_normal(parameters,covariance_matrix, 100)
            lerr_cov = np.asarray([np.std([np.poly1d(c)(xx) for c in parameters_sample]) for xx in x])
            return function, lerr_cov, parameters_sample
        def fitting_polynomial_right(x,y,err):
            parameters, covariance_matrix = np.polyfit(x,y,1,w=1/err,cov = True)
            function = np.poly1d(parameters)
            parameters_sample = np.random.multivariate_normal(parameters,covariance_matrix, 100)
            lerr_cov = np.asarray([np.std([np.poly1d(c)(xx) for c in parameters_sample]) for xx in x ])
            return function, lerr_cov, parameters_sample
        
        # Cache expensive operations
        _v1_time = self.V1.time()
        _v2_time = self.V2.time()
        _v1_velocity = self.V1.velocity()
        _v2_velocity = self.V2.velocity()
        _v1_error = self.V1.error_velocity()
        _v2_error = self.V2.error_velocity()
        _v1_jump_time = self.V1.time_of_jumps()[1]
        
        # Precompute indices
        _ii_v1 = index_for_minimum_v1()
        _ii_v2 = index_for_minimum_v2()

        def update(val):
            ni = int(s_ni.val)
            parminl = int(s_parminl.val)
            parmaxl = int(s_parmaxl.val)
            parminr = int(s_parminr.val)
            parmaxr = int(s_parmaxr.val)

            # Clear axes efficiently
            for ax in axes:
                ax.clear()

            # Use cached values
            ii_v1 = _ii_v1
            ii_v2 = _ii_v2

            # Calculate slice indices once
            start_l_v1 = ii_v1 - parminl * ni
            end_l_v1 = ii_v1 - parmaxl * ni
            start_r_v1 = ii_v1 + parminr * ni  
            end_r_v1 = ii_v1 + parmaxr * ni
            
            start_l_v2 = ii_v2 - parminl * ni
            end_l_v2 = ii_v2 - parmaxl * ni
            start_r_v2 = ii_v2 + parminr * ni
            end_r_v2 = ii_v2 + parmaxr * ni

            # Slicing with cached arrays
            t1l = _v1_time[start_l_v1:end_l_v1]
            v1l = _v1_velocity[start_l_v1:end_l_v1]
            errv1l = _v1_error[start_l_v1:end_l_v1]
            t2l = _v2_time[start_l_v2:end_l_v2]
            v2l = _v2_velocity[start_l_v2:end_l_v2]
            errv2l = _v2_error[start_l_v2:end_l_v2]

            t1r = _v1_time[start_r_v1:end_r_v1]
            v1r = _v1_velocity[start_r_v1:end_r_v1]
            errv1r = _v1_error[start_r_v1:end_r_v1]
            t2r = _v2_time[start_r_v2:end_r_v2]
            v2r = _v2_velocity[start_r_v2:end_r_v2]
            errv2r = _v2_error[start_r_v2:end_r_v2]

            # Fitting - only if we have data points
            if len(t1l) > 1 and len(t2l) > 1:
                vv1l = fitting_polynomial_left(t1l, v1l, errv1l)
                vv2l = fitting_polynomial_left(t2l, v2l, errv2l)
            else:
                vv1l = vv2l = (lambda x: x, np.array([]), np.array([]))
                
            if len(t1r) > 1 and len(t2r) > 1:
                vv1r = fitting_polynomial_right(t1r, v1r, errv1r)
                vv2r = fitting_polynomial_right(t2r, v2r, errv2r)
            else:
                vv1r = vv2r = (lambda x: x, np.array([]), np.array([]))

            # Plot VISAR data with cached arrays
            axes[0].plot(_v1_time, _v1_velocity, alpha=0.8, label='V1')
            axes[0].plot(_v2_time, _v2_velocity, alpha=0.8, label='V2')
            axes[0].axvline(x=_v1_jump_time, color='k', ls='--', label='Jump')

            # Region lines - use cached time array
            axes[0].axvline(_v1_time[start_l_v1], color='red', ls='--')
            axes[0].axvline(_v1_time[end_l_v1], color='red', ls='--')
            axes[0].axvline(_v1_time[start_r_v1], color='magenta', ls='--')
            axes[0].axvline(_v1_time[end_r_v1], color='magenta', ls='--')

            axes[0].set_title('VISAR data')
            axes[0].set_xlabel('Time [ns]')
            axes[0].set_ylabel('Velocity [μm/ns]')
            axes[0].legend()

            # Left fit
            if len(t1l) > 0:
                axes[1].errorbar(t1l, v1l, errv1l, label='V1')
                axes[1].plot(t1l, vv1l[0](t1l), 'k', label='V1-fit')
            if len(t2l) > 0:
                axes[1].errorbar(t2l, v2l, errv2l, label='V2')
                axes[1].plot(t2l, vv2l[0](t2l), label='V2-fit')
            axes[1].set_title('Reference (Left)')
            axes[1].legend()

            # Right fit
            if len(t1r) > 0:
                axes[2].errorbar(t1r, v1r, errv1r, label='V1')
                axes[2].plot(t1r, vv1r[0](t1r),'k', label='V1-fit')
            if len(t2r) > 0:
                axes[2].errorbar(t2r, v2r, errv2r, label='V2')
                axes[2].plot(t2r, vv2r[0](t2r), label='V2-fit')
            axes[2].set_title('Sample (Right)')
            axes[2].legend()

            plt.draw()

        # === Figure + sliders ===
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        plt.subplots_adjust(bottom=0.5)

        ax_ni = plt.axes([0.15, 0.23, 0.65, 0.03])
        ax_parminl = plt.axes([0.15, 0.18, 0.65, 0.03])
        ax_parmaxl = plt.axes([0.15, 0.13, 0.65, 0.03])
        ax_parminr = plt.axes([0.15, 0.08, 0.65, 0.03])
        ax_parmaxr = plt.axes([0.15, 0.03, 0.65, 0.03])

        s_ni = Slider(ax_ni, 'ni', 1, 10, valinit=2, valstep=1)
        s_parminl = Slider(ax_parminl, 'parminL', 1, 50, valinit=20, valstep=1)
        s_parmaxl = Slider(ax_parmaxl, 'parmaxL', 1, 20, valinit=5, valstep=1)
        s_parminr = Slider(ax_parminr, 'parminR', 1, 50, valinit=20, valstep=1)
        s_parmaxr = Slider(ax_parmaxr, 'parmaxR', 1, 100, valinit=60, valstep=1)

        for s in [s_ni, s_parminl, s_parmaxl, s_parminr, s_parmaxr]:
            s.on_changed(update)

        update(None)
        plt.show()

            
        

   
path_of_txt = "..."
path_xlsx = "..."
paths_of_files_to_process = []
for p in os.listdir(path_of_txt):
    if re.search(r'_(\d+)', p):  paths_of_files_to_process.append(os.path.join(path_of_txt, p))
paths_of_files_to_process.sort(key=lambda x: int(re.search(r'...', os.path.basename(x)).group(1)))


def run_process_calculate_shock_velocity_reference_material(args):
    path, xlsxpath = args
    return process_txt(path,xlsxpath).calculate_shock_velocity_reference_material

def run_process_calculate_shock_velocity_sample_material(args):
    path, xlsxpath = args
    return process_txt(path,xlsxpath).calculate_shock_velocity_sample_material

with get_context("fork").Pool(6) as pool:
        args_list = [(path_txt,path_xlsx) for path_txt in paths_of_files_to_process]
        shock_velocity_reference_material = pool.map(run_process_calculate_shock_velocity_reference_material, args_list)


with get_context("fork").Pool(6) as pool:
        args_list = [(path_txt,path_xlsx) for path_txt in paths_of_files_to_process]
        shock_velocity_sample_material = pool.map(run_process_calculate_shock_velocity_sample_material, args_list)

