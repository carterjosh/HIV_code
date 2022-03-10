import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.optimize import curve_fit
import math

class Octet():
    '''
    A convenience class for reading in and performing various plots and fits on Octet BLI data.
    '''

    def read_data(self, path, var_list):
        '''
        Reads in data from directory given by path, changes column names to match those provided in var_list.
        
        Arguments:
            path filepath to directory
            var_list list of column names
            
        Returns:
            Merged dataframe of Octet data with time set to 0 at beginning of association
        '''
        print('Reading in data')
        files = sorted([file for file in os.listdir(path) if file.endswith(".tsv")])
        
        out = None
        
        for file in files:
            try:
                if isinstance(out, pd.DataFrame):
                    tmp = pd.read_csv(''.join([path,file]), sep='\t')
                    out[file[0:2]] = tmp['Data1']
                else:
                    out = pd.read_csv(''.join([path,file]), sep='\t')
                    out = out[['Time1', 'Data1']]

            except:
                print(' '.join([file, 'not found!']))
                
        assert len(out.columns)==len(var_list), 'Variable list does not match number of columns!'
        out.columns = var_list
        
        out['Time'] = out['Time'] - min(out['Time'])
        
        return out
    
    def plot_data(self, data, scale='linear'):
        
        '''
        Plots raw binding data.
        
        Arguments:
            data     Dataframe of data. Must contain column called 'Time'
            scale    Str option on whether yscale is linear (default) or log
        '''
        
        for col in data.columns[~data.columns.str.contains('Time')]:
            s = sns.scatterplot(data=data, x='Time', y = col, edgecolor='None', label=col, s=5)
            plt.ylabel('Response (nm)')
            plt.xlabel('Time (s)')
            plt.legend(loc = 'best', fontsize='x-small')
            
            
            if scale == 'log':
                s.set(yscale='log')
                
        return s
    
    def fit_data(self, data, conc_range, dissoc_time, binding_model='1to1', global_fit=True, scale='linear'):
        '''
        Function that fits binding curves to data. Can perform global or individual fits to data. Currently only
        supports a 1to1 binding model.
        
        Arguments:
            data              The dataset to fit. Must contain a column labeled Time (Dataframe)
            conc_range        A list of the concentration ranges for each column. Must be in same order as column
            dissoc_time       The time that dissociation began in seconds (float)
            binding_model     Which type of binding model to fit (str)
            global_fit        Boolean on whether or not to perform a global fit
            scale             Whether to plot the y axis as linear (default) or log (str)
            
        Returns:
            popt              Either tuple (global) or dataframe (individual) of fitted values
            s                 Seaborn plot object of fitted curves
        '''
        
        def association(t, conc, kon, koff, rmax):
            return (conc * rmax) / (conc + (koff / kon)) * (1 - math.e**(-1 * (kon * conc + koff) * t))

        def dissociation(t, r0, koff):
            return r0 * math.e**(-1 * koff * t)
        
        def binding(X, kon, koff, rmax):
            
            conc, t = X
            
            r0 = association(t[t==t[dissoc_time*5]], conc[t==t[dissoc_time*5]], kon, koff, rmax)
            new_r0 = []
            for item in r0:
                new_r0.extend([item]*int(len(t[t>=dissoc_time])/len(set(conc))))
            r0 = new_r0
            
            out = np.empty(len(t))
            out[t<dissoc_time] = association(t[t<dissoc_time], conc[t<dissoc_time], kon, koff, rmax)
            out[t>=dissoc_time] = dissociation(t[t>=dissoc_time] - t[dissoc_time*5], r0, koff)
            
            return out
         
        
        time = np.array(data['Time'])
        
        #global fitting
        if global_fit == True:
            print('Performing global fitting...')
            
            if binding_model == '1to1':
            
                columns = data.columns[~data.columns.str.contains('Time')]
                combo_Y = []
                for col in columns:
                    combo_Y.extend(data[col].to_list())
                    
                conc_list = []
                t_list = []
                for conc in conc_range:
                    conc_list.extend([conc]*len(time))
                    t_list.extend(time)
                    
                combo_X = np.row_stack([np.array(conc_list), np.array(t_list)])

                popt, pcov = curve_fit(binding, combo_X, combo_Y)
                print('Fitted Kd is', popt[1]/popt[0])
                
                fit_data = binding(combo_X, *popt)
                #splits combined response values into correct size chunks
                split_fit = [fit_data[i:i + len(time)] for i in range(0, len(fit_data), len(time))]
                
                for conc, fit in zip(conc_range, split_fit):
                    s = sns.scatterplot(x = time, y = fit, label=conc, s=1, color='black', edgecolor='None')
                    plt.ylabel('Response (nm)')
                    plt.xlabel('Time (s)')
                    plt.legend(loc = 'best', fontsize='x-small')


                    if scale == 'log':
                        s.set(yscale='log')
        
        #individual fitting
        else:
            print('Performing inividual fitting...')
            
            if binding_model == '1to1':
                
                kon_list = []
                koff_list = []
                rmax_list = []
                for col, conc in zip(data.columns[~data.columns.str.contains('Time')], conc_range):
                    combo_X = np.row_stack([np.array([conc]*len(time)), time])
                    try:
                        popt, pcov = curve_fit(binding, combo_X, data[col])
                        kon_list.append(popt[0])
                        koff_list.append(popt[1])
                        rmax_list.append(popt[2])

                        fit = binding(combo_X, *popt)
                        s = sns.scatterplot(x = time, y = fit, label=conc, s=1, color='black', edgecolor='None')
                        plt.ylabel('Response (nm)')
                        plt.xlabel('Time (s)')
                        plt.legend(loc = 'best', fontsize='x-small')
                    
                    except:
                        print('Fitting failed for', conc)
                        kon_list.append(None)
                        koff_list.append(None)
                        rmax_list.append(None)
                        
                
                popt = pd.DataFrame({'Kon':kon_list,'Koff':koff_list,'Rmax':rmax_list})

        return popt, s
