#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 09:43:54 2021

@author: millij
"""

# from astropy.io import fits
import argparse
# import os
# import glob
# from matplotlib.colors import LogNorm,SymLogNorm
from pathlib import Path
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
# import matplotlib.gridspec as gridspec 
import matplotlib.pyplot as plt


path_package = Path(__file__).parent
path_data = path_package.joinpath('data')

def read_input_csv(filename,verbose=False):
    """
    Open and read a csv table containing at least the SPF (in brightness) and/or
    a polarisation fraction as a function of the phase angle or scattering angle. 
    The function checks that the mandatory columns are present
    Output:
        - panda table
    """
    csv_table = pd.read_csv(filename)
    columns = csv_table.keys()
    
    # check that phase or scattering angles are present in the columns of the table
    angles_present = False
    if 'phase_angle' in columns:
        angles_present = True
        max_phase_angle = np.nanmax(csv_table['phase_angle'])
        if max_phase_angle<=np.pi:
            print('Warning, the maxmimum phase angle is {0:f}deg. Make sure it is indeed expressed in degrees'.format(max_phase_angle))         
        if 'scat_angle' not in columns:        
            csv_table['scat_angle'] = 180-csv_table['Phase_angle']
    if 'scat_angle' in columns:        
        angles_present = True
        max_scat_angle = np.nanmax(csv_table['scat_angle'])
        if max_scat_angle<=np.pi:
            print('Warning, the maxmimum scattering angle is {0:f}deg. Make sure it is indeed expressed in degrees'.format(max_scat_angle))         
    if not angles_present:
        raise ValueError('phase_angle or scat_angle not in the column names ({0:s})'.format(', '.join(columns)))
        
    # check that brightness or polarisation phase functions are present in the columns of the table
    phase_curve_present = False
    if 'spf' in columns:        
        phase_curve_present=True
        # if 'error_spf' not in columns:
        #     raise ValueError('error_spf not in the column names ({0:s})'.format(', '.join(columns))) 
    if 'polar_frac' in columns:        
        phase_curve_present=True
        # if 'error_polar_frac' not in columns:
        #     raise ValueError('error_polar_frac not in the column names ({0:s})'.format(', '.join(columns))) 
        max_polar_frac = np.nanmax(csv_table['polar_frac'])
        if max_polar_frac>1:
            raise ValueError('The maximum polarisation fraction is {0:f} > 1. Make sure it is expressed in decimal values and not in %'.format(max_polar_frac)) 
    if not phase_curve_present:
        raise ValueError('spf or polar_frac not in the column names ({0:s})'.format(', '.join(columns)))     

    if verbose:
        print('\nExtract of the csv input table (beginning)')
        print(csv_table.head())
        # print()
        print('\nExtract of the csv input table (end)')
        print(csv_table.tail())
        print()

    return csv_table

def compare_progra2_db(csv_table,path,scat_angles_interp=None,verbose=False,\
                       plot=True,name='comparison'):
    """
    """

    filenames = sorted(path_data.glob('**/*.resul'))
    if scat_angles_interp is not None:
        scat_angles = np.asarray(scat_angles_interp)
        scat_angles.sort()
        if verbose:
            print('The input phase function will be resampled at '+\
                  'the following scattering angles before comparison '+\
                  'to the PROGRA2 database: '+\
                  ', '.join(['{0:.2f}'.format(s) for s in scat_angles])+'\n')
    else:
        scat_angles = np.asarray(csv_table['scat_angle'])
    
    # 2 arrays to store the names of the polarimetric/brightness data
    name_polar = []
    name_brightness = []
    
    # 2 arrays to store the chi2 of the brightness data
    chi2_polar = []
    chi2_brightness = []
    
    # generic arrays to store both polarimetric and brightness data
    generic_name = []
    generic_isPolar = []
    generic_chi2 = []
    
    for ii,filename in enumerate(filenames):
        experiment_name = filename.parent.name+'_'+filename.stem 
        generic_experiment_name = experiment_name.replace('Polar','').replace('Bright','')
        
        data = pd.read_csv(filename,delim_whitespace=True,skiprows=[1])
        data['scat_angle'] = 180-data['Phase_angle']
        data = data.sort_values('scat_angle')
    
        if 'Brightness' in data.columns and 'spf' in csv_table.columns: 
            name_brightness.append(experiment_name)
            generic_isPolar.append(False)
            generic_name.append(generic_experiment_name)
            
            # we compute the best scaling factor
            
            spf_function = interp1d(data['scat_angle'],data['Brightness'],bounds_error=False)
            spf_err_function = interp1d(data['scat_angle'],data['Error'],bounds_error=False)
            spf_data_interpolated = spf_function(scat_angles)
            spf_err_data_interpolated = spf_err_function(scat_angles)
            
            if scat_angles_interp is not None:
                # we perform an interpolation in this case:
                csv_spf_interp_function = interp1d(csv_table['scat_angle'],csv_table['spf'],bounds_error=False)
                csv_spf_data_interpolated = csv_spf_interp_function(scat_angles)
            else:
                csv_spf_data_interpolated = np.asarray(csv_table['spf'])                
            
            scaling = np.sum(spf_data_interpolated*csv_spf_data_interpolated)/np.sum(csv_spf_data_interpolated**2)
            residuals = (spf_data_interpolated-scaling*csv_spf_data_interpolated)/spf_err_data_interpolated
            chi2 = np.sum(residuals**2)
            chi2_brightness.append(chi2)
            generic_chi2.append(chi2)

            if verbose:
                if np.any(~np.isfinite(residuals)):
                    print('{0:s} chi2 = {1:.1f}. WARNING ! The interpolation range was likely too large for this sample ({2:.1f}deg - {3:.1f}deg)'.format(experiment_name,chi2,np.min(scat_angles),np.max(scat_angles)))
                    # print('WARNING, the interpolation likely failed:',np.asarray(residuals),', chi2={0:.1f}'.format(chi2))
                else:
                    print('{0:s} chi2 = {1:.1f}'.format(experiment_name,chi2))
            
            # plt.close(0)
            if plot:
                fig = plt.figure(0, figsize=[8,6])
                fig.clf()
                ax=plt.subplot(111)
                ax.plot(data['scat_angle'],data['Brightness'],color='tomato',\
                     linestyle='-',label='{0:d} {1:s}'.format(ii,experiment_name))
                ax.fill_between(data['scat_angle'],\
                                 data['Brightness']+data['Error'],\
                                 data['Brightness']-data['Error'],\
                                 alpha=0.3,color='tomato',label='Uncertainty')
                ax.plot(csv_table['scat_angle'],csv_table['spf']*scaling,color='black',\
                         linestyle='-',label=name)
                ax.fill_between(csv_table['scat_angle'],\
                                 csv_table['spf']*scaling+csv_table['spf_error']*scaling,\
                                 csv_table['spf']*scaling-csv_table['spf_error']*scaling,\
                                 alpha=0.3,color='grey',label='Uncertainty')
                ax.legend(frameon=False,loc='best')
                ax.set_ylabel('SPF')
                ax.set_xlabel('Scattering angle in $^\circ$')
                ax.set_title('{0:s}: $\\chi^2$={1:.1f}'.format(experiment_name,chi2))
                ax.grid()
                plt.tight_layout()
                fig.savefig(path.joinpath(experiment_name+'.pdf'))    
                
        elif ('Polarization' in data.columns and 'polar_frac' in csv_table.columns): 
            name_polar.append(experiment_name)
            generic_isPolar.append(True)
            generic_name.append(generic_experiment_name)
            
            # we compute the chi2
            p_function = interp1d(data['scat_angle'],data['Polarization'],bounds_error=False)
            p_data_interpolated = p_function(scat_angles)
            if np.all(data['Polarization']<1):
                print('Warning {0:s} has likely polarisation degree expressed in decimal values < 1. CHECK !'.format(experiment_name))
            p_err_function = interp1d(data['scat_angle'],data['Error'],bounds_error=False)
            p_err_data_interpolated = p_err_function(scat_angles)
            
            if scat_angles_interp is not None:
                # we perform an interpolation in this case:
                csv_polar_frac_interp_function = interp1d(csv_table['scat_angle'],csv_table['polar_frac'],bounds_error=False)
                csv_polar_frac_data_interpolated = csv_polar_frac_interp_function(scat_angles)
            else:
                csv_polar_frac_data_interpolated = np.asarray(csv_table['polar_frac'])                

            residuals = (p_data_interpolated-100*csv_polar_frac_data_interpolated)/p_err_data_interpolated
            # print('np.sum(residuals**2)',np.sum(residuals**2))
            # print('np.nansum(residuals**2)',np.nansum(residuals**2))
            chi2 = np.nansum(residuals**2)
            chi2_polar.append(chi2)
            generic_chi2.append(chi2)
            if verbose:
                if np.any(~np.isfinite(residuals)):
                    print('{0:s} chi2 = {1:.1f}. WARNING ! The interpolation range was likely too large for this sample ({2:.1f}deg - {3:.1f}deg)'.format(experiment_name,chi2,np.min(scat_angles),np.max(scat_angles)))
                    # print('WARNING, the interpolation likely failed:',np.asarray(residuals),', chi2={0:.1f}'.format(chi2))
                else:
                    print('{0:s} chi2 = {1:.1f}'.format(experiment_name,chi2))
            
            
            
            if plot:
                fig = plt.figure(1, figsize=[8,6])
                fig.clf()
                ax=plt.subplot(111)
                # plt.close(1)
                # plt.figure(1)
                ax.plot(data['scat_angle'],data['Polarization'],color='tomato',\
                         linestyle='-',label='{0:d} {1:s}'.format(ii,experiment_name))
                ax.fill_between(data['scat_angle'],\
                                 data['Polarization']+data['Error'],\
                                 data['Polarization']-data['Error'],\
                                 alpha=0.3,color='tomato',label='Uncertainty')
                ax.plot(csv_table['scat_angle'],csv_table['polar_frac']*100,color='black',\
                         linestyle='-',label=name)
                ax.fill_between(csv_table['scat_angle'],\
                                 csv_table['polar_frac']*100+csv_table['error_polar_frac']*100,\
                                 csv_table['polar_frac']*100-csv_table['error_polar_frac']*100,\
                                 alpha=0.3,color='grey',label='Uncertainty')
        
                ax.legend(frameon=False,loc='best')
                ax.set_ylabel('Polar. fraction in %')
                ax.set_xlabel('Scattering angle in $^\circ$')
                ax.set_title('{0:s}: $\\chi^2$={1:.1f}'.format(experiment_name,chi2))
                ax.grid()
                plt.tight_layout()
                fig.savefig(path.joinpath(experiment_name+'.pdf'))
                
        # else:
            # In this case, the file read from the database is polarimetric 
            # (respectively in brightness) and the csv table is in brightness 
            # (respectively in polarimetry )
            
            # raise IOError('The data {0:s} contains neither polarimetric nor brightness data. Check the database'.format(experiment_name))
            # chi2 = np.nan
        
    unique_generic_name,unique_inverse_id = np.unique(generic_name,return_inverse=True)
    nb_unique_generic_name = len(unique_generic_name)
    chi2_polar = np.ones((nb_unique_generic_name))*np.nan
    chi2_brightness = np.ones((nb_unique_generic_name))*np.nan

    print()
    print('There are {0:d} comparisons (and chi2) available in polarimery'.format(len(name_polar)))
    print('There are {0:d} comparisons (and chi2) available in brightness'.format(len(name_brightness)))
    
    for i,generic_id in enumerate(unique_inverse_id):
        if generic_isPolar[i]:
            chi2_polar[generic_id] = generic_chi2[i]
        else:
            chi2_brightness[generic_id] = generic_chi2[i]    
    pd_chi2 = pd.DataFrame({'chi2_polar':chi2_polar,'chi2_brightness':chi2_brightness,\
                            'chi2_sum':chi2_polar+chi2_brightness},index=unique_generic_name)
    chi2_filename = path.joinpath('chi2_summary.csv')
    if verbose:
        print()
        print('The smallest 20 chi^2 in polar and brightness are:')
        print(pd_chi2.sort_values('chi2_sum').head(20))
        print()

        print()
        print('The smallest 20 chi^2 in polar only are:')
        print(pd_chi2.sort_values('chi2_polar').head(20))
        print()
        
        print()
        print('The smallest 20 chi^2 in brightness only are:')
        print(pd_chi2.sort_values('chi2_brightness').head(20))
        print()
        
        print('Saving the table of chi2 as {0:s}'.format(str(chi2_filename)))
    pd_chi2.to_csv(path.joinpath('chi2_summary.csv'))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare a brightness or"+\
            " polarisation scattering phase function (SPF) to the PROGRA2 database. The input"+\
            " SPF should be formatted as a csv table with a header row"+\
            " and at least 2 mandatory columns called: phase_angle (or scat_angle) and "+\
            "polar_frac (or spf). The uncertainty error_polar_frac (or error_spf) is not yet taken into account. \n"+\
            "Here are 2 examples to run the code on csv provided with this package: \n: "+\
            "'python progra2.py PROGRA2/examples/debris_disk_SPF_example.csv --scat_angle 70 90 110 --plot' and \n"+\
            "'python progra2.py PROGRA2/examples/typical_comets.csv --scat_angle 70 90 11 --plot' \n")
            
    parser.add_argument('files', type=str, help='csv file(s) you want to process', nargs='*')
    parser.add_argument('-d','--dir', help='directory where output files are saved',type=str,default='.')
    parser.add_argument('-p','--plot', help='plot the comparison between the SPF and PROGRA2 database',\
                        action='store_true') # False by default
    parser.add_argument("-v", "--verbose", help="increase output verbosity (True by default)",\
                    action="store_false") # True by default
    parser.add_argument('--scat_angles', nargs='*',type=float,\
                        help='list of scattering angles (in deg, separated by a white space) to interpolate '+\
                            'the input SPF at specific scattering angles')

    args = parser.parse_args()
    files = args.files
    directory = args.dir
    plot  = args.plot
    verbose = args.verbose
    scat_angles = args.scat_angles
    
    output_path = Path(directory)
    if not output_path.exists():
        raise OSError('The output directory {0:s} does not exist.'.format(directory))
    else:
        if verbose:
            print('The output directory is {0:s}'.format(str(output_path.absolute())))

    # print(scat_angles) 
    # print(type(scat_angles))

    for filename in files:
        csv_table = read_input_csv(filename,verbose=verbose)
        name = Path(filename).stem
        compare_progra2_db(csv_table,output_path,verbose=verbose,plot=plot,\
                           scat_angles_interp=scat_angles,name=name)    