#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy
import scipy.stats as ss
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from coffea import util
from awkward import JaggedArray
import numpy as np
import itertools
import pandas as pd
from numpy.random import RandomState
from collections import defaultdict
import os
import os.path
from os import path
import matplotlib.pyplot as plt


# In[ ]:


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


# In[ ]:


def DoesDirectoryExist(mypath): #extra precaution (Probably overkill...)
    '''Checks to see if Directory exists before running mkdir_p'''
    
    if path.exists(mypath):
        pass
    else:
        mkdir_p(mypath)


# In[ ]:


maindirectory = os.getcwd() # changes accordingly
print(maindirectory)


# In[ ]:


# ---- Reiterate categories ---- #
ttagcats = ["at"] #, "0t", "1t", "It", "2t"]
btagcats = ["0b", "1b", "2b"]
ycats = ['cen', 'fwd']

list_of_cats = [ t+b+y for t,b,y in itertools.product( ttagcats, btagcats, ycats) ]


# In[ ]:


from Filesets import filesets


# In[ ]:


outputs_unweighted = {}
for name,files in filesets.items():
    outputs_unweighted[name] = util.load('TTbarResCoffea_' + name + '_unweighted_output_partial_2021_dask_run.coffea')
outputs_unweighted


# In[ ]:


""" ---------------- CREATE RAW MISTAG PLOTS ---------------- """
# ---- Only Use This Cell When LookUp Tables Are Not In Use (i.e. UseLookUpTables = False) ---- #
# ---- Mistag plot for every dataset in every category for debugging if necessary or for curiosity ---- #
# ---- Look up tables are a bit more sophisticated and useful to the analysis ---- #

SaveDirectory = maindirectory + '/MistagPlots/'
DoesDirectoryExist(SaveDirectory) # no need to create the directory several times

# Function sqrt(x)
def forward(x):
    return x**(1/2)


def inverse(x):
    return x**2

print(SaveDirectory)
for iset in filesets:
    for icat in list_of_cats:
        print(iset)
        print(icat)
        title = iset + ' mistag ' + icat
        filename = 'mistag_' + iset + '_' + icat + '.' + 'png'
        print(outputs_unweighted[iset]['numerator'])
        Numerator = outputs_unweighted[iset]['numerator'].integrate('anacat', icat).integrate('dataset', iset)
        Denominator = outputs_unweighted[iset]['denominator'].integrate('anacat', icat).integrate('dataset', iset)
        print(Numerator)
        print(Denominator)
        mistag = hist.plotratio(num = Numerator, denom = Denominator,
                                error_opts={'marker': '.', 'markersize': 10., 'color': 'k', 'elinewidth': 1},
                                unc = 'num')
        plt.title(title)
        plt.ylim(bottom = 0, top = 0.12)
        plt.xlim(left = 100, right = 2500)
        
        # ---- Better mistag plots are made in 'TTbarResCoffea_MistagAnalysis-BkgEst' python script ---- #
        # ---- However, if one wants to save these raw plots, they may uncomment the following 5 lines ---- #
        
        #plt.xticks(np.array([0, 500, 600, 700]))
        #mistag.set_xscale('function', functions=(forward, inverse))
        #mistag.set_xscale('log')
        #plt.savefig(SaveDirectory+filename, bbox_inches="tight")
        #print(filename + ' saved')


# In[ ]:


""" ---------------- Scale-Factors for JetHT Data According to Year---------------- """
Nevts2016 = 625516390. # from dasgoclient
Nevts2017 = 410461585. # from dasgoclient
Nevts2018 = 676328827. # from dasgoclient
Nevts = Nevts2016 + Nevts2017 + Nevts2018 # for all three years

if 'JetHT2016_Data' in filesets:
    Nevts2016_sf = Nevts2016/outputs_unweighted['JetHT2016_Data']['cutflow']['all events']
    print(Nevts2016_sf)
if 'JetHT2017_Data' in filesets:
    Nevts2017_sf = Nevts2017/outputs_unweighted['JetHT2017_Data']['cutflow']['all events']
    print(Nevts2017_sf)
if 'JetHT2018_Data' in filesets:
    Nevts2018_sf = Nevts2018/outputs_unweighted['JetHT2018_Data']['cutflow']['all events']
    print(Nevts2018_sf)
if 'JetHT' in filesets:
    Nevts_sf = Nevts / outputs_unweighted['JetHT']['cutflow']['all events']
    print(Nevts_sf)


# In[ ]:


""" ---------------- Luminosities, Cross Sections, Scale-Factors ---------------- """ 
Lum2016 = 35920. # pb^-1 from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable
Lum2017 = 41530.
Lum2018 = 59740.
Lum     = 137190.

ttbar_BR = 0.457 # 0.442 from PDG 2018
ttbar_xs = 1.0   # Monte Carlo already includes xs in event weight!! Otherwise, ttbar_xs = 831.76 * ttbar_BR  pb

ttbar2016_sf = ttbar_xs*Lum2016/(142155064.)
ttbar2017_sf = ttbar_xs*Lum2017/(142155064.)
ttbar2018_sf = ttbar_xs*Lum2018/(142155064.)
ttbar_sf = ttbar_xs*Lum/(142155064.)

print(ttbar2016_sf)
print(ttbar2017_sf)
print(ttbar2018_sf)
print(ttbar_sf)

qcd_xs = 1370000000.0 #pb From https://cms-gen-dev.cern.ch/xsdb
#qcd_sf = qcd_xs*Lum/18455107.


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


""" ---------------- CREATE LOOK UP TABLE .CSV FILES ---------------- """

runLUTS = True # Make separate Directory to place Look-Up Tables and perform ttbar subtraction for mistag weights

def multi_dict(K, type): # definition from https://www.geeksforgeeks.org/python-creating-multidimensional-dictionary/
    if K == 1: 
        return defaultdict(type) 
    else: 
        return defaultdict(lambda: multi_dict(K-1, type))
    
luts = {}
luts = multi_dict(2, str)

if runLUTS : 

    SaveDirectory = maindirectory + '/LookupTables/'
    DoesDirectoryExist(SaveDirectory)
    
    # ---- Check if TTbar simulation was used in previous processor ---- #
    if 'TTbar' in filesets:
        for iset in filesets:
            filename = 'mistag_' + iset + '_' + icat + '.' + 'csv'
            #if iset != 'TTbar' or iset != 'QCD': # if JetHT filesets are found...
            if 'JetHT' in iset:
                print('\t\tfileset: ' + iset + '\n*****************************************************\n')
                for icat in list_of_cats:
                    title = iset + ' mistag ' + icat

                    # ---- Info from TTbar ---- #
                    Numerator_tt = outputs_unweighted['TTbar']['numerator'].integrate('anacat',icat).integrate('dataset','TTbar')
                    Denominator_tt = outputs_unweighted['TTbar']['denominator'].integrate('anacat',icat).integrate('dataset','TTbar')
                    N_vals_tt = Numerator_tt.values()[()] 
                    D_vals_tt = Denominator_tt.values()[()] 

                    # ---- Info from JetHT datasets ---- #
                    Numerator = outputs_unweighted[iset]['numerator'].integrate('anacat',icat).integrate('dataset',iset)
                    Denominator = outputs_unweighted[iset]['denominator'].integrate('anacat',icat).integrate('dataset',iset)
                    N_vals = Numerator.values()[()]
                    D_vals = Denominator.values()[()]

                    # ---- Properly scale chunks of data and ttbar MC according to year of dataset used---- #
                    if '2016' in iset:
                        N_vals *= Nevts2016_sf 
                        D_vals *= Nevts2016_sf
                        N_vals_tt *= ttbar2016_sf
                        D_vals_tt *= ttbar2016_sf
                    elif '2017' in iset:
                        N_vals *= Nevts2017_sf 
                        D_vals *= Nevts2017_sf
                        N_vals_tt *= ttbar2017_sf
                        D_vals_tt *= ttbar2017_sf
                    elif '2018' in iset:
                        N_vals *= Nevts2018_sf 
                        D_vals *= Nevts2018_sf
                        N_vals_tt *= ttbar2018_sf
                        D_vals_tt *= ttbar2018_sf
                    else: # all years
                        N_vals *= Nevts_sf 
                        D_vals *= Nevts_sf
                        N_vals_tt *= ttbar_sf
                        D_vals_tt *= ttbar_sf

                    # ---- Subtract ttbar MC probe momenta from datasets' ---- #
                    N_vals_diff = np.abs(N_vals-N_vals_tt)
                    D_vals_diff = np.abs(D_vals-D_vals_tt)

                    print(N_vals_diff)
                    print(D_vals_diff)
                    print()

                    # ---- Define Mistag values ---- #
                    mistag_vals = np.where(D_vals_diff > 0, N_vals_diff/D_vals_diff, 0)
                    
                    # ---- Define Momentum values ---- #
                    p_vals = []
                    for iden in Numerator.identifiers('jetp'):
                        p_vals.append(iden)

                    # ---- Display and Save Dataframe, df, as Look-up Table ---- #
                    print('fileset:  ' + iset)
                    print('category: ' + icat)
                    print('________________________________________________\n')

                    d = {'p': p_vals, 'M(p)': mistag_vals} # 'data'

                    print("d vals = ", d)
                    print()
                    df = pd.DataFrame(data=d)
                    luts[iset][icat] = df

                    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
                        print(df)
                    print('\n')

                    df.to_csv(SaveDirectory+filename) # use later to collect bins and weights for re-scaling
            else: # If iset is not JetHT...
                for icat in list_of_cats:
                    Numerator = outputs_unweighted[iset]['numerator'].integrate('anacat',icat).integrate('dataset',iset)
                    Denominator = outputs_unweighted[iset]['denominator'].integrate('anacat',icat).integrate('dataset',iset)
                    N_vals = Numerator.values()[()]
                    D_vals = Denominator.values()[()]
                    print(N_vals)
                    print(D_vals)
                    print()
                    mistag_vals = np.where(D_vals > 0, N_vals/D_vals, 0)

                    p_vals = [] # Momentum values
                    for iden in Numerator.identifiers('jetp'):
                        p_vals.append(iden)
                    print('fileset:  ' + iset)
                    print('category: ' + icat)
                    print('________________________________________________\n')
                    d = {'p': p_vals, 'M(p)': mistag_vals}

                    print("d vals = ", d)
                    print()
                    df = pd.DataFrame(data=d)
                    luts[iset][icat] = df

                    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
                        print(df)
                    print('\n')

                    df.to_csv(SaveDirectory+filename) # use later to collect bins and weights for re-scaling

    else: # If iset did not run over 'TTbar' Simulation...
        for iset in filesets:
            print('\t\tfileset: ' + iset + '\n*****************************************************\n')
            for icat in list_of_cats:
                Numerator = outputs_unweighted[iset]['numerator'].integrate('anacat',icat).integrate('dataset',iset)
                Denominator = outputs_unweighted[iset]['denominator'].integrate('anacat',icat).integrate('dataset',iset)
                N_vals = Numerator.values()[()]
                D_vals = Denominator.values()[()]
                print(N_vals)
                print(D_vals)
                print()
                
                mistag_vals = np.where(D_vals > 0, N_vals/D_vals, 0)

                p_vals = []
                for iden in Numerator.identifiers('jetp'):
                    p_vals.append(iden)
                    
                print('fileset:  ' + iset)
                print('category: ' + icat)
                print('________________________________________________\n')
                d = {'p': p_vals, 'M(p)': mistag_vals}

                print("d vals = ", d)
                print()
                df = pd.DataFrame(data=d)
                luts[iset][icat] = df

                with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
                    print(df)
                print('\n')

                df.to_csv(SaveDirectory+filename) # use later to collect bins and weights for re-scaling
            
else : # If runLUTS = False, read in [previously made] Look Up Table csv's
    for iset in filesets:
        print('\t\tfileset: ' + iset + '\n*****************************************************\n')
        for icat in list_of_cats:
            title = iset + ' mistag ' + icat
            filename = 'mistag_' + iset + '_' + icat + '.' + 'csv'
            luts[iset][icat] = pd.read_csv(filename)
print(luts)


# In[ ]:


#!jupyter nbconvert --to script TTbarResLookUpTables.ipynb


# In[ ]:




