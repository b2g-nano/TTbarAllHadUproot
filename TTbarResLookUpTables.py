#!/usr/bin/env python
# coding: utf-8

from coffea import hist
from coffea import util
import numpy as np
import itertools
import pandas as pd
from collections import defaultdict
import os
from os import path
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------- Imported into TTbarCoffeOutputs to get the mistag rates in the form of look up tables for the desired datasets -------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

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

def DoesDirectoryExist(mypath): #extra precaution (Probably overkill...)
    '''Checks to see if Directory exists before running mkdir_p'''
    
    if path.exists(mypath):
        pass
    else:
        mkdir_p(mypath)

maindirectory = os.getcwd() # changes accordingly

# ---- Reiterate categories ---- #
ttagcats = ["at"] #, "0t", "1t", "It", "2t"]
btagcats = ["0b", "1b", "2b"]
ycats = ['cen', 'fwd']

list_of_cats = [ t+b+y for t,b,y in itertools.product( ttagcats, btagcats, ycats) ]

""" ---------------- CREATE RAW MISTAG PLOTS ---------------- """
# ---- Only Use This When LookUp Tables Were Not In Use for Previous Uproot Job (i.e. UseLookUpTables = False) ---- #
# ---- This Creates Mistag plots for every dataset in every category for debugging if necessary or for curiosity ---- #
# ---- Look up tables are a bit more sophisticated and much more useful to the analysis ---- #

SaveDirectory = maindirectory + '/TTbarAllHadUproot/MistagPlots/'
DoesDirectoryExist(SaveDirectory) # no need to create the directory several times

# Function sqrt(x)
def forward(x):
    return x**(1/2)

def inverse(x):
    return x**2

import warnings
warnings.filterwarnings("ignore")

""" ---------------- CREATE LOOK UP TABLE .CSV FILES ---------------- """

def multi_dict(K, type): # definition from https://www.geeksforgeeks.org/python-creating-multidimensional-dictionary/
    if K == 1: 
        return defaultdict(type) 
    else: 
        return defaultdict(lambda: multi_dict(K-1, type))
    
luts = {}
luts = multi_dict(2, str) #Annoying, but necessary definition of the dictionary

def CreateLUTS(Filesets, Outputs, Year, VFP, RemoveContam, Save):
    '''
    Filesets     --> Dictionary of datasets
    Outputs      --> Dictionary of uproot outputs from 1st run
    Year         --> Integer for the year of datasets used in the 1st uproot run
    VFP          --> string; either preVFP or postVFP
    RemoveContam --> bool; Remove the ttbar contamination from mistag when selecting --mistag option in TTbarResCoffeaOutputs.py
    Save         --> bool; Save mistag rates or not
    '''
    
#     -------------------------------------------------------------------
#     GGGGGGG EEEEEEE TTTTTTT     FFFFFFF IIIIIII L       EEEEEEE   SSSSS     
#     G       E          T        F          I    L       E        S          
#     G       E          T        F          I    L       E       S           
#     G  GGGG EEEEEEE    T        FFFFFFF    I    L       EEEEEEE  SSSSS      
#     G     G E          T        F          I    L       E             S     
#     G     G E          T        F          I    L       E            S      
#      GGGGG  EEEEEEE    T        F       IIIIIII LLLLLLL EEEEEEE SSSSS
#     -------------------------------------------------------------------
        
    
    outputs_unweighted = {}
    if Year != 0:
        filestring_prefix = 'UL' + str(Year-2000) + VFP + '_'
        filestring_prefix_data = str(Year)
    else:
        filestring_prefix = ''
        filestring_prefix_data = ''

#     if loadAllFiles: # Use everything that is in filesets,
#         for name,files in Filesets.items():
#             outputs_unweighted[name] = util.load('TTbarAllHadUproot/CoffeaOutputs/UnweightedOutputs/TTbarResCoffea_' + name + '_unweighted_output.coffea')
#         Outputs = outputs_unweighted

#     ---------------------------------------
#     PPPPPP  L         OOO   TTTTTTT   SSSSS     
#     P     P L        O   O     T     S          
#     P     P L       O     O    T    S           
#     PPPPPP  L       O     O    T     SSSSS      
#     P       L       O     O    T          S     
#     P       L        O   O     T         S      
#     P       LLLLLLL   OOO      T    SSSSS 
#     ---------------------------------------
    
    for iset in Filesets:
        for icat in list_of_cats:
            print(iset)
            print(icat)
            title = iset + ' mistag ' + icat
            filename = 'mistag_' + iset + '_' + icat + '.' + 'png'
            print(Outputs[iset]['numerator'])
            Numerator = Outputs[iset]['numerator'].integrate('anacat', icat).integrate('dataset', iset)
            Denominator = Outputs[iset]['denominator'].integrate('anacat', icat).integrate('dataset', iset)
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

#     outputs_unweighted
    
#     -------------------------------------------------------    
#       SSSSS   CCCC     A    L       IIIIIII N     N GGGGGGG     
#      S       C        A A   L          I    NN    N G           
#     S       C        A   A  L          I    N N   N G           
#      SSSSS  C        AAAAA  L          I    N  N  N G  GGGG     
#           S C       A     A L          I    N   N N G     G     
#          S   C      A     A L          I    N    NN G     G     
#     SSSSS     CCCC  A     A LLLLLLL IIIIIII N     N  GGGGG  
#     -------------------------------------------------------   
    
    """ ---------------- Scale-Factors for JetHT Data According to Year---------------- """
    Nevts2016 = 625502676. #625516390. # from dasgoclient
    Nevts2017 = 410461585. # from dasgoclient
    Nevts2018 = 676328827. # from dasgoclient
    Nevts = Nevts2016 + Nevts2017 + Nevts2018 # for all three years
    
    Nevts2016_sf = 0.
    Nevts2017_sf = 0.
    Nevts2018_sf = 0.
    Nevts_sf = 0.
    
    if 'JetHT2016_Data' in Filesets:
        Nevts2016_sf = Nevts2016/Outputs['JetHT2016_Data']['cutflow']['all events']
    if 'JetHT2017_Data' in Filesets:
        Nevts2017_sf = Nevts2017/Outputs['JetHT2017_Data']['cutflow']['all events']
    if 'JetHT2018_Data' in Filesets:
        Nevts2018_sf = Nevts2018/Outputs['JetHT2018_Data']['cutflow']['all events']
    if 'JetHT_Data' in Filesets:
        Nevts_sf = Nevts / Outputs['JetHT_Data']['cutflow']['all events']

    """ ---------------- Luminosities, Cross Sections, Scale-Factors ---------------- """ 
    Lum2016 = 35920. # pb^-1 from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable
    Lum2017 = 41530.
    Lum2018 = 59740.
    Lum     = 137190. # total Luminosity of all years

    ttbar_BR = 0.457 # 0.442 from PDG 2018
    ttbar_xs = 1.0   # Monte Carlo already includes xs in event weight!! Otherwise, ttbar_xs = 831.76 * ttbar_BR  pb
    
    ttbar2016_sf = 0.
    ttbar2017_sf = 0.
    ttbar2018_sf = 0.
    ttbar_sf = 0.

    if 'UL16' and 'TTbar' in Outputs.items():
        ttbar2016_sf = ttbar_xs*ttbar_BR*Lum2016/Outputs['UL16'+VFP+'_TTbar']['cutflow']['all events']
    if 'UL17' and 'TTbar' in Outputs.items():
        ttbar2017_sf = ttbar_xs*ttbar_BR*Lum2017/Outputs['UL17'+VFP+'_TTbar']['cutflow']['all events']
    if 'UL18' and 'TTbar' in Outputs.items():
        ttbar2018_sf = ttbar_xs*ttbar_BR*Lum2018/Outputs['UL18'+VFP+'_TTbar']['cutflow']['all events']
    if ('TTbar' in Outputs.items()) and (Year == 0):
        ttbar_sf = ttbar_xs*ttbar_BR*Lum/Outputs[VFP+'_TTbar']['cutflow']['all events']

    qcd_xs = 1370000000.0 #pb From https://cms-gen-dev.cern.ch/xsdb
    #qcd_sf = qcd_xs*Lum/18455107.
    
#     -------------------------------------------------------------------------------------------
#     M     M IIIIIII   SSSSS TTTTTTT    A    GGGGGGG     RRRRRR     A    TTTTTTT EEEEEEE   SSSSS     
#     MM   MM    I     S         T      A A   G           R     R   A A      T    E        S          
#     M M M M    I    S          T     A   A  G           R     R  A   A     T    E       S           
#     M  M  M    I     SSSSS     T     AAAAA  G  GGGG     RRRRRR   AAAAA     T    EEEEEEE  SSSSS      
#     M     M    I          S    T    A     A G     G     R   R   A     A    T    E             S     
#     M     M    I         S     T    A     A G     G     R    R  A     A    T    E            S      
#     M     M IIIIIII SSSSS      T    A     A  GGGGG      R     R A     A    T    EEEEEEE SSSSS 
#     -------------------------------------------------------------------------------------------
    
    SaveDirectory = maindirectory + '/TTbarAllHadUproot/LookupTables/'
    DoesDirectoryExist(SaveDirectory)

    # ---- Check if TTbar simulation was used in previous processor ---- #
    for iset in Filesets: 
        if ('JetHT' in iset) and any('TTbar' in i for i in Outputs) and RemoveContam:
            print('\t\tfileset: ' + iset + 'With Contamination Removed!\n*****************************************************\n')
            for icat in list_of_cats:
                filename = 'mistag_' + iset + '_ttContaminationRemoved_' + icat + '.' + 'csv'
                title = iset + ' mistag ' + icat

                # ---- Info from TTbar ---- #
                Numerator_tt = Outputs[filestring_prefix+'TTbar']['numerator'].integrate('anacat',icat).integrate('dataset',filestring_prefix+'TTbar')
                Denominator_tt = Outputs[filestring_prefix+'TTbar']['denominator'].integrate('anacat',icat).integrate('dataset',filestring_prefix+'TTbar')
                N_vals_tt = Numerator_tt.values()[()] 
                D_vals_tt = Denominator_tt.values()[()] 

                # ---- Info from JetHT datasets ---- #
                Numerator = Outputs[iset]['numerator'].integrate('anacat',icat).integrate('dataset',iset)
                Denominator = Outputs[iset]['denominator'].integrate('anacat',icat).integrate('dataset',iset)
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
                print('fileset:  ' + iset + '_ttContaminationRemoved')
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
                if Save:
                    df.to_csv(SaveDirectory+filename) # use later to collect bins and weights for re-scaling
        else: # Make mistag rate of any dataset that was run in the 1st uproot job
            print('\t\tfileset: ' + iset + '\n*****************************************************\n')
            for icat in list_of_cats:
                filename = 'mistag_' + iset + '_' + icat + '.' + 'csv'
                Numerator = Outputs[iset]['numerator'].integrate('anacat',icat).integrate('dataset',iset)
                Denominator = Outputs[iset]['denominator'].integrate('anacat',icat).integrate('dataset',iset)
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
                if Save:
                    df.to_csv(SaveDirectory+filename) # use later to collect bins and weights for re-scaling

# else : # If runLUTS = False, read in [previously made] Look Up Table csv's
#     print("\n\nLoading Previously Made Look Up Tables for Mistag Rate\n\nDoes not Correspond to Test Files...\n\n")
#     if path.exists('TTbarAllHadUproot/LookupTables/JetHT'+filestring_prefix_data+'_Data_ttContaminationRemoved_at*'):
#         datafile = 'JetHT' + filestring_prefix_data + '_Data_ttContaminationRemoved'
#         for icat in list_of_cats:
#             title = datafile + ' mistag ' + icat
#             filename = 'mistag_' + datafile + '_' + icat + '.' + 'csv'
#             luts[iset][icat] = pd.read_csv('TTbarAllHadUproot/LookupTables/'+filename)
#     for iset in Filesets:
#         print('\t\tfileset: ' + iset + '\n*****************************************************\n')
#         for icat in list_of_cats:
#             title = iset + ' mistag ' + icat
#             filename = 'mistag_' + iset + '_' + icat + '.' + 'csv'
#             luts[iset][icat] = pd.read_csv('TTbarAllHadUproot/LookupTables/'+filename)
    print(luts)
    return(luts)

#!jupyter nbconvert --to script TTbarResLookUpTables.ipynb
