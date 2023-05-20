#!/usr/bin/env python
# coding: utf-8

from coffea import util
import hist
import numpy as np
import itertools
import mplhep as hep
import pandas as pd
from collections import defaultdict
import os
from os import path
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
        
def plotratio2d(numerator, denominator, ax=None, cmap='Blues', cbar=True):
    NumeratorAxes = numerator.axes()
    DenominatorAxes = denominator.axes()
    
    # integer number of bins in this axis #
    NumeratorAxis1_BinNumber = NumeratorAxes[0].size - 3 # Subtract 3 to remove overflow
    NumeratorAxis2_BinNumber = NumeratorAxes[1].size - 3
    
    DenominatorAxis1_BinNumber = DenominatorAxes[0].size - 3 
    DenominatorAxis2_BinNumber = DenominatorAxes[1].size - 3 
    
    if(NumeratorAxis1_BinNumber != DenominatorAxis1_BinNumber 
       or NumeratorAxis2_BinNumber != DenominatorAxis2_BinNumber):
        raise Exception('Numerator and Denominator axes are different sizes; Cannot perform division.')
    else:
        Numerator = numerator.to_hist()
        Denominator = denominator.to_hist()

        ratio = Numerator / Denominator.values()
        
        return hep.hist2dplot(ratio, ax=ax, cmap=cmap, norm=colors.Normalize(0.,1.), cbar=cbar)

#os.chdir('../') # Runs the code from within the working directory without manually changing all directory paths!
maindirectory = os.getcwd() # changes accordingly

# ---- Reiterate categories ---- #
ttagcats = ["at"] #, "0t", "1t", "It", "2t"]
btagcats = ["0b", "1b", "2b"]
ycats = ['cen', 'fwd']

list_of_cats = [ t+b+y for t,b,y in itertools.product( ttagcats, btagcats, ycats) ]
list_of_ints = [6, 7, 8, 9, 10, 11]
catmap = dict(zip(list_of_ints, list_of_cats))

""" ---------------- CREATE RAW MISTAG PLOTS ---------------- """
# ---- Only Use This When LookUp Tables Were Not In Use for Previous Uproot Job (i.e. UseLookUpTables = False) ---- #
# ---- This Creates Mistag plots for every dataset in every category for debugging if necessary or for curiosity ---- #
# ---- Look up tables are a bit more sophisticated and much more useful to the analysis ---- #

# SaveDirectory = maindirectory + '/TTbarAllHadUproot/MistagPlots/'
# DoesDirectoryExist(SaveDirectory) # no need to create the directory several times

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

def LoadDataLUTS(bdiscDirectory, Year, VFP, RemoveContam, ListOfLetters, uploadDir):
    
    contam = ''
    if RemoveContam == True:
        contam = '_ttContaminationRemoved'
    
    for letter in ListOfLetters:
        for icat in list_of_cats:
            df = pd.read_csv(uploadDir+'/LookupTables/' + bdiscDirectory + 'mistag_UL' + str(Year-2000) + VFP + '_JetHT' + '_Data'+contam+'_' + icat[:4] + 'inc.csv')
            luts['JetHT' + str(Year) + letter + '_Data'][icat] = df

    return(luts)

def CreateLUTS(Filesets, Outputs, bdiscDirectory, Year, VFP, ListOfLetters, Save):
    '''
    Filesets        --> Dictionary of datasets
    Outputs         --> Dictionary of uproot outputs from 1st run
    bdiscDirectory  --> string; Directory path for chosen b discriminator
    Year            --> Integer for the year of datasets used in the 1st uproot run
    VFP             --> string; either preVFP or postVFP
    ListOfLetters   --> List; List of the letters corresponding to each run year (if that was chosen for the processor run)
    Save            --> bool; Save mistag rates or not
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
    filestring_prefix = 'UL' + str(Year-2000) + VFP

#     ---------------------------------------
#     PPPPPP  L         OOO   TTTTTTT   SSSSS     
#     P     P L        O   O     T     S          
#     P     P L       O     O    T    S           
#     PPPPPP  L       O     O    T     SSSSS      
#     P       L       O     O    T          S     
#     P       L        O   O     T         S      
#     P       LLLLLLL   OOO      T    SSSSS 
#     ---------------------------------------
    
#     SavePlotDirectory = maindirectory + '/TTbarAllHadUproot/PrelimMistagPlots/' + bdiscDirectory
#     DoesDirectoryExist(SavePlotDirectory)
#     for iset in Filesets:
#         for icat in list_of_ints:
#             # print(iset)
#             # print(icat)
#             title = iset + ' mistag ' + catmap[icat]
#             filename = 'mistag_' + iset + '_' + catmap[icat] + '.' + 'png'
#             # print(Outputs[iset]['numerator'])
#             Numerator = Outputs[iset]['numerator'].project('jetp')
#             Denominator = Outputs[iset]['denominator'].project('jetp')
#             # Numerator = Outputs[iset]['numerator'][iset, icat, sum]
#             # Denominator = Outputs[iset]['denominator'][iset, icat, sum]
#             # print(Numerator)
#             # print(Denominator)
#             # mistag = hist.plotratio(num = Numerator, denom = Denominator,
#             #                         error_opts={'marker': '.', 'markersize': 10., 'color': 'k', 'elinewidth': 1},
#             #                         unc = 'num')
#             mistag = Numerator.plot_ratio(Denominator, rp_uncertainty_type="efficiency")
#             plt.title(title)
#             plt.ylim(bottom = 0, top = 0.12)
#             plt.xlim(left = 100, right = 2500)

#             # ----- Better mistag plots are made in 'TTbarResCoffea_MistagAnalysis-BkgEst' python script ------ #
#             # ---- However, if one wants to save these raw plots, they may uncomment the following 5 lines ---- #
#             # ------------- NOTE: MAYBE THINK OF MAKING A SWITCH FOR PLOTTING/SAVING THESE LATER? ------------- # 

#             #plt.xticks(np.array([0, 500, 600, 700]))
#             #mistag.set_xscale('function', functions=(forward, inverse))
#             #mistag.set_xscale('log')
#             #plt.savefig(SavePlotDirectory+filename, bbox_inches="tight")
#             #print(filename + ' saved')
                    

    #     -------------------------------------------------------------------------------------------
    #     M     M IIIIIII   SSSSS TTTTTTT    A    GGGGGGG     RRRRRR     A    TTTTTTT EEEEEEE   SSSSS     
    #     MM   MM    I     S         T      A A   G           R     R   A A      T    E        S          
    #     M M M M    I    S          T     A   A  G           R     R  A   A     T    E       S           
    #     M  M  M    I     SSSSS     T     AAAAA  G  GGGG     RRRRRR   AAAAA     T    EEEEEEE  SSSSS      
    #     M     M    I          S    T    A     A G     G     R   R   A     A    T    E             S     
    #     M     M    I         S     T    A     A G     G     R    R  A     A    T    E            S      
    #     M     M IIIIIII SSSSS      T    A     A  GGGGG      R     R A     A    T    EEEEEEE SSSSS 
    #     -------------------------------------------------------------------------------------------

    SaveDirectory = maindirectory + '/TTbarAllHadUproot/LookupTables/' + bdiscDirectory
    DoesDirectoryExist(SaveDirectory)

    # ---- Mistag Rates for other samples if curious ---- #
    for iset in Filesets: 
        for icat in list_of_ints:
            filename = 'mistag_' + iset + '_' + catmap[icat] + '.' + 'csv'
            Numerator = Outputs[iset]['numerator'][iset, icat, :]
            Denominator = Outputs[iset]['denominator'][iset, icat, :]
            N_vals = Numerator.view().value
            D_vals = Denominator.view().value
            # print(N_vals)
            # print(D_vals)
            # print()
            mistag_vals = np.where(D_vals > 0, N_vals/D_vals, 0)
            # print(mistag_vals)

            p_vals = [] # Momentum values
            for iden in Numerator.axes['jetp']:
                p_vals.append(iden)
                # print('fileset:  ' + iset)
                # print('category: ' + icat)
                # print('________________________________________________\n')
                # print(p_vals)
                d = {'p': p_vals, 'M(p)': mistag_vals}

                # print("d vals = ", d)
                # print()
                df = pd.DataFrame(data=d)
                luts[iset][catmap[icat]] = df

                # with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
                #     print(df)
                # print('\n')
            if Save:
                df.to_csv(SaveDirectory+filename) # use later to collect bins and weights for re-scaling

    # print(luts)
    # return(luts)
