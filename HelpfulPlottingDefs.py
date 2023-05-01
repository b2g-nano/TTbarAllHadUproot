#!/usr/bin/env python 
# coding: utf-8

from errno import EEXIST
from os import makedirs, path
import mplhep as hep
import matplotlib.pyplot as plt

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def DoesDirectoryExist(mypath): #extra precaution (Probably overkill...)
    '''Checks to see if Directory exists before running mkdir_p'''
    import os.path
    
    if path.exists(mypath):
        pass
    else:
        mkdir_p(mypath)
        
def ConvertLabelToInt(mapping, str_label):
    for intkey, string in mapping.items():
        if str_label == string:
            return intkey
        
def plotratio(numerator, denominator, ax=None, histtype='errorbar', marker='.', markersize=5., color='k', alpha=0.1):
    NumeratorAxes = numerator.axes
    DenominatorAxes = denominator.axes
    
    # integer number of bins in this axis #
    NumeratorAxis1_BinNumber = NumeratorAxes[0].size - 3 # Subtract 3 to remove overflow
    
    DenominatorAxis1_BinNumber = DenominatorAxes[0].size - 3 
    
    if(NumeratorAxis1_BinNumber != DenominatorAxis1_BinNumber):
        raise Exception('Numerator and Denominator axes are different sizes; Cannot perform division.')
    # else:
    #     Numerator = numerator.to_hist()
    #     Denominator = denominator.to_hist()
        
    ratio = numerator / denominator.values()
    
    if histtype == 'errorbar':
        return hep.histplot(ratio, ax=ax, histtype=histtype, marker=marker, markersize=markersize, color=color)
    elif histtype == 'fill':
        return hep.histplot(ratio, ax=ax, histtype=histtype, color=color, alpha=alpha, lw=5.)
    else:
        return hep.histplot(ratio, ax=ax, histtype=histtype, color=color)
    
def printColorText(text, color): # both the input text and the color desired are input as strings
    whichcolor = {
                    'red':'\033[91m' + text,
                    'yellow':'\033[93m' + text,
                    'green':'\033[92m' + text,
                    'blue':'\033[96m' + text,
                    'indigo':'\033[94m' + text,
                    'violet':'\033[95m' + text,
                }
    print(whichcolor.get(color) + '\033[90m')
    # The added string on the end resets the default colored text to black #
    
def AnalysisAxes(HistName, axis):
    CMSx, CMSy = 0.01, 0.98 # Position of CMS Preliminary label
    
    if 'ttbarmass' in HistName:
        axis.set_xlim(950,6000)
        CMSx = 0.14 
        CMSy = 0.98
    elif 'jetpt' in HistName:
        axis.set_xlim(400,2000)
        CMSx = 0.26 
        CMSy = 0.98
    elif 'jeteta' in HistName:
        axis.set_xlim(-2.3,2.3)
    elif 'jetphi' in HistName:
        axis.set_xlim(-3.14, 3.14)
    elif 'jety' in HistName:
        axis.set_xlim(-3., 3.)
    elif 'jetdy' in HistName:
        axis.set_xlim(0., 5.)
    elif 'probept' in HistName:
        axis.set_xlim(400., 2000.)
        CMSx = 0.14 
        CMSy = 0.98
    elif 'probep' in HistName:
        axis.set_xlim(400., 7000.)  
        CMSx = 0.14 
        CMSy = 0.98

def AnalysisLabels(Luminosity, axis):
    CMSx, CMSy = 0.01, 0.98 # Position of CMS Preliminary label
    Lint = str(Luminosity*.001) # Integrated Luminosity
    
    lumi = plt.text(1.15, 1.07, "L = " + Lint[:6] + " fb$^{-1}$",
            fontsize='x-large',
            horizontalalignment='right',
            verticalalignment='top',
            transform=axis.transAxes
           )
    CMS = plt.text(CMSx, CMSy, 'CMS Preliminary',
            fontsize='x-large',
            horizontalalignment='left',
            verticalalignment='top',
            transform=axis.transAxes
           )
    coffea = plt.text(1.00, 0.85, u"☕",
              fontsize=50,
              horizontalalignment='left',
              verticalalignment='bottom',
              transform=axis.transAxes
             )
