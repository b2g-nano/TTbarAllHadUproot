#!/usr/bin/env python 
# coding: utf-8

from errno import EEXIST
from os import makedirs, path
import mplhep as hep

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