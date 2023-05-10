#!/usr/bin/env python 
# coding: utf-8

from errno import EEXIST
from os import makedirs, path
import mplhep as hep
import matplotlib.pyplot as plt
from hist.intervals import ratio_uncertainty


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
    
def plotefficiency(numerator, denominator, ax=None, histtype='errorbar', marker='.', markersize=5., color='k', alpha=0.1, xerr=0.):
    NumeratorAxes = numerator.axes
    DenominatorAxes = denominator.axes
    
    # integer number of bins in this axis #
    NumeratorAxis1_BinNumber = NumeratorAxes[0].size - 3 # Subtract 3 to remove overflow
    
    DenominatorAxis1_BinNumber = DenominatorAxes[0].size - 3 
    
    if(NumeratorAxis1_BinNumber != DenominatorAxis1_BinNumber):
        raise Exception('Numerator and Denominator axes are different sizes; Cannot perform division.')
        
    ratio = numerator / denominator.values()

    err_up, err_down = ratio_uncertainty(numerator.values(), denominator.values(), 'efficiency')
    yerror = [err_up, err_down]

#     for ra, u, d in zip(ratio, err_up, err_down):
#         print(f'{ra} +{u} -{d}\n')
#     print('=================================================\n')
    
    if histtype == 'errorbar':
        return hep.histplot(ratio, ax=ax, histtype=histtype, marker=marker, markersize=markersize, color=color, 
                            xerr=xerr, yerr=yerror)
    elif histtype == 'fill':
        return hep.histplot(ratio, ax=ax, histtype=histtype, color=color, alpha=alpha, lw=5., 
                           yerr=yerror)
    else:
        return hep.histplot(ratio, ax=ax, histtype=histtype, color=color, yerr=yerror)
    
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
    
def GetMistagInfo(JetHT, btag, year, VFP):
    '''
        JetHT --> dict, coffea output of jetht loaded from LoadData.py script
        btag --> int, number of btags (0, 1, 2)
        year --> int, run year        (2016, 2017, 2018)
        VFP  --> str, APV or no APV   (preVFP, postVFP) 
    '''
    
    if btag == 0:
        cen = 6
        fwd = 7
    elif btag == 1:
        cen = 8
        fwd = 9
    elif btag == 2:
        cen = 10
        fwd = 11
    else:
        print('No available number of btags selected')
        return 0
        
    if year == 2016 and VFP == 'preVFP':
        
        JetHT_str = 'UL16preVFP_JetHTB_Data'
        
        Num_cen = JetHT['B_preVFP']['numerator'][JetHT_str, cen, :]
        Denom_cen = JetHT['B_preVFP']['denominator'][JetHT_str, cen, :]
        Num_fwd = JetHT['B_preVFP']['numerator'][JetHT_str, fwd, :]
        Denom_fwd = JetHT['B_preVFP']['denominator'][JetHT_str, fwd, :]
        
        for Era in ['C', 'D', 'E', 'F']: #exclude B because histogram is initialized with B era
            
            JetHT_str = f'UL16preVFP_JetHT{Era}_Data'
            
            Num_cen += JetHT[Era+'_preVFP']['numerator'][JetHT_str, cen, :]
            Denom_cen += JetHT[Era+'_preVFP']['denominator'][JetHT_str, cen, :]
            Num_fwd += JetHT[Era+'_preVFP']['numerator'][JetHT_str, fwd, :]
            Denom_fwd += JetHT[Era+'_preVFP']['denominator'][JetHT_str, fwd, :]
            
    elif year == 2016 and VFP == 'postVFP':
        
        JetHT_str = 'UL16postVFP_JetHTF_Data'
        
        Num_cen = JetHT['F_postVFP']['numerator'][JetHT_str, cen, :]
        Denom_cen = JetHT['F_postVFP']['denominator'][JetHT_str, cen, :]
        Num_fwd = JetHT['F_postVFP']['numerator'][JetHT_str, fwd, :]
        Denom_fwd = JetHT['F_postVFP']['denominator'][JetHT_str, fwd, :]
        
        for Era in ['F', 'G', 'H']: #exclude F because histogram is initialized with F era
            
            JetHT_str = f'UL16postVFP_JetHT{Era}_Data'
            
            Num_cen += JetHT[Era+'_postVFP']['numerator'][JetHT_str, cen, :]
            Denom_cen += JetHT[Era+'_postVFP']['denominator'][JetHT_str, cen, :]
            Num_fwd += JetHT[Era+'_postVFP']['numerator'][JetHT_str, fwd, :]
            Denom_fwd += JetHT[Era+'_postVFP']['denominator'][JetHT_str, fwd, :]
            
    Num_inc = Num_cen + Num_fwd
    Denom_inc = Denom_cen + Denom_fwd

    Output = {
        'Num':   Num_inc,
        'Denom': Denom_inc
    }
    
    return Output
    
def GetMistagInfoCR(TTbar, JetHT, btag, year, VFP, scale): # CR --> "Contamination Removed"
    '''
        TTbar --> dict, coffea output of ttbar loaded from LoadMC.py script
        JetHT --> dict, coffea output of jetht loaded from LoadData.py script
        btag --> int, number of btags (0, 1, 2)
        year --> int, run year        (2016, 2017, 2018)
        VFP  --> str, APV or no APV   (preVFP, postVFP) 
        scale --> list, collection of ttbar scale factors; first sf for 700-1000 and the other for 1000-Inf
    '''
    
    str1 = '700_1000'
    str2 = '1000_Inf'
    stry = 'UL'+str(year-2000)
    
    if btag == 0:
        cen = 6
        fwd = 7
    elif btag == 1:
        cen = 8
        fwd = 9
    elif btag == 2:
        cen = 10
        fwd = 11
    else:
        print('No available number of btags selected')
        return 0
    
    NumTT1cen = TTbar[f'{str1}_{VFP}']['numerator'][f'{stry}{VFP}_TTbar_{str1}', cen, :]
    DenomTT1cen = TTbar[f'{str1}_{VFP}']['denominator'][f'UL16{VFP}_TTbar_{str1}', cen, :]
    NumTT1fwd = TTbar[f'{str1}_{VFP}']['numerator'][f'UL16{VFP}_TTbar_{str1}', fwd, :]               
    DenomTT1fwd = TTbar[f'{str1}_{VFP}']['denominator'][f'UL16{VFP}_TTbar_{str1}', fwd, :]

    NumTT1_inc = NumTT1cen + NumTT1fwd
    DenomTT1_inc = DenomTT1cen + DenomTT1fwd

    NumTT2cen = TTbar[f'{str2}_{VFP}']['numerator'][f'{stry}{VFP}_TTbar_{str2}', cen, :]
    DenomTT2cen = TTbar[f'{str2}_{VFP}']['denominator'][f'UL16{VFP}_TTbar_{str2}', cen, :]
    NumTT2fwd = TTbar[f'{str2}_{VFP}']['numerator'][f'UL16{VFP}_TTbar_{str2}', fwd, :]               
    DenomTT2fwd = TTbar[f'{str2}_{VFP}']['denominator'][f'UL16{VFP}_TTbar_{str2}', fwd, :]

    NumTT2_inc = NumTT2cen + NumTT2fwd
    DenomTT2_inc = DenomTT2cen + DenomTT2fwd

    NumTT1_inc *= (-scale[0])
    DenomTT1_inc *= (-scale[0])

    NumTT2_inc *= (-scale[1])
    DenomTT2_inc *= (-scale[1])

    NumTT_inc = NumTT1_inc + NumTT2_inc
    DenomTT_inc = DenomTT1_inc + DenomTT2_inc
        
    if year == 2016 and VFP == 'preVFP':
        
        JetHT_str = 'UL16preVFP_JetHTB_Data'
        
        Num_cen = JetHT['B_preVFP']['numerator'][JetHT_str, cen, :]
        Denom_cen = JetHT['B_preVFP']['denominator'][JetHT_str, cen, :]
        Num_fwd = JetHT['B_preVFP']['numerator'][JetHT_str, fwd, :]
        Denom_fwd = JetHT['B_preVFP']['denominator'][JetHT_str, fwd, :]
        
        for Era in ['C', 'D', 'E', 'F']: #exclude B because histogram is initialized with B era
            
            JetHT_str = f'UL16preVFP_JetHT{Era}_Data'
            
            Num_cen += JetHT[Era+'_preVFP']['numerator'][JetHT_str, cen, :]
            Denom_cen += JetHT[Era+'_preVFP']['denominator'][JetHT_str, cen, :]
            Num_fwd += JetHT[Era+'_preVFP']['numerator'][JetHT_str, fwd, :]
            Denom_fwd += JetHT[Era+'_preVFP']['denominator'][JetHT_str, fwd, :]
            
    elif year == 2016 and VFP == 'postVFP':
        
        JetHT_str = 'UL16postVFP_JetHTF_Data'
        
        Num_cen = JetHT['F_postVFP']['numerator'][JetHT_str, cen, :]
        Denom_cen = JetHT['F_postVFP']['denominator'][JetHT_str, cen, :]
        Num_fwd = JetHT['F_postVFP']['numerator'][JetHT_str, fwd, :]
        Denom_fwd = JetHT['F_postVFP']['denominator'][JetHT_str, fwd, :]
        
        for Era in ['F', 'G', 'H']: #exclude F because histogram is initialized with F era
            
            JetHT_str = f'UL16postVFP_JetHT{Era}_Data'
            
            Num_cen += JetHT[Era+'_postVFP']['numerator'][JetHT_str, cen, :]
            Denom_cen += JetHT[Era+'_postVFP']['denominator'][JetHT_str, cen, :]
            Num_fwd += JetHT[Era+'_postVFP']['numerator'][JetHT_str, fwd, :]
            Denom_fwd += JetHT[Era+'_postVFP']['denominator'][JetHT_str, fwd, :]
            
    Num_inc = Num_cen + Num_fwd + NumTT_inc
    Denom_inc = Denom_cen + Denom_fwd + DenomTT_inc

    Output = {
        'Num':   Num_inc,
        'Denom': Denom_inc
    }
    
    return Output
        