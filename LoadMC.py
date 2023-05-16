#!/usr/bin/env python 
# coding: utf-8

from coffea import util
import numpy as np

def DM_Unweighted(btag, year):
    
    Output = {}
    
    if btag == '':
        od = '_oldANdisc'
    else:
        od = ''
    
    if year == 2016:
        
        for Mass in range(1000, 5500, 500):
            datastr = f'TTbarRes_0l_UL16preVFP_DM{str(Mass)}'
            PathToFile = f'CoffeaOutputsForCombine/Coffea_FirstRun/ZprimeDMToTTbar/{btag}/{str(year)}/APV/{datastr}{od}.coffea'
            Output[str(Mass)+'_preVFP'] = util.load(PathToFile)
            
        for Mass in range(1000, 5500, 500):
            datastr = f'TTbarRes_0l_UL16postVFP_DM{str(Mass)}'
            PathToFile = f'CoffeaOutputsForCombine/Coffea_FirstRun/ZprimeDMToTTbar/{btag}/{str(year)}/noAPV/{datastr}{od}.coffea'
            Output[str(Mass)+'_postVFP'] = util.load(PathToFile)
            
    return (Output)
            
def DM_Weighted(btag, year, contam, unc, mr):
    '''
        btag --> string ('', 'LooseBTag', 'MediumBTag'); Select btag directory files are saved in
        year --> integer (2016, 2017, 2018); Select run year
        contam --> bool (True, False); Is the mistag contamination removed? Yay or neah
        unc --> string; Select the uncertainty type (e.g. btagUnc_central)
        mr --> bool (True, False); Mistag Rate Weights Only? Yay or neah
    '''
    
    Output = {}
    
    if btag == '':
        od = '_oldANdisc'
    else:
        od = ''

    if contam == True:
        Contam = '_ttbarContamRemoved'
    else:
        Contam = ''

    if mr == True:
        MR = '_MistagOnly'
    else:
        MR = ''
    
    if year == 2016:

        for Mass in range(1000, 5500, 500):
            datastr = f'TTbarRes_0l_UL16preVFP_DM{str(Mass)}_weighted{Contam}{unc}{MR}'
            PathToFile = f'CoffeaOutputsForCombine/Coffea_SecondRun/ZprimeDMToTTbar/{btag}/{str(year)}/APV/{datastr}{od}.coffea'
            Output[str(Mass)+'_preVFP'] = util.load(PathToFile)

        for Mass in range(1000, 5500, 500):
            datastr = f'TTbarRes_0l_UL16postVFP_DM{str(Mass)}_weighted{Contam}{unc}{MR}'
            PathToFile = f'CoffeaOutputsForCombine/Coffea_SecondRun/ZprimeDMToTTbar/{btag}/{str(year)}/noAPV/{datastr}{od}.coffea'
            Output[str(Mass)+'_postVFP'] = util.load(PathToFile)
        
    return(Output)

def RSGluon_Unweighted(btag, year):
    
    Output = {}
    
    if btag == '':
        od = '_oldANdisc'
    else:
        od = ''
    
    if year == 2016:
        
        for Mass in range(1000, 5500, 500):
            datastr = f'TTbarRes_0l_UL16preVFP_RSGluon{str(Mass)}'
            PathToFile = f'CoffeaOutputsForCombine/Coffea_FirstRun/RSGluonToTT/{btag}/{str(year)}/APV/{datastr}{od}.coffea'
            Output[str(Mass)+'_preVFP'] = util.load(PathToFile)
            
        for Mass in range(1000, 5500, 500):
            datastr = f'TTbarRes_0l_UL16postVFP_RSGluon{str(Mass)}'
            PathToFile = f'CoffeaOutputsForCombine/Coffea_FirstRun/RSGluonToTT/{btag}/{str(year)}/noAPV/{datastr}{od}.coffea'
            Output[str(Mass)+'_postVFP'] = util.load(PathToFile)
            
    return(Output)
            
def RSGluon_Weighted(btag, year, contam, unc, mr):
    '''
        btag --> string ('', 'LooseBTag', 'MediumBTag'); Select btag directory files are saved in
        year --> integer (2016, 2017, 2018); Select run year
        contam --> bool (True, False); Is the mistag contamination removed? Yay or neah
        unc --> string; Select the uncertainty type (e.g. btagUnc_central)
        mr --> bool (True, False); Mistag Rate Weights Only? Yay or neah
    '''
    
    Output = {}
    
    if btag == '':
        od = '_oldANdisc'
    else:
        od = ''

    if contam == True:
        Contam = '_ttbarContamRemoved'
    else:
        Contam = ''

    if mr == True:
        MR = '_MistagOnly'
    else:
        MR = ''
    
    if year == 2016:

        for Mass in range(1000, 5500, 500):
            datastr = f'TTbarRes_0l_UL16preVFP_RSGluon{str(Mass)}_weighted{Contam}{unc}{MR}'
            PathToFile = f'CoffeaOutputsForCombine/Coffea_SecondRun/RSGluonToTT/{btag}/{str(year)}/APV/{datastr}{od}.coffea'
            Output[str(Mass)+'_preVFP'] = util.load(PathToFile)

        for Mass in range(1000, 5500, 500):
            datastr = f'TTbarRes_0l_UL16postVFP_RSGluon{str(Mass)}_weighted{Contam}{unc}{MR}'
            PathToFile = f'CoffeaOutputsForCombine/Coffea_SecondRun/RSGluonToTT/{btag}/{str(year)}/noAPV/{datastr}{od}.coffea'
            Output[str(Mass)+'_postVFP'] = util.load(PathToFile)
        
    return(Output)

def TTbar_Unweighted(btag, year):
    
    Output = {}
    
    if btag == '':
        od = '_oldANdisc'
    else:
        od = ''
        
    if year == 2016:
        
        datastr11 = f'TTbarRes_0l_UL16preVFP_TTbar_700_1000'
        datastr12 = f'TTbarRes_0l_UL16postVFP_TTbar_700_1000'
        datastr21 = f'TTbarRes_0l_UL16preVFP_TTbar_1000_Inf'
        datastr22 = f'TTbarRes_0l_UL16postVFP_TTbar_1000_Inf'
        
        PathToFile = f'CoffeaOutputsForCombine/Coffea_FirstRun/TT/{btag}/{str(year)}'
        
        Output['700_1000_preVFP'] = util.load(f'{PathToFile}/APV/{datastr11}{od}.coffea')
        Output['700_1000_postVFP'] = util.load(f'{PathToFile}/noAPV/{datastr12}{od}.coffea')
        Output['1000_Inf_preVFP'] = util.load(f'{PathToFile}/APV/{datastr21}{od}.coffea')
        Output['1000_Inf_postVFP'] = util.load(f'{PathToFile}/noAPV/{datastr22}{od}.coffea')
        
    return(Output)

def TTbar_Weighted(btag, year, contam, unc, mr):
    
    '''
        btag --> string ('', 'LooseBTag', 'MediumBTag'); Select btag directory files are saved in
        year --> integer (2016, 2017, 2018); Select run year
        contam --> bool (True, False); Is the mistag contamination removed? Yay or neah
        unc --> string; Select the uncertainty type (e.g. btagUnc_central)
        mr --> bool (True, False); Mistag Rate Weights Only? Yay or neah
    '''
    
    Output = {}
    
    if btag == '':
        od = '_oldANdisc'
    else:
        od = ''

    if contam == True:
        Contam = '_ttbarContamRemoved'
    else:
        Contam = ''

    if mr == True:
        MR = '_MistagOnly'
    else:
        MR = ''
        
    if year == 2016:
        
        datastr11 = f'TTbarRes_0l_UL16preVFP_TTbar_700_1000_weighted{Contam}{unc}{MR}'
        datastr12 = f'TTbarRes_0l_UL16postVFP_TTbar_700_1000_weighted{Contam}{unc}{MR}'
        datastr21 = f'TTbarRes_0l_UL16preVFP_TTbar_1000_Inf_weighted{Contam}{unc}{MR}'
        datastr22 = f'TTbarRes_0l_UL16postVFP_TTbar_1000_Inf_weighted{Contam}{unc}{MR}'
        
        PathToFile = f'CoffeaOutputsForCombine/Coffea_SecondRun/TT/{btag}/{str(year)}'
        
        Output['700_1000_preVFP'] = util.load(f'{PathToFile}/APV/{datastr11}{od}.coffea')
        Output['700_1000_postVFP'] = util.load(f'{PathToFile}/noAPV/{datastr12}{od}.coffea')
        Output['1000_Inf_preVFP'] = util.load(f'{PathToFile}/APV/{datastr21}{od}.coffea')
        Output['1000_Inf_postVFP'] = util.load(f'{PathToFile}/noAPV/{datastr22}{od}.coffea')
        
    return(Output)

def QCD_Unweighted(btag, year):
    
    Output = {}
    
    if btag == '':
        od = '_oldANdisc'
    else:
        od = ''
        
    if year == 2016:
        
        QCD_unwgt_str1 = f'TTbarRes_0l_UL16preVFP_QCD'
        QCD_unwgt_str2 = f'TTbarRes_0l_UL16postVFP_QCD'
        
        PathToFile = f'CoffeaOutputsForCombine/Coffea_FirstRun/QCD/{btag}/{str(year)}'
        
        Output['preVFP'] = util.load(f'{PathToFile}/APV/{QCD_unwgt_str1}{od}.coffea')
        Output['postVFP'] = util.load(f'{PathToFile}/noAPV/{QCD_unwgt_str2}{od}.coffea')
        
    return(Output)

def QCD_Weighted(btag, year, contam, unc, mr):
    
    '''
        btag --> string ('', 'LooseBTag', 'MediumBTag'); Select btag directory files are saved in
        year --> integer (2016, 2017, 2018); Select run year
        contam --> bool (True, False); Is the mistag contamination removed? Yay or neah
        unc --> string; Select the uncertainty type (e.g. btagUnc_central)
        mr --> bool (True, False); Mistag Rate Weights Only? Yay or neah
    '''
    
    Output = {}
    
    if btag == '':
        od = '_oldANdisc'
    else:
        od = ''

    if contam == True:
        Contam = '_ttbarContamRemoved'
    else:
        Contam = ''

    if mr == True:
        MR = '_MistagOnly'
    else:
        MR = ''
        
    if year == 2016:
        
        QCD_unwgt_str1 = f'TTbarRes_0l_UL16preVFP_QCD_weighted{Contam}{unc}{MR}'
        QCD_unwgt_str2 = f'TTbarRes_0l_UL16postVFP_QCD_weighted{Contam}{unc}{MR}'
        
        PathToFile = f'CoffeaOutputsForCombine/Coffea_SecondRun/QCD/{btag}/{str(year)}'
        
        Output['preVFP'] = util.load(f'{PathToFile}/APV/{QCD_unwgt_str1}{od}.coffea')
        Output['postVFP'] = util.load(f'{PathToFile}/noAPV/{QCD_unwgt_str2}{od}.coffea')
        
    return(Output)

def ScaledTTbar(Output, Year, HistName, CategoryInt, ScaleFactor):
    '''
        Output --> Dictionary of Data Outputs
        Year --> Integer; Year of 
        HistName --> String; Name of Histogram to be Plotted ('ttbarmass', 'jetpt', etc...)
        CategoryInt --> Integer; Number Corresponding to Name of Category ('2t0bcen', 'pret1bfwd', etc...)
        ScaleFactor --> Dictionary of Scale Factors for TTbar
    '''
    
    
    TTbar1 = Output['700_1000_preVFP'][HistName][f'UL{str(Year-2000)}preVFP_TTbar_700_1000', CategoryInt, :]*ScaleFactor['700_1000_preVFP']\
           + Output['700_1000_postVFP'][HistName][f'UL{str(Year-2000)}postVFP_TTbar_700_1000', CategoryInt, :]*ScaleFactor['700_1000_postVFP']
                
    TTbar2 = Output['1000_Inf_preVFP'][HistName][f'UL{str(Year-2000)}preVFP_TTbar_1000_Inf', CategoryInt, :]*ScaleFactor['1000_Inf_preVFP']\
           + Output['1000_Inf_postVFP'][HistName][f'UL{str(Year-2000)}postVFP_TTbar_1000_Inf', CategoryInt, :]*ScaleFactor['1000_Inf_postVFP']

    TTbar = TTbar1 + TTbar2
    
    return(TTbar)

def ScaledTTbarDisc(Output, Year, HistName, Xaxis, ScaleFactor, ptbin=sum, massbin=sum):
    '''
        Output --> Dictionary of Data Outputs
        Year --> Integer; Year of 
        HistName --> String; Name of Histogram to be Plotted ('ttbarmass', 'jetpt', etc...)
        Xaxis --> String; Either 'pt', 'mass', or leave blank, ''.  If blamk, 1D plot will be made with discriminator as x-axis 
        ScaleFactor --> Dictionary of Scale Factors for TTbar
        ptbin --> Integer; pt bin number
        massbin --> Integer; SD mass bin number
    '''
    
    if Xaxis == 'pt':
        
        TTbar1 = Output['700_1000_preVFP'][HistName][f'UL{str(Year-2000)}preVFP_TTbar_700_1000', :, sum, :]*ScaleFactor['700_1000_preVFP']\
               + Output['700_1000_postVFP'][HistName][f'UL{str(Year-2000)}postVFP_TTbar_700_1000', :, sum, :]*ScaleFactor['700_1000_postVFP']

        TTbar2 = Output['1000_Inf_preVFP'][HistName][f'UL{str(Year-2000)}preVFP_TTbar_1000_Inf', :, sum, :]*ScaleFactor['1000_Inf_preVFP']\
               + Output['1000_Inf_postVFP'][HistName][f'UL{str(Year-2000)}postVFP_TTbar_1000_Inf', :, sum, :]*ScaleFactor['1000_Inf_postVFP']
        
    elif Xaxis == 'mass':
        
        TTbar1 = Output['700_1000_preVFP'][HistName][f'UL{str(Year-2000)}preVFP_TTbar_700_1000', sum, :, :]*ScaleFactor['700_1000_preVFP']\
               + Output['700_1000_postVFP'][HistName][f'UL{str(Year-2000)}postVFP_TTbar_700_1000', sum, :, :]*ScaleFactor['700_1000_postVFP']

        TTbar2 = Output['1000_Inf_preVFP'][HistName][f'UL{str(Year-2000)}preVFP_TTbar_1000_Inf', sum, :, :]*ScaleFactor['1000_Inf_preVFP']\
               + Output['1000_Inf_postVFP'][HistName][f'UL{str(Year-2000)}postVFP_TTbar_1000_Inf', sum, :, :]*ScaleFactor['1000_Inf_postVFP']
        
    else:
        
        TTbar1 = Output['700_1000_preVFP'][HistName][f'UL{str(Year-2000)}preVFP_TTbar_700_1000', massbin, ptbin, :]*ScaleFactor['700_1000_preVFP']\
               + Output['700_1000_postVFP'][HistName][f'UL{str(Year-2000)}postVFP_TTbar_700_1000', massbin, ptbin, :]*ScaleFactor['700_1000_postVFP']

        TTbar2 = Output['1000_Inf_preVFP'][HistName][f'UL{str(Year-2000)}preVFP_TTbar_1000_Inf', massbin, ptbin, :]*ScaleFactor['1000_Inf_preVFP']\
               + Output['1000_Inf_postVFP'][HistName][f'UL{str(Year-2000)}postVFP_TTbar_1000_Inf', massbin, ptbin, :]*ScaleFactor['1000_Inf_postVFP']

    TTbar = TTbar1 + TTbar2
    
    return(TTbar)
        
def Cutflow(Output):
    '''
        Output --> Dictionary of Data Outputs
    '''
    
    for name,output in Output.items(): 
        print("-------- " + name + "--------")
        for i,j in output['cutflow'].items():        
            print( '%20s : %12d' % (i,j) )