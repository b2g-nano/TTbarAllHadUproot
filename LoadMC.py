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

def MCCutflow(Output):
    '''
        Output --> Dictionary of Data Outputs
    '''
    
    for name,output in Output.items(): 
        print("-------- " + name + "--------")
        for i,j in output['cutflow'].items():        
            print( '%20s : %12d' % (i,j) )