#!/usr/bin/env python 
# coding: utf-8

from coffea import util
import numpy as np

def JetHT_Unweighted(btag, year):
    
    Output = {}
    
    if year == 2016:
        
        if btag == '':
            od = '_oldANdisc'
        else:
            od = ''

        for Era in ['B', 'C', 'D', 'E', 'F']:
            datastr = f'TTbarRes_0l_UL16preVFP_JetHT{Era}_Data'
            PathToFile = f'CoffeaOutputsForCombine/Coffea_FirstRun/JetHT/{btag}/{str(year)}/APV/{datastr}{od}.coffea'
            Output[Era+'_preVFP'] = util.load(PathToFile)

        for Era in ['F', 'G', 'H']:
            datastr = f'TTbarRes_0l_UL16postVFP_JetHT{Era}_Data'
            PathToFile = f'CoffeaOutputsForCombine/Coffea_FirstRun/JetHT/{btag}/{str(year)}/noAPV/{datastr}{od}.coffea'
            Output[Era+'_postVFP'] = util.load(PathToFile)
        
    return(Output)

def JetHT_Weighted(btag, year, contam, unc, mr):
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

        for Era in ['B', 'C', 'D', 'E', 'F']:
            datastr = f'TTbarRes_0l_UL16preVFP_JetHT{Era}_Data_weighted{Contam}{unc}{MR}'
            PathToFile = f'CoffeaOutputsForCombine/Coffea_SecondRun/JetHT/{btag}/{str(year)}/APV/{datastr}{od}.coffea'
            Output[Era+'_preVFP'] = util.load(PathToFile)

        for Era in ['F', 'G', 'H']:
            datastr = f'TTbarRes_0l_UL16postVFP_JetHT{Era}_Data_weighted{Contam}{unc}{MR}'
            PathToFile = f'CoffeaOutputsForCombine/Coffea_SecondRun/JetHT/{btag}/{str(year)}/noAPV/{datastr}{od}.coffea'
            Output[Era+'_postVFP'] = util.load(PathToFile)
        
    return(Output)

def Cutflow(Output):
    '''
        Output --> Dictionary of Data Outputs
    '''
    
    EraB = np.array([])
    EraC = np.array([])
    EraD = np.array([])
    EraE = np.array([])
    EraF1 = np.array([])
    EraF2 = np.array([])
    EraG = np.array([])
    EraH = np.array([])
    for dataset,output in Output.items():
        if 'B' in dataset:
            # print("-------" + dataset + " Cutflow--------")
            for i,j in output['cutflow'].items(): 
                EraB = np.append(EraB,j)
            # print(EraB)
        elif 'C' in dataset:
            # print("-------" + dataset + " Cutflow--------")
            for i,j in output['cutflow'].items(): 
                EraC = np.append(EraC,j)
            # print(EraC)
        elif 'D' in dataset:
            # print("-------" + dataset + " Cutflow--------")
            for i,j in output['cutflow'].items(): 
                EraD = np.append(EraD,j)
            # print(EraD)
        elif 'E' in dataset:
            # print("-------" + dataset + " Cutflow--------")
            for i,j in output['cutflow'].items(): 
                EraE = np.append(EraE,j)
            # print(EraE)
        elif 'F_pre' in dataset:
            # print("-------" + dataset + " Cutflow--------")
            for i,j in output['cutflow'].items(): 
                EraF1 = np.append(EraF1,j)
            # print(EraF1)
        elif 'F_post' in dataset:
            # print("-------" + dataset + " Cutflow--------")
            for i,j in output['cutflow'].items(): 
                EraF2 = np.append(EraF2,j)
            # print(EraF2)
        elif 'G' in dataset:
            # print("-------" + dataset + " Cutflow--------")
            for i,j in output['cutflow'].items(): 
                EraG = np.append(EraG,j)
            # print(EraG)
        if 'H' in dataset:
            # print("-------" + dataset + " Cutflow--------")
            for i,j in output['cutflow'].items(): 
                EraH = np.append(EraH,j)
            # print(EraH)
            
    AllData = EraB + EraC + EraD + EraE + EraF1 + EraF2 + EraG + EraH
    index = 0
    print("------- Data Sum of Cutflows --------")
    for i,j in Output['C_preVFP']['cutflow'].items():
        print( '%20s : %10i' % (i,AllData[index]) )
        index+=1
    
    # Total should be 625441538