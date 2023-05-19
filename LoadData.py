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

def JetHT_Total(Input):
    
    Output = {}
    for i,j in Input.items():
        for k,l in j.items():
            if k in Output: 
                Output[k] += l
            else: 
                Output[k] = l.copy()
                
    return(Output)

def SingleMu_Unweighted(year):
    
    Output = {}
    
    if year == 2016:
        
        for Era in ['B', 'C', 'D', 'E', 'F']:
            datastr = f'TTbarRes_0l_UL16preVFP_SingleMu{Era}_Data_TriggerAnalysis'
            PathToFile = f'CoffeaOutputsForTriggerAnalysis/SingleMu/MediumBTag/{str(year)}/APV/{datastr}.coffea'
            Output[Era+'_preVFP'] = util.load(PathToFile)

        for Era in ['F', 'G', 'H']:
            datastr = f'TTbarRes_0l_UL16postVFP_SingleMu{Era}_Data_TriggerAnalysis'
            PathToFile = f'CoffeaOutputsForTriggerAnalysis/SingleMu/MediumBTag/{str(year)}/noAPV/{datastr}.coffea'
            Output[Era+'_postVFP'] = util.load(PathToFile)
            
    elif year == 2017:
        
        for Era in ['B', 'C', 'D', 'E', 'F']:
            datastr = f'TTbarRes_0l_UL17postVFP_SingleMu{Era}_Data_TriggerAnalysis'
            PathToFile = f'CoffeaOutputsForTriggerAnalysis/SingleMu/MediumBTag/{str(year)}/noAPV/{datastr}.coffea'
            Output[Era+'_postVFP'] = util.load(PathToFile)
            
    elif year == 2018:
        
        for Era in ['A', 'B', 'C', 'D']:
            datastr = f'TTbarRes_0l_UL18postVFP_SingleMu{Era}_Data_TriggerAnalysis'
            PathToFile = f'CoffeaOutputsForTriggerAnalysis/SingleMu/MediumBTag/{str(year)}/noAPV/{datastr}.coffea'
            Output[Era+'_postVFP'] = util.load(PathToFile)
        
    return(Output)

def AddEraHists(Output, Year, HistName, CategoryInt):
    '''
        Output --> Dictionary of Data Outputs
        Year --> Integer; Year of 
        HistName --> String; Name of Histogram to be Plotted ('ttbarmass', 'jetpt', etc...)
        CategoryInt --> Integer; Number Corresponding to Name of Category ('2t0bcen', 'pret1bfwd', etc...)
    '''
    
    JetHT_str = f'UL{str(Year - 2000)}preVFP_JetHTB_Data'
    HistObject = Output['B_preVFP'][HistName][JetHT_str, CategoryInt, :]
    
    # ---- Add all data together ---- #
    for vfp in ['preVFP', 'postVFP']:
        #---- Define Histograms from Coffea Outputs ----# 
        if vfp == 'preVFP':
            for Era in ['C', 'D', 'E', 'F']: #exclude B because histogram is initialized with B era
                JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, CategoryInt, :]
                
        else:
            for Era in ['F', 'G', 'H']: #exclude B because histogram is initialized with B era
                JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, CategoryInt, :]
                
    return HistObject

def AddEraDisc(Output, Year, HistName, Xaxis, ptbin=sum, massbin=sum, rhobin=sum, taggerbin=None):
    '''
        Output --> Dictionary of Data Outputs
        Year --> Integer; Year of 
        HistName --> String; Name of Discriminator Histogram to be Plotted ('deepTagMD_TvsQCD', 'deepB_subjet', etc...)
        Xaxis --> String; Either 'pt', 'mass', or leave blank, ''.  If blamk, 1D plot will be made with discriminator as x-axis 
        ptbin --> Integer; pt bin number
        massbin --> Integer; SD mass bin number
    '''
    
    JetHT_str = f'UL{str(Year - 2000)}preVFP_JetHTB_Data'
        
    if Xaxis == 'pt':
        
        if taggerbin != None:
        
            HistObject = Output['B_preVFP'][HistName][JetHT_str, :, sum, sum, taggerbin]

            # ---- Add all data together ---- #
            for vfp in ['preVFP', 'postVFP']:
                #---- Define Histograms from Coffea Outputs ----# 
                if vfp == 'preVFP':
                    for Era in ['C', 'D', 'E', 'F']: 
                        JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, :, sum, sum, taggerbin]
                else:
                    for Era in ['F', 'G', 'H']: 
                        JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, :, sum, sum, taggerbin]
                        
        else:
            
            HistObject = Output['B_preVFP'][HistName][JetHT_str, :, sum, sum, :]

            # ---- Add all data together ---- #
            for vfp in ['preVFP', 'postVFP']:
                #---- Define Histograms from Coffea Outputs ----# 
                if vfp == 'preVFP':
                    for Era in ['C', 'D', 'E', 'F']: 
                        JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, :, sum, sum, :]
                else:
                    for Era in ['F', 'G', 'H']: 
                        JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, :, sum, sum, :]

    elif Xaxis == 'mass':
        
        if taggerbin != None:
        
            HistObject = Output['B_preVFP'][HistName][JetHT_str, sum, :, sum, taggerbin]

            # ---- Add all data together ---- #
            for vfp in ['preVFP', 'postVFP']:
                #---- Define Histograms from Coffea Outputs ----# 
                if vfp == 'preVFP':
                    for Era in ['C', 'D', 'E', 'F']: 
                        JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, sum, :, sum, taggerbin]
                else:
                    for Era in ['F', 'G', 'H']: 
                        JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, sum, :, sum, taggerbin]
                        
        else:
            
            HistObject = Output['B_preVFP'][HistName][JetHT_str, sum, :, sum, :]

            # ---- Add all data together ---- #
            for vfp in ['preVFP', 'postVFP']:
                #---- Define Histograms from Coffea Outputs ----# 
                if vfp == 'preVFP':
                    for Era in ['C', 'D', 'E', 'F']: 
                        JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, sum, :, sum, :]
                else:
                    for Era in ['F', 'G', 'H']: 
                        JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, sum, :, sum, :]
                        
    elif Xaxis == 'rho':
        
        if taggerbin != None:
        
            HistObject = Output['B_preVFP'][HistName][JetHT_str, sum, sum, :, taggerbin]

            # ---- Add all data together ---- #
            for vfp in ['preVFP', 'postVFP']:
                #---- Define Histograms from Coffea Outputs ----# 
                if vfp == 'preVFP':
                    for Era in ['C', 'D', 'E', 'F']: 
                        JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, sum, sum, :, taggerbin]
                else:
                    for Era in ['F', 'G', 'H']: 
                        JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, sum, sum, :, taggerbin]
                        
        else:
            
            HistObject = Output['B_preVFP'][HistName][JetHT_str, sum, sum, :, :]

            # ---- Add all data together ---- #
            for vfp in ['preVFP', 'postVFP']:
                #---- Define Histograms from Coffea Outputs ----# 
                if vfp == 'preVFP':
                    for Era in ['C', 'D', 'E', 'F']: 
                        JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, sum, sum, :, :]
                else:
                    for Era in ['F', 'G', 'H']: 
                        JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, sum, sum, :, :]

    else: # Make 1D Plots
            
        HistObject = Output['B_preVFP'][HistName][JetHT_str, massbin, ptbin, rhobin, :]

        # ---- Add all data together ---- #
        for vfp in ['preVFP', 'postVFP']:
            #---- Define Histograms from Coffea Outputs ----# 
            if vfp == 'preVFP':
                for Era in ['C', 'D', 'E', 'F']: 
                    JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                    HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, massbin, ptbin, rhobin, :]
            else:
                for Era in ['F', 'G', 'H']: 
                    JetHT_str = f'UL{str(Year - 2000)}{vfp}_JetHT{Era}_Data'
                    HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, massbin, ptbin, rhobin, :]
                
    return HistObject

def AddEraHistsSingleMu(Output, Year, HistName, Xaxis, CategoryInt):
    '''
        Output --> Dictionary of Data Outputs
        Year --> Integer; Year of 
        HistName --> String; Name of Histogram to be Plotted ('ttbarmass', 'jetpt', etc...)
        Xaxis --> String; xaxis to be Plotted ('ht', 'sd', 'tt')
        CategoryInt --> Integer; Number Corresponding to Name of Category ('2t0bcen', 'pret1bfwd', etc...)
    '''
    
    if Year == 2016:
        
        JetHT_str = f'UL{str(Year - 2000)}preVFP_SingleMuB_Data'
        
        if Xaxis == 'ht':
            HistObject = Output['B_preVFP'][HistName][JetHT_str, CategoryInt, :, sum]
        elif Xaxis == 'sd':
            HistObject = Output['B_preVFP'][HistName][JetHT_str, CategoryInt, sum, :]
        else:
            print('Invalid xaxis')

        # ---- Add all data together ---- #
        for vfp in ['preVFP', 'postVFP']:
            #---- Define Histograms from Coffea Outputs ----# 
            if vfp == 'preVFP':
                for Era in ['C', 'D', 'E', 'F']: #exclude B because histogram is initialized with B era
                    JetHT_str = f'UL{str(Year - 2000)}{vfp}_SingleMu{Era}_Data'
                    if Xaxis == 'ht':
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, CategoryInt, :, sum]
                    elif Xaxis == 'sd':
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, CategoryInt, sum, :]
                    else:
                        print('Invalid xaxis')
            else:
                for Era in ['F', 'G', 'H']: #exclude B because histogram is initialized with B era
                    JetHT_str = f'UL{str(Year - 2000)}{vfp}_SingleMu{Era}_Data'
                    if Xaxis == 'ht':
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, CategoryInt, :, sum]
                    elif Xaxis == 'sd':
                        HistObject += Output[Era+'_'+vfp][HistName][JetHT_str, CategoryInt, sum, :]
                    else:
                        print('Invalid xaxis')
                        
    elif Year == 2017 and '3' in HistName:
        
        JetHT_str = f'UL{str(Year - 2000)}postVFP_SingleMuC_Data'
        
        if Xaxis == 'ht':
            HistObject = Output['C_postVFP'][HistName][JetHT_str, CategoryInt, :, sum]
        elif Xaxis == 'sd':
            HistObject = Output['C_postVFP'][HistName][JetHT_str, CategoryInt, sum, :]
        else:
            print('Invalid xaxis')

        # ---- Add all data together ---- #
        for Era in ['D', 'E', 'F']: #exclude B because histogram is initialized with C era
            JetHT_str = f'UL{str(Year - 2000)}postVFP_SingleMu{Era}_Data'
            if Xaxis == 'ht':
                HistObject += Output[Era+'_postVFP'][HistName][JetHT_str, CategoryInt, :, sum]
            elif Xaxis == 'sd':
                HistObject += Output[Era+'_postVFP'][HistName][JetHT_str, CategoryInt, sum, :]
            else:
                print('Invalid xaxis')
                        
    elif Year == 2017 and '3' not in HistName:
        
        JetHT_str = f'UL{str(Year - 2000)}postVFP_SingleMuB_Data'
        
        if Xaxis == 'ht':
            HistObject = Output['B_postVFP'][HistName][JetHT_str, CategoryInt, :, sum]
        elif Xaxis == 'sd':
            HistObject = Output['B_postVFP'][HistName][JetHT_str, CategoryInt, sum, :]
        else:
            print('Invalid xaxis')

        # ---- Add all data together ---- #
        for Era in ['C', 'D', 'E', 'F']: #exclude B because histogram is initialized with B era
            JetHT_str = f'UL{str(Year - 2000)}postVFP_SingleMu{Era}_Data'
            if Xaxis == 'ht':
                HistObject += Output[Era+'_postVFP'][HistName][JetHT_str, CategoryInt, :, sum]
            elif Xaxis == 'sd':
                HistObject += Output[Era+'_postVFP'][HistName][JetHT_str, CategoryInt, sum, :]
            else:
                print('Invalid xaxis')
                
    elif Year == 2018:
        
        JetHT_str = f'UL{str(Year - 2000)}postVFP_SingleMuA_Data'
        
        if Xaxis == 'ht':
            HistObject = Output['A_postVFP'][HistName][JetHT_str, CategoryInt, :, sum]
        elif Xaxis == 'sd':
            HistObject = Output['A_postVFP'][HistName][JetHT_str, CategoryInt, sum, :]
        else:
            print('Invalid xaxis')

        # ---- Add all data together ---- #
        for Era in ['B', 'C', 'D']: #exclude A because histogram is initialized with B era
            JetHT_str = f'UL{str(Year - 2000)}postVFP_SingleMu{Era}_Data'
            if Xaxis == 'ht':
                HistObject += Output[Era+'_postVFP'][HistName][JetHT_str, CategoryInt, :, sum]
            elif Xaxis == 'sd':
                HistObject += Output[Era+'_postVFP'][HistName][JetHT_str, CategoryInt, sum, :]
            else:
                print('Invalid xaxis')
                
    return HistObject
    
    
    
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