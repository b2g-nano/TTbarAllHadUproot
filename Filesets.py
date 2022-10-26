#!/usr/bin/env python
# coding: utf-8

# 
# # ---- All Datasets used for analysis and dasgoclient searches ---- #
'''
/JetHT/Run2016B-ver1_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD, 8.6GB,   11 files, 9726665 events
/JetHT/Run2016B-ver2_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD, 152.2GB, 74 files, 133752091 events
/JetHT/Run2016C-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD,      57.9GB,  45 files, 46495988 events
/JetHT/Run2016D-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD,      89.8GB,  65 files, 73330042 events
/JetHT/Run2016E-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD,      87.1GB,  49 files, 69219288 events
/JetHT/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD,           8.3GB,   5 files,  6613811 events
/JetHT/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD,           154.2GB, 70 files, 120688407 events
/JetHT/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD,           156.1GB, 72 files, 124050331 events
                                                                                    583876623 events total
                                                                                    
/JetHT/Run2017B-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD,           67.8GB,  33 files, 63043590 events
/JetHT/Run2017C-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD,           107.8GB, 66 files, 96264601 events
/JetHT/Run2017D-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD,           52.5GB,  37 files, 46145204 events
/JetHT/Run2017E-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD,           110.6GB, 58 files, 89630771 events
/JetHT/Run2017F-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD,           147.8GB, 66 files, 115429972 events
                                                                                    410514138 events total

/JetHT/Run2018A-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD,           227.6GB, 146 files, 171484635 events
/JetHT/Run2018B-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD,           105.4GB, 45 files,  78255208 events
/JetHT/Run2018C-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD,           96.1GB,  57 files,  70027804 events
/JetHT/Run2018D-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD,           493.1GB, 232 files, 356976276 events
                                                                                     676743923 events total
'''
#
# Weighted JetHT files are for the data driven background (pre-tag region)

import os

def CollectDatasets(redirector_str):
    """
        redirector_str --> string for either running over lpc of coffea-casa
            Two acceptable inputs: 'root://xcache/' Only works in Coffea-Casa Environment
                                   'root://cmsxrootd.fnal.gov/'
    """
    filedir = 'TTbarAllHadUproot/nanoAODv9Files/'
    Years = ['UL16', 'UL17', 'UL18']
    VFP = ['preVFP', 'postVFP']

    filesets = {} # To be filled and returned by this function
 
    # ---- Before concatenation with +=, lists should be declard ---- #
    
    for v in VFP:
        filesets['UL'+v+'_QCD'] = []
        filesets['UL'+v+'_TTbar'] = []
        
        for i in range(1000, 5500, 500):
            filesets['UL'+v+'_DM'+str(i)] = []
            filesets['UL'+v+'_RSGluon'+str(i)] = []
    
    # ---- Loop through years and VFP status, filling the filesets dictionary with the MC file locations from corresponding txt files ---- #
    
    for y in Years:
        if '16' in y:
            for v in VFP:
                # ---- QCD ---- #
                ulqcdfilename = filedir + 'QCD/QCD_NanoAODv9_' + y + '_' + v + '.txt'
                with open(ulqcdfilename) as f:
                    ulqcdfiles = [redirector_str + s.strip() for s in f.readlines()]
                filesets[y+v+'_QCD'] = ulqcdfiles
                filesets['UL'+v+'_QCD'] += ulqcdfiles # Combine files of all three years for both VFP conditions

                # ---- TTbar ---- #
                ulttbar700to1000filename = filedir + 'TT/TT_Mtt-700to1000_NanoAODv9_' + y + '_' + v + '.txt'
                with open(ulttbar700to1000filename) as f:
                    ulttbar700to1000files = [redirector_str + s.strip() for s in f.readlines()]
                ulttbar1000toInffilename = filedir + 'TT/TT_Mtt-1000toInf_NanoAODv9_' + y + '_' + v + '.txt'
                with open(ulttbar1000toInffilename) as f:
                    ulttbar1000toInffiles = [redirector_str + s.strip() for s in f.readlines()]
                ulttbarfiles = ulttbar700to1000files + ulttbar1000toInffiles # inclusion of both biased samples
                filesets[y+v+'_TTbar'] = ulttbarfiles
                filesets['UL'+v+'_TTbar'] += ulttbarfiles # Combine files of all three years for both VFP conditions

                # ---- Z' Dark Matter Mediator ---- #
                ulZprimeDMfilename = filedir + 'ZprimeDMToTTbar/ZprimeDMToTTbar_NanoAODv9_' + y + '_' + v + '.txt'
                ulDMfiles=[]
                k=0
                for i in range(1000, 5500, 500):
                    with open(ulZprimeDMfilename) as f:
                        ulDMfiles.append([redirector_str + s.strip() for s in f.readlines() if "ResoIncl_MZp"+str(i) in s])
                    filesets[y+v+'_DM'+str(i)] = ulDMfiles[k]
                    filesets['UL'+v+'_DM'+str(i)] += ulDMfiles[k] # Combine files of all three years for both VFP conditions
                    k += 1
                # ---- RS KK Gluon ---- #
                ulRSGluonfilename = filedir + 'RSGluonToTT/RSGluonToTT_NanoAODv9_' + y + '_' + v + '.txt'
                ulRSGluonfiles=[]
                l=0
                for i in range(1000, 5500, 500):
                    with open(ulRSGluonfilename) as f:
                        ulRSGluonfiles.append([redirector_str + s.strip() for s in f.readlines() if "RSGluonToTT_M-"+str(i) in s])
                    filesets[y+v+'_RSGluon'+str(i)] = ulRSGluonfiles[l]
                    filesets['UL'+v+'_RSGluon'+str(i)] += ulRSGluonfiles[l] # Combine files of all three years for both VFP conditions
                    l += 1
        else: # UL17 and UL18
            v = VFP[1] # No preVFP after 2016 Run vertex problem was fixed
            # ---- QCD ---- #
            ulqcdfilename = filedir + 'QCD/QCD_NanoAODv9_' + y + '_' + v + '.txt'
            with open(ulqcdfilename) as f:
                ulqcdfiles = [redirector_str + s.strip() for s in f.readlines()]
            filesets[y+v+'_QCD'] = ulqcdfiles
            filesets['UL'+v+'_QCD'] += ulqcdfiles # Combine files of all three years for both VFP conditions

            # ---- TTbar ---- #
            ulttbar700to1000filename = filedir + 'TT/TT_Mtt-700to1000_NanoAODv9_' + y + '_' + v + '.txt'
            with open(ulttbar700to1000filename) as f:
                ulttbar700to1000files = [redirector_str + s.strip() for s in f.readlines()]
            ulttbar1000toInffilename = filedir + 'TT/TT_Mtt-1000toInf_NanoAODv9_' + y + '_' + v + '.txt'
            with open(ulttbar1000toInffilename) as f:
                ulttbar1000toInffiles = [redirector_str + s.strip() for s in f.readlines()]
            ulttbarfiles = ulttbar700to1000files + ulttbar1000toInffiles # inclusion of both biased samples
            filesets[y+v+'_TTbar'] = ulttbarfiles
            filesets['UL'+v+'_TTbar'] += ulttbarfiles # Combine files of all three years for both VFP conditions

            # ---- Z' Dark Matter Mediator ---- #
            ulZprimeDMfilename = filedir + 'ZprimeDMToTTbar/ZprimeDMToTTbar_NanoAODv9_' + y + '_' + v + '.txt'
            ulDMfiles=[]
            k=0
            for i in range(1000, 5500, 500):
                with open(ulZprimeDMfilename) as f:
                    ulDMfiles.append([redirector_str + s.strip() for s in f.readlines() if "ResoIncl_MZp"+str(i) in s])
                filesets[y+v+'_DM'+str(i)] = ulDMfiles[k]
                filesets['UL'+v+'_DM'+str(i)] += ulDMfiles[k] # Combine files of all three years for both VFP conditions
                k += 1
            # ---- RS KK Gluon ---- #
            ulRSGluonfilename = filedir + 'RSGluonToTT/RSGluonToTT_NanoAODv9_' + y + '_' + v + '.txt'
            ulRSGluonfiles=[]
            l=0
            for i in range(1000, 5500, 500):
                with open(ulRSGluonfilename) as f:
                    ulRSGluonfiles.append([redirector_str + s.strip() for s in f.readlines() if "RSGluonToTT_M-"+str(i) in s])
                filesets[y+v+'_RSGluon'+str(i)] = ulRSGluonfiles[l]
                filesets['UL'+v+'_RSGluon'+str(i)] += ulRSGluonfiles[l] # Combine files of all three years for both VFP conditions
                l += 1
            
    
    # ---- JetHT ---- #
    datafilelist = os.listdir(filedir + 'JetHT/')
    for filename in datafilelist:
        if 'Run2016' in filename:
            with open(filedir + 'JetHT/' + filename) as f:
                jetdatafiles2016 = [redirector_str + s.strip() for s in f.readlines()] 
        elif 'Run2017' in filename:
            with open(filedir + 'JetHT/' + filename) as g:
                jetdatafiles2017 = [redirector_str + s.strip() for s in g.readlines()[::3]] # Every third datafile
        else:
            with open(filedir + 'JetHT/' + filename) as h:
                jetdatafiles2018 = [redirector_str + s.strip() for s in h.readlines()[::3]] 
        
        if 'Run2016B' in filename:
            with open(filedir + 'JetHT/' + filename) as b:
                jetdatafiles2016b = [redirector_str + s.strip() for s in b.readlines()] 
        elif 'Run2016C' in filename:
            with open(filedir + 'JetHT/' + filename) as c:
                jetdatafiles2016c = [redirector_str + s.strip() for s in c.readlines()] 
        elif 'Run2016D' in filename:
            with open(filedir + 'JetHT/' + filename) as d:
                jetdatafiles2016d = [redirector_str + s.strip() for s in d.readlines()] 
        elif 'Run2016E' in filename:
            with open(filedir + 'JetHT/' + filename) as e:
                jetdatafiles2016e = [redirector_str + s.strip() for s in e.readlines()] 
        elif 'Run2016F' in filename:
            with open(filedir + 'JetHT/' + filename) as f:
                jetdatafiles2016f = [redirector_str + s.strip() for s in f.readlines()] 
        elif 'Run2016G' in filename:
            with open(filedir + 'JetHT/' + filename) as g:
                jetdatafiles2016g = [redirector_str + s.strip() for s in g.readlines()] 
        elif 'Run2016H' in filename:
            with open(filedir + 'JetHT/' + filename) as h:
                jetdatafiles2016h = [redirector_str + s.strip() for s in h.readlines()] 
                
        if 'Run2017B' in filename:
            with open(filedir + 'JetHT/' + filename) as b:
                jetdatafiles2017b = [redirector_str + s.strip() for s in b.readlines()[::3]] 
        elif 'Run2017C' in filename:
            with open(filedir + 'JetHT/' + filename) as c:
                jetdatafiles2017c = [redirector_str + s.strip() for s in c.readlines()[::3]] 
        elif 'Run2017D' in filename:
            with open(filedir + 'JetHT/' + filename) as d:
                jetdatafiles2017d = [redirector_str + s.strip() for s in d.readlines()[::3]] 
        elif 'Run2017E' in filename:
            with open(filedir + 'JetHT/' + filename) as e:
                jetdatafiles2017e = [redirector_str + s.strip() for s in e.readlines()[::3]] 
        elif 'Run2017F' in filename:
            with open(filedir + 'JetHT/' + filename) as f:
                jetdatafiles2017f = [redirector_str + s.strip() for s in f.readlines()[::3]] 
                
        if 'Run2018A' in filename:
            with open(filedir + 'JetHT/' + filename) as a:
                jetdatafiles2018a = [redirector_str + s.strip() for s in a.readlines()[::3]] 
        elif 'Run2018B' in filename:
            with open(filedir + 'JetHT/' + filename) as b:
                jetdatafiles2018b = [redirector_str + s.strip() for s in b.readlines()[::3]] 
        elif 'Run2018C' in filename:
            with open(filedir + 'JetHT/' + filename) as c:
                jetdatafiles2018c = [redirector_str + s.strip() for s in c.readlines()[::3]] 
        elif 'Run2018D' in filename:
            with open(filedir + 'JetHT/' + filename) as d:
                jetdatafiles2018d = [redirector_str + s.strip() for s in d.readlines()[::3]] 
                
    filesets['JetHT2016_Data'] = jetdatafiles2016   
    filesets['JetHT2017_Data'] = jetdatafiles2017 
    filesets['JetHT2018_Data'] = jetdatafiles2018 
    jetdatafiles = jetdatafiles2016 + jetdatafiles2017 + jetdatafiles2018 # All data
    filesets['JetHT_Data'] = jetdatafiles
    
    filesets['JetHT2016B_Data'] = jetdatafiles2016b
    filesets['JetHT2016C_Data'] = jetdatafiles2016c
    filesets['JetHT2016D_Data'] = jetdatafiles2016d
    filesets['JetHT2016E_Data'] = jetdatafiles2016e
    filesets['JetHT2016F_Data'] = jetdatafiles2016f
    filesets['JetHT2016G_Data'] = jetdatafiles2016g
    filesets['JetHT2016H_Data'] = jetdatafiles2016h
    
    filesets['JetHT2017B_Data'] = jetdatafiles2017b
    filesets['JetHT2017C_Data'] = jetdatafiles2017c
    filesets['JetHT2017D_Data'] = jetdatafiles2017d
    filesets['JetHT2017E_Data'] = jetdatafiles2017e
    filesets['JetHT2017F_Data'] = jetdatafiles2017f
    
    filesets['JetHT2018A_Data'] = jetdatafiles2018a
    filesets['JetHT2018B_Data'] = jetdatafiles2018b
    filesets['JetHT2018C_Data'] = jetdatafiles2018c
    filesets['JetHT2018D_Data'] = jetdatafiles2018d
    
    # ---- Single Muon ---- #
    datafilelist = os.listdir(filedir + 'SingleMu/')
    for filename in datafilelist:
        if 'Run2016' in filename:
            with open(filedir + 'SingleMu/' + filename) as f:
                singlemudatafiles2016 = [redirector_str + s.strip() for s in f.readlines()]
        elif 'Run2017' in filename:
            with open(filedir + 'SingleMu/' + filename) as g:
                singlemudatafiles2017 = [redirector_str + s.strip() for s in g.readlines()]
        else:
            with open(filedir + 'SingleMu/' + filename) as h:
                singlemudatafiles2018 = [redirector_str + s.strip() for s in h.readlines()] 
    filesets['SingleMu2016_Data'] = singlemudatafiles2016            
    filesets['SingleMu2017_Data'] = singlemudatafiles2017
    filesets['SingleMu2018_Data'] = singlemudatafiles2018  
    
    return filesets

# CollectDatasets('root://xcache/')
# xrootdstr1 = 'root://cmseos.fnal.gov//eos/uscms'
# xrootdstr2 = 'root://xcache/' # Only works in Coffea-Casa Environment
# xrootdstr3 = 'root://cmsxrootd-site.fnal.gov/' #If 2nd redirector fails, file is probably here
# xrootdstr4 = 'root://cmsxrootd.fnal.gov/'

