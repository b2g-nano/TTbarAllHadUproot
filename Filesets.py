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
(PUT MC DATASETS HERE WHEN I'M NOT FEELING LAZY)
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
    Years = ['UL16']#, 'UL17', 'UL18']
    VFP = ['preVFP', 'postVFP']

    filesets = {}

    for y in Years:
        for v in VFP:
            ulqcdfilename = filedir + 'QCD/QCD_NanoAODv9_' + y + '_' + v + '.txt'
            with open(ulqcdfilename) as f:
                ulqcdfiles = [redirector_str + s.strip() for s in f.readlines()]
            filesets[y+v+'_QCD'] = ulqcdfiles

            ulttbar700to1000filename = filedir + 'TT/TT_Mtt-700to1000_NanoAODv9_' + y + '_' + v + '.txt'
            with open(ulttbar700to1000filename) as f:
                ulttbar700to1000files = [redirector_str + s.strip() for s in f.readlines()]
            ulttbar1000toInffilename = filedir + 'TT/TT_Mtt-1000toInf_NanoAODv9_' + y + '_' + v + '.txt'
            with open(ulttbar1000toInffilename) as f:
                ulttbar1000toInffiles = [redirector_str + s.strip() for s in f.readlines()]
            ulttbarfiles = ulttbar700to1000files + ulttbar1000toInffiles # inclusion of both biased samples
            filesets[y+v+'_TTbar'] = ulttbarfiles

            ulZprimeDMfilename = filedir + 'ZprimeDMToTTbar/ZprimeDMToTTbar_NanoAODv9_' + y + '_' + v + '.txt'
            with open(ulZprimeDMfilename) as f:
                ulDM1000files = [redirector_str + s.strip() for s in f.readlines() if "ResoIncl_MZp1000" in s]
            filesets[y+v+'_DM1000'] = ulDM1000files
            with open(ulZprimeDMfilename) as f:
                ulDM1500files = [redirector_str + s.strip() for s in f.readlines() if "ResoIncl_MZp1500" in s]
            filesets[y+v+'_DM1500'] = ulDM1500files
            with open(ulZprimeDMfilename) as f:
                ulDM2000files = [redirector_str + s.strip() for s in f.readlines() if "ResoIncl_MZp2000" in s]
            filesets[y+v+'_DM2000'] = ulDM2000files
            with open(ulZprimeDMfilename) as f:
                ulDM2500files = [redirector_str + s.strip() for s in f.readlines() if "ResoIncl_MZp2500" in s]
            filesets[y+v+'_DM2500'] = ulDM2500files
            with open(ulZprimeDMfilename) as f:
                ulDM3000files = [redirector_str + s.strip() for s in f.readlines() if "ResoIncl_MZp3000" in s]
            filesets[y+v+'_DM3000'] = ulDM3000files
            with open(ulZprimeDMfilename) as f:
                ulDM3500files = [redirector_str + s.strip() for s in f.readlines() if "ResoIncl_MZp3500" in s]
            filesets[y+v+'_DM3500'] = ulDM3500files
            with open(ulZprimeDMfilename) as f:
                ulDM4000files = [redirector_str + s.strip() for s in f.readlines() if "ResoIncl_MZp4000" in s]
            filesets[y+v+'_DM4000'] = ulDM4000files
            with open(ulZprimeDMfilename) as f:
                ulDM4500files = [redirector_str + s.strip() for s in f.readlines() if "ResoIncl_MZp4500" in s]
            filesets[y+v+'_DM4500'] = ulDM4500files
            with open(ulZprimeDMfilename) as f:
                ulDM5000files = [redirector_str + s.strip() for s in f.readlines() if "ResoIncl_MZp5000" in s]
            filesets[y+v+'_DM5000'] = ulDM5000files

            ulRSGluonfilename = filedir + 'RSGluonToTT/RSGluonToTT_NanoAODv9_' + y + '_' + v + '.txt'
            with open(ulRSGluonfilename) as f:
                ulRSGluon1000files = [redirector_str + s.strip() for s in f.readlines() if "RSGluonToTT_M-1000" in s]
            filesets[y+v+'_RSGluon1000'] = ulRSGluon1000files
            with open(ulRSGluonfilename) as f:
                ulRSGluon1500files = [redirector_str + s.strip() for s in f.readlines() if "RSGluonToTT_M-1500" in s]
            filesets[y+v+'_RSGluon1500'] = ulRSGluon1500files
            with open(ulRSGluonfilename) as f:
                ulRSGluon2000files = [redirector_str + s.strip() for s in f.readlines() if "RSGluonToTT_M-2000" in s]
            filesets[y+v+'_RSGluon2000'] = ulRSGluon2000files
            with open(ulRSGluonfilename) as f:
                ulRSGluon2500files = [redirector_str + s.strip() for s in f.readlines() if "RSGluonToTT_M-2500" in s]
            filesets[y+v+'_RSGluon2500'] = ulRSGluon2500files
            with open(ulRSGluonfilename) as f:
                ulRSGluon3000files = [redirector_str + s.strip() for s in f.readlines() if "RSGluonToTT_M-3000" in s]
            filesets[y+v+'_RSGluon3000'] = ulRSGluon3000files
            with open(ulRSGluonfilename) as f:
                ulRSGluon3500files = [redirector_str + s.strip() for s in f.readlines() if "RSGluonToTT_M-3500" in s]
            filesets[y+v+'_RSGluon3500'] = ulRSGluon3500files
            with open(ulRSGluonfilename) as f:
                ulRSGluon4000files = [redirector_str + s.strip() for s in f.readlines() if "RSGluonToTT_M-4000" in s]
            filesets[y+v+'_RSGluon4000'] = ulRSGluon4000files
            with open(ulRSGluonfilename) as f:
                ulRSGluon4500files = [redirector_str + s.strip() for s in f.readlines() if "RSGluonToTT_M-4500" in s]
            filesets[y+v+'_RSGluon4500'] = ulRSGluon4500files
            with open(ulRSGluonfilename) as f:
                ulRSGluon5000files = [redirector_str + s.strip() for s in f.readlines() if "RSGluonToTT_M-5000" in s]
            filesets[y+v+'_RSGluon5000'] = ulRSGluon5000files

    datafilelist = os.listdir(filedir + 'JetHT/')
    for filename in datafilelist:
        if 'Run2016' in filename:
            with open(filedir + 'JetHT/' + filename) as f:
                jetdatafiles2016 = [redirector_str + s.strip() for s in f.readlines()[::3]] # Every third datafile
        elif 'Run2017' in filename:
            with open(filedir + 'JetHT/' + filename) as g:
                jetdatafiles2017 = [redirector_str + s.strip() for s in g.readlines()[::3]]
        else:
            with open(filedir + 'JetHT/' + filename) as h:
                jetdatafiles2018 = [redirector_str + s.strip() for s in h.readlines()[::3]] 
    filesets['JetHT2016_Data'] = jetdatafiles2016 
    filesets['JetHT2017_Data'] = jetdatafiles2017
    filesets['JetHT2018_Data'] = jetdatafiles2018
    jetdatafiles = jetdatafiles2016 + jetdatafiles2017 + jetdatafiles2018 # All data after unblinding
    
    return filesets


# xrootdstr1 = 'root://cmseos.fnal.gov//eos/uscms'
# xrootdstr2 = 'root://xcache/' # Only works in Coffea-Casa Environment
# xrootdstr3 = 'root://cmsxrootd-site.fnal.gov/' #If 2nd redirector fails, file is probably here
# xrootdstr4 = 'root://cmsxrootd.fnal.gov/'

