#!/usr/bin/env python
# coding: utf-8

# Reads in the `.root` files for the corresponding datasets.  One can verify this notebook works and produce the corresponding .py module.  In that module, one is free to include or comment out any wanted/unwanted group of files in the dictionary `filesets`.  This dictionary 'filesets' is imported into other modules, including `TTbarResCoffeaOutputs`, so make sure you specify the files you want in that module before you make your outputs.
# 
# # ---- All JetHT Datasets used for analysis and dasgoclient searches ---- #
# 
# /JetHT/Run2016B_ver1-Nano25Oct2019_ver1-v1/NANOAOD 
# /JetHT/Run2016B_ver2-Nano25Oct2019_ver2-v1/NANOAOD 
# /JetHT/Run2016C-Nano25Oct2019-v1/NANOAOD 
# /JetHT/Run2016D-Nano25Oct2019-v1/NANOAOD 
# /JetHT/Run2016E-Nano25Oct2019-v1/NANOAOD 
# /JetHT/Run2016F-Nano25Oct2019-v1/NANOAOD 
# /JetHT/Run2016G-Nano25Oct2019-v1/NANOAOD 
# /JetHT/Run2016H-Nano25Oct2019-v1/NANOAOD 
# /JetHT/Run2017B-Nano25Oct2019-v1/NANOAOD 
# /JetHT/Run2017C-Nano25Oct2019-v1/NANOAOD 
# /JetHT/Run2017D-Nano25Oct2019-v1/NANOAOD 
# /JetHT/Run2017E-Nano25Oct2019-v1/NANOAOD 
# /JetHT/Run2017F-Nano25Oct2019-v1/NANOAOD 
# /JetHT/Run2018A-Nano25Oct2019-v1/NANOAOD 
# /JetHT/Run2018B-Nano25Oct2019-v1/NANOAOD 
# /JetHT/Run2018C-Nano25Oct2019-v2/NANOAOD 
# /JetHT/Run2018D-Nano25Oct2019_ver2-v1/NANOAOD
#
# Weighted JetHT files are for the data driven background (pre-tag region)

import os

xrootdstr1 = 'root://cmseos.fnal.gov/'
xrootdstr2 = 'root://cmsxrootd.fnal.gov/'
xrootdstr3 = 'root://cmsxrootd-site.fnal.gov/'

filedir = 'TTbarAllHadUproot/nanoAODv9Files/'
Years = ['UL16']#, 'UL17', 'UL18']
VFP = ['preVFP', 'postVFP']

filesets = {}

for y in Years:
    for v in VFP:
        ulqcdfilename = filedir + 'QCD/QCD_NanoAODv9_' + y + '_' + v + '.txt'
        with open(ulqcdfilename) as f:
            ulqcdfiles = [xrootdstr2 + s.strip() for s in f.readlines()]
        filesets[y+v+'_QCD'] = ulqcdfiles
        
        ulttbar700to1000filename = filedir + 'TT/TT_Mtt-700to1000_NanoAODv9_' + y + '_' + v + '.txt'
        with open(ulttbar700to1000filename) as f:
            ulttbar700to1000files = [xrootdstr2 + s.strip() for s in f.readlines()]
        ulttbar1000toInffilename = filedir + 'TT/TT_Mtt-1000toInf_NanoAODv9_' + y + '_' + v + '.txt'
        with open(ulttbar1000toInffilename) as f:
            ulttbar1000toInffiles = [xrootdstr2 + s.strip() for s in f.readlines()]
        ulttbarfiles = ulttbar700to1000files + ulttbar1000toInffiles # inclusion of both biased samples
        filesets[y+v+'_TTbar'] = ulttbarfiles

        ulZprimeDMfilename = filedir + 'ZprimeDMToTTbar/ZprimeDMToTTbar_NanoAODv9_' + y + '_' + v + '.txt'
        with open(ulZprimeDMfilename) as f:
            ulDM1000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp1000" in s]
        filesets[y+v+'_DM1000'] = ulDM1000files
        with open(ulZprimeDMfilename) as f:
            ulDM1500files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp1500" in s]
        filesets[y+v+'_DM1500'] = ulDM1500files
        with open(ulZprimeDMfilename) as f:
            ulDM2000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp2000" in s]
        filesets[y+v+'_DM2000'] = ulDM2000files
        with open(ulZprimeDMfilename) as f:
            ulDM2500files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp2500" in s]
        filesets[y+v+'_DM2500'] = ulDM2500files
        with open(ulZprimeDMfilename) as f:
            ulDM3000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp3000" in s]
        filesets[y+v+'_DM3000'] = ulDM3000files
        with open(ulZprimeDMfilename) as f:
            ulDM3500files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp3500" in s]
        filesets[y+v+'_DM3500'] = ulDM3500files
        with open(ulZprimeDMfilename) as f:
            ulDM4000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp4000" in s]
        filesets[y+v+'_DM4000'] = ulDM4000files
        with open(ulZprimeDMfilename) as f:
            ulDM4500files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp4500" in s]
        filesets[y+v+'_DM4500'] = ulDM4500files
        with open(ulZprimeDMfilename) as f:
            ulDM5000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp5000" in s]
        filesets[y+v+'_DM5000'] = ulDM5000files

        ulRSGluonfilename = filedir + 'RSGluonToTT/RSGluonToTT_NanoAODv9_' + y + '_' + v + '.txt'
        with open(ulRSGluonfilename) as f:
            ulRSGluon1000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-1000" in s]
        filesets[y+v+'_RSGluon1000'] = ulRSGluon1000files
        with open(ulRSGluonfilename) as f:
            ulRSGluon1500files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-1500" in s]
        filesets[y+v+'_RSGluon1500'] = ulRSGluon1500files
        with open(ulRSGluonfilename) as f:
            ulRSGluon2000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-2000" in s]
        filesets[y+v+'_RSGluon2000'] = ulRSGluon2000files
        with open(ulRSGluonfilename) as f:
            ulRSGluon2500files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-2500" in s]
        filesets[y+v+'_RSGluon2500'] = ulRSGluon2500files
        with open(ulRSGluonfilename) as f:
            ulRSGluon3000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-3000" in s]
        filesets[y+v+'_RSGluon3000'] = ulRSGluon3000files
        with open(ulRSGluonfilename) as f:
            ulRSGluon3500files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-3500" in s]
        filesets[y+v+'_RSGluon3500'] = ulRSGluon3500files
        with open(ulRSGluonfilename) as f:
            ulRSGluon4000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-4000" in s]
        filesets[y+v+'_RSGluon4000'] = ulRSGluon4000files
        with open(ulRSGluonfilename) as f:
            ulRSGluon4500files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-4500" in s]
        filesets[y+v+'_RSGluon4500'] = ulRSGluon4500files
        with open(ulRSGluonfilename) as f:
            ulRSGluon5000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-5000" in s]
        filesets[y+v+'_RSGluon5000'] = ulRSGluon5000files

datafilelist = os.listdir(filedir + 'JetHT/')
for filename in datafilelist:
    if 'Run2016' in filename:
        with open(filedir + 'JetHT/' + filename) as f:
            jetdatafiles2016 = [xrootdstr2 + s.strip() for s in f.readlines()[::3]] # Every third datafile
    elif 'Run2017' in filename:
        with open(filedir + 'JetHT/' + filename) as g:
            jetdatafiles2017 = [xrootdstr2 + s.strip() for s in g.readlines()[::3]]
    else:
        with open(filedir + 'JetHT/' + filename) as h:
            jetdatafiles2018 = [xrootdstr2 + s.strip() for s in h.readlines()[::3]] 
filesets['JetHT2016_Data'] = jetdatafiles2016            
jetdatafiles = jetdatafiles2016 + jetdatafiles2017 + jetdatafiles2018 # All data after unblinding

""" Comment out whichever files you wish to not be included """

# filesets = {
#     'UL16_QCD':ul16qcdfiles,
#     'UL16_DM1000':ul16DM1000files,
#     'UL16_DM1500':ul16DM1500files,
#     'UL16_DM2000':ul16DM2000files,
#     'UL16_DM2500':ul16DM2500files,
#     'UL16_DM3000':ul16DM3000files,
#     'UL16_DM3500':ul16DM3500files,
#     'UL16_DM4000':ul16DM4000files,
#     'UL16_DM4500':ul16DM4500files,
#     'UL16_DM5000':ul16DM5000files,
#     'UL16_RSGluon1000':ul16RSGluon1000files,
#     'UL16_RSGluon1500':ul16RSGluon1500files,
#     'UL16_RSGluon2000':ul16RSGluon2000files,
#     'UL16_RSGluon2500':ul16RSGluon2500files,
#     'UL16_RSGluon3000':ul16RSGluon3000files,
#     'UL16_RSGluon3500':ul16RSGluon3500files,
#     'UL16_RSGluon4000':ul16RSGluon4000files,
#     'UL16_RSGluon4500':ul16RSGluon4500files,
#     'UL16_RSGluon5000':ul16RSGluon5000files,
#     'UL16_TTbar':ul16ttbarfiles,
#     'JetHT2016_Data':jetdatafiles2016,
    #-----------------------------------------#
#     'UL17_QCD':ul17qcdfiles,
#     'UL17_DM1000':ul17DM1000files,
#     'UL17_DM1500':ul17DM1500files,
#     'UL17_DM2000':ul17DM2000files,
#     'UL17_DM2500':ul17DM2500files,
#     'UL17_DM3000':ul17DM3000files,
#     'UL17_DM3500':ul17DM3500files,
#     'UL17_DM4000':ul17DM4000files,
#     'UL17_DM4500':ul17DM4500files,
#     'UL17_DM5000':ul17DM5000files,
#     'UL17_RSGluon1000':ul17RSGluon1000files,
#     'UL17_RSGluon1500':ul17RSGluon1500files,
#     'UL17_RSGluon2000':ul17RSGluon2000files,
#     'UL17_RSGluon2500':ul17RSGluon2500files,
#     'UL17_RSGluon3000':ul17RSGluon3000files,
#     'UL17_RSGluon3500':ul17RSGluon3500files,
#     'UL17_RSGluon4000':ul17RSGluon4000files,
#     'UL17_RSGluon4500':ul17RSGluon4500files,
#     'UL17_RSGluon5000':ul17RSGluon5000files,
#     'UL17_TTbar':ul17ttbarfiles,
#     'JetHT2017_Data':jetdatafiles2017,
    #-----------------------------------------#
#     'UL18_QCD':ul18qcdfiles,
#     'UL18_DM1000':ul18DM1000files,
#     'UL18_DM1500':ul18DM1500files,
#     'UL18_DM2000':ul18DM2000files,
#     'UL18_DM2500':ul18DM2500files,
#     'UL18_DM3000':ul18DM3000files,
#     'UL18_DM3500':ul18DM3500files,
#     'UL18_DM4000':ul18DM4000files,
#     'UL18_DM4500':ul18DM4500files,
#     'UL18_DM5000':ul18DM5000files,
#     'UL18_RSGluon1000':ul18RSGluon1000files,
#     'UL18_RSGluon1500':ul18RSGluon1500files,
#     'UL18_RSGluon2000':ul18RSGluon2000files,
#     'UL18_RSGluon2500':ul18RSGluon2500files,
#     'UL18_RSGluon3000':ul18RSGluon3000files,
#     'UL18_RSGluon3500':ul18RSGluon3500files,
#     'UL18_RSGluon4000':ul18RSGluon4000files,
#     'UL18_RSGluon4500':ul18RSGluon4500files,
#     'UL18_RSGluon5000':ul18RSGluon5000files,
#     'UL18_TTbar':ul18ttbarfiles,
#     'JetHT2018_Data':jetdatafiles2018
    #-----------------------------------------#
#     'QCD':qcdfiles,
#     'DM1000':DM1000files,
#     'DM1500':DM1500files,
#     'DM2000':DM2000files,
#     'DM2500':DM2500files,
#     'DM3000':DM3000files,
#     'DM3500':DM3500files,
#     'DM4000':DM4000files,
#     'DM4500':DM4500files,
#     'DM5000':DM5000files,
#     'RSGluon1000':RSGluon1000files,
#     'RSGluon1500':RSGluon1500files,
#     'RSGluon2000':RSGluon2000files,
#     'RSGluon2500':RSGluon2500files,
#     'RSGluon3000':RSGluon3000files,
#     'RSGluon3500':RSGluon3500files,
#     'RSGluon4000':RSGluon4000files,
#     'RSGluon4500':RSGluon4500files,
#     'RSGluon5000':RSGluon5000files,
#     'TTbar':ttbarfiles,
#     'JetHT_Data':jetdatafiles, # all years
# }


