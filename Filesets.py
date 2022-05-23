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
VFP = '_postVFP.txt'

ul16qcdfilename = filedir + 'QCD/QCD_NanoAODv9_UL16' + VFP
with open(ul16qcdfilename) as f:
    ul16qcdfiles = [xrootdstr2 + s.strip() for s in f.readlines()]

ul16ttbar700to1000filename = filedir + 'TT/TT_Mtt-700to1000_NanoAODv9_UL16' + VFP
with open(ul16ttbar700to1000filename) as f:
    ul16ttbar700to1000files = [xrootdstr2 + s.strip() for s in f.readlines()]
    
ul16ttbar1000toInffilename = filedir + 'TT/TT_Mtt-1000toInf_NanoAODv9_UL16' + VFP
with open(ul16ttbar1000toInffilename) as f:
    ul16ttbar1000toInffiles = [xrootdstr2 + s.strip() for s in f.readlines()]
    
ul16ttbarfiles = ul16ttbar700to1000files + ul16ttbar1000toInffiles # inclusion of both biased samples
    
ul16ZprimeDMfilename = filedir + 'ZprimeDMToTTbar/ZprimeDMToTTbar_NanoAODv9_UL16' + VFP
with open(ul16ZprimeDMfilename) as f:
    ul16DM1000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp1000" in s]
with open(ul16ZprimeDMfilename) as f:
    ul16DM1500files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp1500" in s]
with open(ul16ZprimeDMfilename) as f:
    ul16DM2000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp2000" in s]
with open(ul16ZprimeDMfilename) as f:
    ul16DM2500files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp2500" in s]
with open(ul16ZprimeDMfilename) as f:
    ul16DM3000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp3000" in s]
with open(ul16ZprimeDMfilename) as f:
    ul16DM3500files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp3500" in s]
with open(ul16ZprimeDMfilename) as f:
    ul16DM4000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp4000" in s]
with open(ul16ZprimeDMfilename) as f:
    ul16DM4500files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp4500" in s]
with open(ul16ZprimeDMfilename) as f:
    ul16DM5000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp5000" in s]

ul16RSGluonfilename = filedir + 'RSGluonToTT/RSGluonToTT_NanoAODv9_UL16' + VFP
with open(ul16RSGluonfilename) as f:
    ul16RSGluon1000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-1000" in s]
with open(ul16RSGluonfilename) as f:
    ul16RSGluon1500files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-1500" in s]
with open(ul16RSGluonfilename) as f:
    ul16RSGluon2000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-2000" in s]
with open(ul16RSGluonfilename) as f:
    ul16RSGluon2500files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-2500" in s]
with open(ul16RSGluonfilename) as f:
    ul16RSGluon3000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-3000" in s]
with open(ul16RSGluonfilename) as f:
    ul16RSGluon3500files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-3500" in s]
with open(ul16RSGluonfilename) as f:
    ul16RSGluon4000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-4000" in s]
with open(ul16RSGluonfilename) as f:
    ul16RSGluon4500files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-4500" in s]
with open(ul16RSGluonfilename) as f:
    ul16RSGluon5000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-5000" in s]

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
            
jetdatafiles = jetdatafiles2016 + jetdatafiles2017 + jetdatafiles2018 # All data after unblinding

""" Comment out whichever files you wish to not be included """

filesets = {
    'UL16_QCD':ul16qcdfiles,
    'UL16_DM1000':ul16DM1000files,
    'UL16_DM1500':ul16DM1500files,
    'UL16_DM2000':ul16DM2000files,
    'UL16_DM2500':ul16DM2500files,
    'UL16_DM3000':ul16DM3000files,
    'UL16_DM3500':ul16DM3500files,
    'UL16_DM4000':ul16DM4000files,
    'UL16_DM4500':ul16DM4500files,
    'UL16_DM5000':ul16DM5000files,
    'UL16_RSGluon1000':ul16RSGluon1000files,
    'UL16_RSGluon1500':ul16RSGluon1500files,
    'UL16_RSGluon2000':ul16RSGluon2000files,
    'UL16_RSGluon2500':ul16RSGluon2500files,
    'UL16_RSGluon3000':ul16RSGluon3000files,
    'UL16_RSGluon3500':ul16RSGluon3500files,
    'UL16_RSGluon4000':ul16RSGluon4000files,
    'UL16_RSGluon4500':ul16RSGluon4500files,
    'UL16_RSGluon5000':ul16RSGluon5000files,
    'UL16_TTbar':ul16ttbarfiles,
    'JetHT2016_Data':jetdatafiles2016,
#     'JetHT2017_Data':jetdatafiles2017,
#     'JetHT2018_Data':jetdatafiles2018
#     'JetHT_Data':jetdatafiles, # all years
}

# filesets_forweights = {
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
#     'UL16_TTbar':ul16ttbarfiles, # ttbarfiles to be subtracted from all years of JetHT data
#     'JetHT2016_Data':jetdatafiles2016,
#     'JetHT2017_Data':jetdatafiles2017,
#     'JetHT2018_Data':jetdatafiles2018
#     'JetHT':jetdatafiles, # all years
# }

