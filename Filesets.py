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

#     REMINDER     #
# -----------------
# preVFP ==> APV
# postVFP ==> noAPV
# -----------------

qcdfilename = filedir + 'QCD/QCD_NanoAODv9_UL16_preVFP.txt'
with open(qcdfilename) as f:
    qcdfiles = [xrootdstr2 + s.strip() for s in f.readlines()]

ttbar700to1000filename = filedir + 'TT/TT_Mtt-700to1000_NanoAODv9_UL16_preVFP.txt'
with open(ttbar700to1000filename) as f:
    ttbar700to1000files = [xrootdstr2 + s.strip() for s in f.readlines()]
    
ttbar1000toInffilename = filedir + 'TT/TT_Mtt-1000toInf_NanoAODv9_UL16_preVFP.txt'
with open(ttbar1000toInffilename) as f:
    ttbar1000toInffiles = [xrootdstr2 + s.strip() for s in f.readlines()]
    
ttbarfiles = ttbar700to1000files + ttbar1000toInffiles # inclusion of both biased samples
    
ZprimeDMfilename = filedir + 'ZprimeDMToTTbar/ZprimeDMToTTbar_NanoAODv9_preVFP.txt'
with open(ZprimeDMfilename) as f:
    DM1000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp1000" in s]
with open(ZprimeDMfilename) as f:
    DM1500files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp1500" in s]
with open(ZprimeDMfilename) as f:
    DM2000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp2000" in s]
with open(ZprimeDMfilename) as f:
    DM2500files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp2500" in s]
with open(ZprimeDMfilename) as f:
    DM3000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp3000" in s]
with open(ZprimeDMfilename) as f:
    DM3500files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp3500" in s]
with open(ZprimeDMfilename) as f:
    DM4000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp4000" in s]
with open(ZprimeDMfilename) as f:
    DM4500files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp4500" in s]
with open(ZprimeDMfilename) as f:
    DM5000files = [xrootdstr2 + s.strip() for s in f.readlines() if "ResoIncl_MZp5000" in s]

RSGluonfilename = filedir + 'RSGluonToTT/RSGluonToTT_NanoAODv9_UL16_preVFP.txt'
with open(RSGluonfilename) as f:
    RSGluon1000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-1000" in s]
with open(RSGluonfilename) as f:
    RSGluon1500files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-1500" in s]
with open(RSGluonfilename) as f:
    RSGluon2000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-2000" in s]
with open(RSGluonfilename) as f:
    RSGluon2500files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-2500" in s]
with open(RSGluonfilename) as f:
    RSGluon3000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-3000" in s]
with open(RSGluonfilename) as f:
    RSGluon3500files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-3500" in s]
with open(RSGluonfilename) as f:
    RSGluon4000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-4000" in s]
with open(RSGluonfilename) as f:
    RSGluon4500files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-4500" in s]
with open(RSGluonfilename) as f:
    RSGluon5000files = [xrootdstr2 + s.strip() for s in f.readlines() if "RSGluonToTT_M-5000" in s]

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
    'QCD':qcdfiles,
    'DM1000':DM1000files,
    # 'DM1500':DM1500files,
    # 'DM2000':DM2000files,
    # 'DM2500':DM2500files,
    # 'DM3000':DM3000files,
    # 'DM3500':DM3500files,
    # 'DM4000':DM4000files,
    # 'DM4500':DM4500files,
    # 'DM5000':DM5000files,
    'RSGluon1000':RSGluon1000files,
    # 'RSGluon1500':RSGluon1500files,
    # 'RSGluon2000':RSGluon2000files,
    # 'RSGluon2500':RSGluon2500files,
    # 'RSGluon3000':RSGluon3000files,
    # 'RSGluon3500':RSGluon3500files,
    # 'RSGluon4000':RSGluon4000files,
    # 'RSGluon4500':RSGluon4500files,
    # 'RSGluon5000':RSGluon5000files,
    'TTbar':ttbarfiles,
    # 'TTbar_biased_700to1000': ttbar700to1000files,
    # 'TTbar_biased_1000toInf': ttbar1000toInffiles,
    # 'JetHT':jetdatafiles,
    'JetHT2016_Data':jetdatafiles2016,
    # 'JetHT2017_Data':jetdatafiles2017,
    # 'JetHT2018_Data':jetdatafiles2018
}

filesets_forweights = {
    # 'QCD':qcdfiles,
    # 'DM1000':DM1000files,
    # 'DM1500':DM1500files,
    # 'DM2000':DM2000files,
    # 'DM2500':DM2500files,
    # 'DM3000':DM3000files,
    # 'DM3500':DM3500files,
    # 'DM4000':DM4000files,
    # 'DM4500':DM4500files,
    # 'DM5000':DM5000files,
    # 'RSGluon1000':RSGluon1000files,
    # 'RSGluon1500':RSGluon1500files,
    # 'RSGluon2000':RSGluon2000files,
    # 'RSGluon2500':RSGluon2500files,
    # 'RSGluon3000':RSGluon3000files,
    # 'RSGluon3500':RSGluon3500files,
    # 'RSGluon4000':RSGluon4000files,
    # 'RSGluon4500':RSGluon4500files,
    # 'RSGluon5000':RSGluon5000files,
    # 'TTbar':ttbarfiles, # ttbarfiles to be subtracted from all years of JetHT data
    # 'TTbar_2016':ttbarfiles, # ttbarfiles to be subtracted from JetHT2016 data
    # 'TTbar_2017':ttbarfiles, # ttbarfiles to be subtracted from JetHT2017 data
    # 'TTbar_2018':ttbarfiles, # ttbarfiles to be subtracted from JetHT2018 data
    # 'JetHT':jetdatafiles, # all years
    # 'JetHT2016_Data':jetdatafiles2016,
    # 'JetHT2017_Data':jetdatafiles2017,
    # 'JetHT2018_Data':jetdatafiles2018
}

