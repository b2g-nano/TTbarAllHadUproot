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

xrootdstr1 = 'root://cmseos.fnal.gov//'
xrootdstr2 = 'root://cmsxrootd.fnal.gov//'
xrootdstr3 = 'root://cmsxrootd-site.fnal.gov/'

qcdfilename = 'TTbarAllHadUproot/QCD.txt'
#qcdfilename = 'QCD.txt'
with open(qcdfilename) as f:
    qcdfiles = [xrootdstr2 + s.strip() for s in f.readlines()]

ttbarfilename = 'TTbarAllHadUproot/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8.txt'
#ttbarfilename = 'TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8.txt'
with open(ttbarfilename) as f:
    ttbarfiles = [xrootdstr2 + s.strip() for s in f.readlines()]

ZprimeDMfilename = 'TTbarAllHadUproot/ZprimeDMToTTbar.txt'
#ZprimeDMfilename = 'ZprimeDMToTTbar.txt'
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

jetdatafilename = 'TTbarAllHadUproot/JetHT_Data.txt'
with open(jetdatafilename) as f:
    jetdatafiles = [xrootdstr2 + s.strip() for s in f.readlines()[::3]] # Every third datafile
with open(jetdatafilename) as g:
    jetdatafiles2016 = [xrootdstr2 + s.strip() for s in g.readlines() if "/store/data/Run2016" in s]
with open(jetdatafilename) as h:
    jetdatafiles2017 = [xrootdstr2 + s.strip() for s in h.readlines() if "/store/data/Run2017" in s]
with open(jetdatafilename) as i:
    jetdatafiles2018 = [xrootdstr2 + s.strip() for s in i.readlines() if "/store/data/Run2018" in s]

""" Comment out whichever files you wish to not be included """

filesets = {
    'QCD':qcdfiles,
    'DM1000':DM1000files,
    'DM1500':DM1500files,
    'DM2000':DM2000files,
    'DM2500':DM2500files,
    'DM3000':DM3000files,
    'DM3500':DM3500files,
    'DM4000':DM4000files,
    'DM4500':DM4500files,
    'DM5000':DM5000files,
    'TTbar':ttbarfiles,
    'JetHT':jetdatafiles,
    'JetHT2016_Data':jetdatafiles2016,
    'JetHT2017_Data':jetdatafiles2017,
    'JetHT2018_Data':jetdatafiles2018
}

filesets_forweights = {
    'QCD':qcdfiles,
    'DM1000':DM1000files,
    'DM1500':DM1500files,
    'DM2000':DM2000files,
    'DM2500':DM2500files,
    'DM3000':DM3000files,
    'DM3500':DM3500files,
    'DM4000':DM4000files,
    'DM4500':DM4500files,
    'DM5000':DM5000files,
    'TTbar':ttbarfiles,
    'TTbar_2016':ttbarfiles, # ttbarfiles to be subtracted from JetHT2016 data
    'TTbar_2017':ttbarfiles, # ttbarfiles to be subtracted from JetHT2016 data
    'TTbar_2018':ttbarfiles, # ttbarfiles to be subtracted from JetHT2016 data
    'JetHT':jetdatafiles,
    'JetHT2016_Data':jetdatafiles2016,
    'JetHT2017_Data':jetdatafiles2017,
    'JetHT2018_Data':jetdatafiles2018
}





