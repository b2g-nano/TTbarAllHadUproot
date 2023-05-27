#!/usr/bin/env python
# coding: utf-8

import coffea
from coffea import util
import numpy as np
import pandas as pd
import itertools
import os


# ## analysis categories

# analysis categories #
# ttagcats = ["AT&Pt", "at", "pret", "0t", "1t", ",>=1t", "2t", ">=0t"]
ttagcats = ["at", "pret", "2t"]
btagcats = ["0b", "1b", "2b"]
ycats = ['cen', 'fwd']

anacats = [ t+b+y for t,b,y in itertools.product( ttagcats, btagcats, ycats) ]
label_dict = {i: label for i, label in enumerate(anacats)}
label_to_int_dict = {label: i for i, label in enumerate(anacats)}


# ## directories for saving files

save_csv_filename = 'mistag_rate.csv'

# ## load coffea files

coffea_dir = 'outputs/'
coffeaFiles = {
    "JetHT":{
        "2016APV": {
            "B": coffea_dir+'JetHT_2016APVB.coffea',
            "C": coffea_dir+'JetHT_2016APVC.coffea',
            "D": coffea_dir+'JetHT_2016APVD.coffea',
            "E": coffea_dir+'JetHT_2016APVE.coffea',
        },
        "2016": {
            "F": coffea_dir+'JetHT_2016F.coffea',
            "G": coffea_dir+'JetHT_2016G.coffea',
            "H": coffea_dir+'JetHT_2016H.coffea',
        },
        "2017": '',
        "2018": ''
    },
    
    "TTbar": {
        "2016APV": {
            "700to1000": coffea_dir+'TTbar_2016APV_700to1000.coffea',
            "1000toInf": coffea_dir+'TTbar_2016APV_1000toInf.coffea',
        },
        "2016": {
            "700to1000": coffea_dir+'TTbar_2016_700to1000.coffea',
            "1000toInf": coffea_dir+'TTbar_2016_1000toInf.coffea',
        },
        "2017": {
            "700to1000": '',
            "1000toInf": '',
        },
        "2018": {
            "700to1000": '',
            "1000toInf": '',
        }
    }
}


# ## manual jet $p$ bins
pbins = np.array([ 400.,  500.,  600.,  800., 1000., 1500., 2000., 3000., 7000.])


# scale factors
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable

luminosity = {
    "2016APV": 19800.,
    "2016": 16120., #35920 - 19800
    "2017": 41530.,
    "2018": 59740.
}

ttbar_xs = {}
ttbar_xs["700to1000"] = 831.76 * (0.09210)
ttbar_xs["1000toInf"] = 831.76 * (0.02474)
toptag_sf = 0.9
toptag_kf = 0.7


# ## calculate mistag rate

IOVs = ['2016'] #['2016APV', '2016']


for IOV in IOVs:
    
    ttbar_700to1000 = util.load(coffeaFiles['TTbar'][IOV]['700to1000'])
    ttbar_1000toInf = util.load(coffeaFiles['TTbar'][IOV]['1000toInf'])
    print('loading', coffeaFiles['TTbar'][IOV]['700to1000'])
    print('loading', coffeaFiles['TTbar'][IOV]['1000toInf'])
    
    ttbar_evts = {}
    ttbar_evts["700to1000"] = ttbar_700to1000['cutflow']['sumw']
    ttbar_evts["1000toInf"] = ttbar_1000toInf['cutflow']['sumw']
    
    ttbar_SF = {}
    ttbar_SF["700to1000"] = luminosity[IOV] * ttbar_xs["700to1000"] * toptag_sf**2 * toptag_kf / ttbar_evts["1000toInf"]
    ttbar_SF["1000toInf"] = luminosity[IOV] * ttbar_xs["1000toInf"] * toptag_sf**2 * toptag_kf / ttbar_evts["1000toInf"]

    save_csv_filename = f'mistag_rate_{IOV}.csv'

    jetht_files = []

    for era, file in coffeaFiles['JetHT'][IOV].items():
        if os.path.isfile(file):
            print('loading', file)
            jetht_files.append(util.load(file))
            
            
    




    mistag_rate_dict = {}
    mistag_rate_dict_all = {}
    mistag_rate_dict["jetp bins"] = pbins
    mistag_rate_dict_all["jetp bins"] = pbins

    for i, label in label_dict.items():

        jetht_numerator   = jetht_files[0]['numerator'][{'anacat':i}].values()
        jetht_denominator = jetht_files[0]['denominator'][{'anacat':i}].values()

        for file in jetht_files[1:]:
            jetht_numerator   += file['numerator'][{'anacat':i}].values()
            jetht_denominator += file['denominator'][{'anacat':i}].values()

        ttbar_numerator   = ttbar_700to1000['numerator'][{'anacat':i}].values() * ttbar_SF['700to1000']
        ttbar_denominator = ttbar_700to1000['denominator'][{'anacat':i}].values() * ttbar_SF['700to1000']

        ttbar_numerator   += ttbar_1000toInf['numerator'][{'anacat':i}].values() * ttbar_SF['1000toInf']
        ttbar_denominator += ttbar_1000toInf['denominator'][{'anacat':i}].values() * ttbar_SF['1000toInf']

        mistag_rate = (jetht_numerator - ttbar_numerator) / (jetht_denominator - ttbar_denominator)

        mistag_rate[np.isnan(mistag_rate)] = 0.

        mistag_rate_dict_all[label] = mistag_rate.flatten()
        
        
        if 'at' in label and 'cen' in label:
            
            # get info for forward and central y regions together (anacat = i+1)
            
            jetht_numerator  += jetht_files[0]['numerator'][{'anacat':i+1}].values()
            jetht_denominator += jetht_files[0]['denominator'][{'anacat':i+1}].values()

            for file in jetht_files[1:]:
                jetht_numerator   += file['numerator'][{'anacat':i+1}].values()
                jetht_denominator += file['denominator'][{'anacat':i+1}].values()

            ttbar_numerator   += ttbar_700to1000['numerator'][{'anacat':i+1}].values() * ttbar_SF['700to1000']
            ttbar_denominator += ttbar_700to1000['denominator'][{'anacat':i+1}].values() * ttbar_SF['700to1000']

            ttbar_numerator   += ttbar_1000toInf['numerator'][{'anacat':i+1}].values() * ttbar_SF['1000toInf']
            ttbar_denominator += ttbar_1000toInf['denominator'][{'anacat':i+1}].values() * ttbar_SF['1000toInf']
            
            
            numerator = jetht_numerator - ttbar_numerator
            denominator = jetht_denominator - ttbar_denominator
            
            
            
            # calculate y inclusive mistag rate
            mistag_rate_yinc = numerator / denominator
            mistag_rate_yinc[np.isnan(mistag_rate_yinc)] = 0.
            mistag_rate_dict[label.replace('at','').replace('cen','')] = mistag_rate_yinc.flatten()
                        
            mistag_rate_err = (numerator/denominator) * np.sqrt((1/numerator + 1/denominator))
            
            
            mistag_rate_err[np.isnan(mistag_rate_err)] = 0.
            mistag_rate_dict[label.replace('at','').replace('cen','')+'err'] = mistag_rate_err.flatten()
    

    
    df_mistag_all = pd.DataFrame(data=mistag_rate_dict_all)
    df_mistag_all.to_csv('mistag/'+save_csv_filename)
    df_mistag_all.to_csv('data/corrections/backgroundEstimate/' + save_csv_filename)
    
    
    df_mistag = pd.DataFrame(data=mistag_rate_dict)
    df_mistag.to_csv('mistag/'+save_csv_filename.replace('.csv', '_inc.csv'))
    
    print('saving', 'mistag/'+save_csv_filename.replace('.csv', '_inc.csv'))
    print('saving', 'mistag/'+save_csv_filename)
    
    # save copy for running uproot job
    print('saving copy to', 'data/corrections/backgroundEstimate/' + save_csv_filename)
    

        


