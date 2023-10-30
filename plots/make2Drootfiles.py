# make2Drootfiles.py



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep as hep
hep.style.use("CMS")
from coffea import util
import itertools
import os, sys
import glob
import copy
import uproot
import time

sys.path.append('../python/')
import functions

functions.makeSaveDirectories()


label_map = functions.getLabelMap()
label_to_int = {label: i for i, label in label_map.items()}
signal_cats = [ i for label, i in label_to_int.items() if '2t' in label]
pretag_cats = [ i for label, i in label_to_int.items() if 'pre' in label]
antitag_cats = [ i for label, i in label_to_int.items() if 'at' in label]




toc = time.time()

if (len(sys.argv) > 1) and (sys.argv[1] in ['2016', '2016APV', '2016all', '2017', '2018', 'all']):
    
    year = sys.argv[1]

else:

    year = '2016'

systematics = ['nominal', 'jes', 'jer', 'pileup', 'pdf', 'q2', 'btag', 'prefiring']
syst_labels = ['nominal']
if '2018' in year: 
    systematics = ['nominal', 'jes', 'jer', 'pileup', 'pdf', 'q2', 'btag']

for s in systematics:
    if not 'nominal' in s and not 'hem' in s:
        syst_labels.append(s+'Down')
        syst_labels.append(s+'Up')
        
print(syst_labels)


tag = ''

dataOnly = False
inclusive = False

if inclusive:
    
    cats, cat_labels = [''], ['']
    
else:

    cats = ['', 'cen', 'fwd', '0bcen', '0bfwd', '1bcen', '1bfwd', '2bcen', '2bfwd']
    cat_labels = ['', 'cen', 'fwd', 'cen0b', 'fwd0b', 'cen1b', 'fwd1b', 'cen2b', 'fwd2b']
    
#     cats = ['cen', 'fwd']
#     cat_labels = ['cen', 'fwd']

signals = []
# signals = ['RSGluon2000']
# signals = ['RSGluon'+str(int(b*100)) for b in [10,15,20,25,30,35,40,45,50,55,60]]
# signals += ['ZPrime'+str(int(b*100))+'_1' for b in [10,12,14,16,18,20,25,30,35,40,45]]
# signals += ['ZPrime'+str(int(b*100))+'_10' for b in [10,12,14,16,18,20,25,30,35,40,45,50,60,70]]
# signals += ['ZPrime'+str(int(b*100))+'_30' for b in [10,12,14,16,18,20,25,30,35,40,45,50,60,70]]
# signals += ['ZPrime'+str(int(b*100))+'_DM' for b in [10,15,20,25,30,35,40,45,50]]

savefileheader = '../outputs/twodalphabet/TTbarAllHad{}_'.format(year.replace('20', '').replace('all',''))
                                                                
fdata  = uproot.recreate(savefileheader+'Data'+tag+'.root')

if not dataOnly:
    
    fttbar = uproot.recreate(savefileheader+'TTbar'+tag+'.root')
    sigfiles = [uproot.recreate(savefileheader+'signal'+sig+tag+'.root') for sig in signals ]

for cat, catname in zip(cats, cat_labels):
    
    if cat == '':
        
        signal_cats = [ i for label, i in label_to_int.items() if '2t' in label]
        antitag_cats = [ i for label, i in label_to_int.items() if 'at' in label]
        sum_axes = ['anacat']

    elif 'b' in cat :
        
        print('b cats')
        
        signal_cats = label_to_int['2t'+cat]
        antitag_cats = label_to_int['at'+cat]
        sum_axes = []
        
    else:
        
        
        signal_cats = []
        antitag_cats = []
        
        for label, i in label_to_int.items():
            
            if '2t' in label and cat in label:
                signal_cats.append(i)
            if 'at' in label and cat in label:
                antitag_cats.append(i)
                
        
        sum_axes = ['anacat']
        
        
    
    for syst in syst_labels:
        print(syst, cat)

        integrate_pass = {'anacat':signal_cats, 'systematic': syst}
        integrate_fail = {'anacat':antitag_cats, 'systematic': syst}

        systname = syst.upper()[:-2] + 'up' if 'Up' in syst else syst.upper()[:-4] + 'down'

        if 'nominal' in syst:

            systname = ''
            hdata_pass = functions.getHist2('mtt_vs_mt', 'JetHT', year, sum_axes=sum_axes, integrate_axes=integrate_pass, tag=tag) 
            hdata_fail = functions.getHist2('mtt_vs_mt', 'JetHT', year, sum_axes=sum_axes, integrate_axes=integrate_fail, tag=tag) 

            fdata["MttvsMt"+catname+"Pass"+systname] = hdata_pass
            fdata["MttvsMt"+catname+"Fail"+systname] = hdata_fail

        if not dataOnly:
            
            sig_pass = [functions.getHist2('mtt_vs_mt', sig, year, sum_axes=sum_axes, integrate_axes=integrate_pass, tag=tag) for sig in signals]
            sig_fail = [functions.getHist2('mtt_vs_mt', sig, year, sum_axes=sum_axes, integrate_axes=integrate_fail, tag=tag) for sig in signals]

            httbar_pass = functions.getHist2('mtt_vs_mt', 'TTbar', year, sum_axes=sum_axes, integrate_axes=integrate_pass, tag=tag) 
            httbar_fail = functions.getHist2('mtt_vs_mt', 'TTbar', year, sum_axes=sum_axes, integrate_axes=integrate_fail, tag=tag) 


            # save hists

            fttbar["MttvsMt"+catname+"Pass"+systname] = httbar_pass
            fttbar["MttvsMt"+catname+"Fail"+systname] = httbar_fail
            
            
            for i, file in enumerate(sigfiles):
                
                file["MttvsMt"+catname+"Pass"+systname] = sig_pass[i]
                file["MttvsMt"+catname+"Fail"+systname] = sig_fail[i]


fdata.close()
                                                                
print('saving '+savefileheader+'Data'+tag+'.root')

if not dataOnly:
    
    fttbar.close()
    for file in sigfiles:
        
        file.close()
        
    
    print('saving '+savefileheader+'TTbar'+tag+'.root')
    
    for sig in signals:
        print('saving '+savefileheader+sig+tag+'.root')

        
        
tic = time.time()
print()
functions.printTime(tic-toc)
