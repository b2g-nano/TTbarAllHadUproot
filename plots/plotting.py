# plotting.py

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
from scipy.optimize import curve_fit
import uproot

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

sys.path.append('../python/')
from functions import (
    loadCoffeaFile, 
    getLabelMap, 
    getCoffeaFilenames, 
    plotBackgroundEstimate, 
    getHist, 
    makePlotDirectories,
    lumi
)


# initialize

IOV = '2016' # choices: 2016APV, 2016, 2017, 2018

makePlotDirectories()

label_map = getLabelMap()
label_to_int = {label: i for i, label in label_map.items()}

signal_cats = [ i for label, i in label_to_int.items() if '2t' in label]
pretag_cats = [ i for label, i in label_to_int.items() if 'pre' in label]
antitag_cats = [ i for label, i in label_to_int.items() if 'at' in label]

systematics = ['nominal','pileup','pdf','btag', 'jes', 'jer']


# categories and systematics labels

cats = ['0bcen', '0bfwd', '1bcen', '1bfwd', '2bcen', '2bfwd']
cat_labels = ['cen0b', 'fwd0b', 'cen1b', 'fwd1b', 'cen2b', 'fwd2b']


syst_labels = ['nominal']
for s in systematics:
    if not 'nominal' in s:
        syst_labels.append(s+'Down')
        syst_labels.append(s+'Up')

        
        
# scale factors

lumi = {
    "2016APV": 19800.,
    "2016": 16120., #35920 - 19800
    "2016all": 35920,
    "2017": 41530.,
    "2018": 59740.
}

t_BR = 0.6741
ttbar_BR = 0.4544 #PDG 2019
ttbar_xs1 = 831.76 * (0.09210) #pb For ttbar mass from 700 to 1000
ttbar_xs2 = 831.76 * (0.02474) #pb For ttbar mass from 1000 to Inf
toptag_sf = 0.9
toptag_kf = 1.0 #0.7
qcd_xs = 13700000.0 #pb From https://cms-gen-dev.cern.ch/xsdb

zprime_xs = {
    '1000': 2.222,
    '1500': 0.387,
    '2000': 0.09428,
    '2500': 0.0279,
    '3000': 0.009327,
    '3500': 0.003507,
    '4000': 0.001484,
    '4500': 0.0007087,
    '5000': 0.0003801,
}

RSGluon_xs = {
    '1000': 21.03,
    '1500': 3.656,
    '2000': 0.9417,
    '2500': 0.3039,
    '3000': 0.1163,
    '3500': 0.05138,
    '4000': 0.02556,
    '4500': 0.01422,
    '5000': 0.008631,
}


# RS Gluon files

RSGluon1000file = util.load('../outputs/RSGluon1000_2016.coffea')
RSGluon1500file = util.load('../outputs/RSGluon1500_2016.coffea')
RSGluon2000file = util.load('../outputs/RSGluon2000_2016.coffea')
RSGluon2500file = util.load('../outputs/RSGluon2500_2016.coffea')
RSGluon3000file = util.load('../outputs/RSGluon3000_2016.coffea')
RSGluon3500file = util.load('../outputs/RSGluon3500_2016.coffea')
RSGluon4000file = util.load('../outputs/RSGluon4000_2016.coffea')
RSGluon4500file = util.load('../outputs/RSGluon4500_2016.coffea')
RSGluon5000file = util.load('../outputs/RSGluon5000_2016.coffea')


RSGluonFiles = {
    "1000": RSGluon1000file,
    "1500": RSGluon1500file,
    "2000": RSGluon2000file,
    "2500": RSGluon2500file,
    "3000": RSGluon3000file,
    "3500": RSGluon3500file,
    "4000": RSGluon4000file,
    "4500": RSGluon4500file,
    "5000": RSGluon5000file,
    
}



        
# systematics plots #

print('\nPlotting systematics\n')


for cat, catname in zip(cats, cat_labels):
    
    signal_cat = label_to_int['2t'+cat]

    for syst in systematics[1:]:

        fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])

        text = r'MC TTbar'+'\n'+syst+' systematic variations'
        
        dytext = ''
        if 'cen' in cat:
            dytext = r'$\Delta y$ < 1.0'
        elif 'fwd' in cat:
            dytext = r'$\Delta y$ > 1.0'

        btext = ''
        if '0b' in cat:
            btext = '0 b-tags'
        elif '1b' in cat:
            btext = '1 b-tag'
        elif '2b' in cat:
            btext = '2 b-tags'

        text = f'MC TTbar\n{syst} systematic variations\n{btext}, {dytext}'
        
        
        hep.cms.label('Preliminary', data=True, lumi='{0:0.1f}'.format(lumi['2016']/1000.), year='2016', loc=1, fontsize=20, ax=ax1)
        hep.cms.text(text, loc=2, fontsize=20, ax=ax1)



        httbar = getHist('ttbarmass', 'TTbar', False, IOV, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats,'systematic':'nominal'})    
        httbarUp = getHist('ttbarmass', 'TTbar', False, IOV, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats,'systematic':syst+'Up'})
        httbarDn = getHist('ttbarmass', 'TTbar', False, IOV, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats,'systematic':syst+'Down'})

        hep.histplot(httbar, histtype='step', color='k', ax=ax1, label='Nominal')
        hep.histplot(httbarUp, histtype='step', color='green', ax=ax1, label='Up')
        hep.histplot(httbarDn, histtype='step', color='red', ax=ax1, label='Down')


        ratioUp = httbarUp / httbar.values()
        ratioDn = httbarDn / httbar.values()

        hep.histplot(ratioUp, histtype='step', color='green', ax=ax2)
        hep.histplot(ratioDn, histtype='step', color='red', ax=ax2)
        ax2.axhline(1, color='black', ls='--')

        ax2.set_ylabel('Syst/Nom')
        ax2.set_xlabel(ax1.get_xlabel())
        ax1.set_xlabel('')
        ax1.set_ylim(1e-1,100)
        ax2.set_ylim(0.5,1.5)


        ax1.legend()
        
        imagefile = f'images/png/systematics/{IOV}/TTbar_{catname}_{syst}.png'

        plt.savefig(imagefile)
        plt.savefig(imagefile.replace('png','pdf'))
        print('saving ', imagefile)
        print('saving ', imagefile.replace('png','pdf'))

        ax1.plot()

    



masses = ['1000', '5000']

for mass in masses:

    for cat, catname in zip(cats, cat_labels):

        signal_cat = label_to_int['2t'+cat]

        for syst in systematics[1:]:

            fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])

            dytext = ''
            if 'cen' in cat:
                dytext = r'$\Delta y$ < 1.0'
            elif 'fwd' in cat:
                dytext = r'$\Delta y$ > 1.0'
                
            btext = ''
            if '0b' in cat:
                btext = '0 b-tags'
            elif '1b' in cat:
                btext = '1 b-tag'
            elif '2b' in cat:
                btext = '2 b-tags'

            text = f'RSGluon {int(mass)/1000} TeV\n{syst} systematic variations\n{btext}, {dytext}'


            hep.cms.label('Preliminary', data=True, lumi='{0:0.1f}'.format(lumi['2016']/1000.), year='2016', loc=1, fontsize=20, ax=ax1)
            hep.cms.text(text, loc=2, fontsize=20, ax=ax1)

            hRSGluon   = RSGluonFiles[mass]['mtt_vs_mt'][{'anacat':signal_cats}][{'anacat':sum,'systematic':'nominal'}][{'jetmass':sum}] * lumi['2016'] * RSGluon_xs[mass] / RSGluon1000file['cutflow']['sumw']
            hRSGluonUp = RSGluonFiles[mass]['mtt_vs_mt'][{'anacat':signal_cats}][{'anacat':sum,'systematic':syst+'Up'}][{'jetmass':sum}] * lumi['2016'] * RSGluon_xs[mass] / RSGluon1000file['cutflow']['sumw']
            hRSGluonDn = RSGluonFiles[mass]['mtt_vs_mt'][{'anacat':signal_cats}][{'anacat':sum,'systematic':syst+'Down'}][{'jetmass':sum}] * lumi['2016'] * RSGluon_xs[mass] / RSGluon1000file['cutflow']['sumw']


            hep.histplot(hRSGluon, histtype='step', color='k', ax=ax1, label='Nominal')
            hep.histplot(hRSGluonUp, histtype='step', color='green', ax=ax1, label='Up')
            hep.histplot(hRSGluonDn, histtype='step', color='red', ax=ax1, label='Down')


            ratioUp = hRSGluonUp / hRSGluon.values()
            ratioDn = hRSGluonDn / hRSGluon.values()

            hep.histplot(ratioUp, histtype='step', color='green', ax=ax2)
            hep.histplot(ratioDn, histtype='step', color='red', ax=ax2)
            ax2.axhline(1, color='black', ls='--')

            ax2.set_ylabel('Syst/Nom')
            ax2.set_xlabel(ax1.get_xlabel())
            ax1.set_xlabel('')
            ax1.set_ylim(1e-1,1500)
            ax2.set_ylim(0.5,1.5)


            ax1.legend()

            imagefile = f'images/png/systematics/{IOV}/RSGluon{mass}_{catname}_{syst}.png'

            plt.savefig(imagefile)
            plt.savefig(imagefile.replace('png','pdf'))
            print('saving ', imagefile)
            print('saving ', imagefile.replace('png','pdf'))

            ax1.plot()







   
