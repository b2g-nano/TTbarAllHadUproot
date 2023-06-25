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

sys.path.append('../python/')
import functions

# suppress warnings
import warnings
warnings.filterwarnings("ignore")



# scale factors and luminosity

lumi = functions.lumi
rsgluon_xs = functions.rsgluon_xs



# initialize

functions.makeSaveDirectories()

IOV = '2016' # choices: 2016APV, 2016, 2017, 2018

label_map = functions.getLabelMap()
label_to_int = {label: i for i, label in label_map.items()}

signal_cats = [ i for label, i in label_to_int.items() if '2t' in label]
pretag_cats = [ i for label, i in label_to_int.items() if 'pre' in label]
antitag_cats = [ i for label, i in label_to_int.items() if 'at' in label]



# categories and systematics labels

cats = ['0bcen', '0bfwd', '1bcen', '1bfwd', '2bcen', '2bfwd']
cat_labels = ['cen0b', 'fwd0b', 'cen1b', 'fwd1b', 'cen2b', 'fwd2b']

systematics = ['nominal', 'jes', 'jer', 'pileup', 'pdf', 'q2', 'btag', 'prefiring']
syst_labels = ['nominal']
for s in systematics:
    if not 'nominal' in s:
        syst_labels.append(s+'Down')
        syst_labels.append(s+'Up')

        

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

def plotSystematics():

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


            hep.cms.label('Preliminary', data=True, lumi='{0:0.1f}'.format(lumi[IOV]/1000.), year=IOV.replace('all',''), loc=1, fontsize=20, ax=ax1)
            hep.cms.text(text, loc=2, fontsize=20, ax=ax1)



            httbar = functions.getHist('ttbarmass', 'TTbar', False, IOV, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats,'systematic':'nominal'})    
            httbarUp = functions.getHist('ttbarmass', 'TTbar', False, IOV, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats,'systematic':syst+'Up'})
            httbarDn = functions.getHist('ttbarmass', 'TTbar', False, IOV, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats,'systematic':syst+'Down'})

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


                hep.cms.label('Preliminary', data=True, lumi='{0:0.1f}'.format(lumi[IOV]/1000.), year=IOV.replace('all',''), loc=1, fontsize=20, ax=ax1)
                hep.cms.text(text, loc=2, fontsize=20, ax=ax1)

                hRSGluon   = RSGluonFiles[mass]['mtt_vs_mt'][{'anacat':signal_cats}][{'anacat':sum,'systematic':'nominal'}][{'jetmass':sum}] * lumi[IOV] * rsgluon_xs[mass] / RSGluon1000file['cutflow']['sumw']
                hRSGluonUp = RSGluonFiles[mass]['mtt_vs_mt'][{'anacat':signal_cats}][{'anacat':sum,'systematic':syst+'Up'}][{'jetmass':sum}] * lumi[IOV] * rsgluon_xs[mass] / RSGluon1000file['cutflow']['sumw']
                hRSGluonDn = RSGluonFiles[mass]['mtt_vs_mt'][{'anacat':signal_cats}][{'anacat':sum,'systematic':syst+'Down'}][{'jetmass':sum}] * lumi[IOV] * rsgluon_xs[mass] / RSGluon1000file['cutflow']['sumw']


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





                
def plotClosureTest():
    
    
    print('\nPlotting Closure Test\n')
    

    fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])



    hbkg = functions.getHist2('ttbarmass', 'JetHT', IOV,
             sum_axes=['anacat'],
             integrate_axes={'systematic':'nominal', 'anacat':antitag_cats},
             tag = '_bkgest'

            )

    httbar = functions.getHist2('ttbarmass', 'TTbar', IOV,
             sum_axes=['anacat'],
             integrate_axes={'systematic':'nominal', 'anacat':signal_cats},        
            )

    hsig = functions.getHist2('ttbarmass', 'JetHT', IOV,
             sum_axes=['anacat'],
             integrate_axes={'systematic':'nominal', 'anacat':signal_cats}        
            )

    text = 'Preliminary'+'\n'+r'$\Delta y$ inclusive'+', '+r'b-tag inclusive'

    hep.cms.label('', data=True, lumi='{0:0.1f}'.format(lumi[IOV]/1000), year=IOV, loc=2, fontsize=20, ax=ax1)
    hep.cms.text(text, loc=2, fontsize=20, ax=ax1)

    hep.histplot(hsig, histtype='errorbar', color='black', label='Data', ax=ax1)
    hep.histplot(hbkg, histtype='fill', color='xkcd:pale gold', label='NTMJ Bkg Est', ax=ax1)
    hep.histplot(httbar, histtype='fill', color='xkcd:deep red', label='SM TTbar', ax=ax1)




    ratio_plot =  hsig / hbkg.values()
    hep.histplot(ratio_plot, ax=ax2, histtype='errorbar', color='black')
    ax2.set_ylim(0,2)
    ax2.axhline(1, color='black', ls='--')
    ax2.set_ylabel('Data/Bkg')


    ax1.legend()
    ax1.set_ylabel('Events')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-1, 1e8)
    ax1.set_xlim(900,6000)
    ax1.set_xlabel('')


    savefigname = f'images/png/closureTest/{IOV}/closure_inclusive.png'
    plt.savefig(savefigname)
    plt.savefig(savefigname.replace('png', 'pdf'))

    print('saving '+savefigname)
    print('saving '+savefigname.replace('png', 'pdf'))





    # plot regions

    for cat in cats:

        fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])


        signal_cat = label_to_int['2t'+cat]
        antitag_cat = label_to_int['at'+cat]


        hbkg = functions.getHist2('ttbarmass', 'JetHT', IOV,
                 sum_axes=[],
                 integrate_axes={'systematic':'nominal', 'anacat':antitag_cat},
                 tag = '_bkgest'

                )

        httbar = functions.getHist2('ttbarmass', 'TTbar', IOV,
                 sum_axes=[],
                 integrate_axes={'systematic':'nominal', 'anacat':signal_cat},        
                )

        hsig = functions.getHist2('ttbarmass', 'JetHT', IOV,
                 sum_axes=[],
                 integrate_axes={'systematic':'nominal', 'anacat':signal_cat}        
                )


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


        text = 'Preliminary'+'\n'+ dytext +', '+ btext

        hep.cms.label('', data=True, lumi='{0:0.1f}'.format(lumi[IOV]/1000), year=IOV, loc=2, fontsize=20, ax=ax1)
        hep.cms.text(text, loc=2, fontsize=20, ax=ax1)

        hep.histplot(hsig, histtype='errorbar', color='black', label='Data', ax=ax1)
        hep.histplot(hbkg, histtype='fill', color='xkcd:pale gold', label='NTMJ Bkg Est', ax=ax1)
        hep.histplot(httbar, histtype='fill', color='xkcd:deep red', label='SM TTbar', ax=ax1)




        ratio_plot =  hsig / hbkg.values()
        hep.histplot(ratio_plot, ax=ax2, histtype='errorbar', color='black')
        ax2.set_ylim(0,2)
        ax2.axhline(1, color='black', ls='--')
        ax2.set_ylabel('Data/Bkg')


        ax1.legend()
        ax1.set_ylabel('Events')
        ax1.set_yscale('log')
        ax1.set_ylim(1e-1, 1e8)
        ax1.set_xlim(900,6000)
        ax1.set_xlabel('')


        savefigname = f'images/png/closureTest/{IOV}/closure_{cat}.png'
        plt.savefig(savefigname)
        plt.savefig(savefigname.replace('png', 'pdf'))

        print('saving '+savefigname)
        print('saving '+savefigname.replace('png', 'pdf'))


    
    

def plotClosureTestQCD():
    
    
    print('\nPlotting QCD Closure Test\n')

    
    fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])

    hbkg = functions.getHist2('ttbarmass', 'QCD', IOV,
             sum_axes=['anacat'],
             integrate_axes={'systematic':'nominal', 'anacat':antitag_cats},
             tag = '_bkgest'

            )

    hsig = functions.getHist2('ttbarmass', 'QCD', IOV,
             sum_axes=['anacat'],
             integrate_axes={'systematic':'nominal', 'anacat':signal_cats}        
            )


    text = 'Preliminary'+'\n'+r'$\Delta y$ inclusive'+', '+r'b-tag inclusive'

    hep.cms.label('', data=True, lumi='{0:0.1f}'.format(lumi[IOV]/1000), year=IOV, loc=2, fontsize=20, ax=ax1)
    hep.cms.text(text, loc=2, fontsize=20, ax=ax1)

    hep.histplot(hsig, histtype='errorbar', color='black', label='QCD SR', ax=ax1)
    hep.histplot(hbkg, histtype='fill', color='xkcd:pale gold', label='QCD Bkg Est', ax=ax1)




    ratio_plot =  hsig / hbkg.values()
    hep.histplot(ratio_plot, ax=ax2, histtype='errorbar', color='black')
    ax2.set_ylim(0,2)
    ax2.axhline(1, color='black', ls='--')
    ax2.set_ylabel('Data/Bkg')


    ax1.legend()
    ax1.set_ylabel('Events')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-1, 1e13)
    ax1.set_xlim(900,6000)
    ax1.set_xlabel('')


    savefigname = f'images/png/closureTest/{IOV}/closureQCD_inclusive.png'
    plt.savefig(savefigname)
    plt.savefig(savefigname.replace('png', 'pdf'))

    print('saving '+savefigname)
    print('saving '+savefigname.replace('png', 'pdf'))


    
    # plot regions
    
    for cat in cats:


        fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])


        signal_cat = label_to_int['2t'+cat]
        antitag_cat = label_to_int['at'+cat]


        hbkg = functions.getHist2('ttbarmass', 'QCD', IOV,
             sum_axes=[],
             integrate_axes={'systematic':'nominal', 'anacat':antitag_cat},
             tag = '_bkgest'

            )

        hsig = functions.getHist2('ttbarmass', 'QCD', IOV,
             sum_axes=[],
             integrate_axes={'systematic':'nominal', 'anacat':signal_cat}        
            )


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
        text = 'Preliminary'+'\n'+ dytext +', '+ btext

        hep.cms.label('', data=True, lumi='{0:0.1f}'.format(lumi[IOV]/1000), year=IOV, loc=2, fontsize=20, ax=ax1)
        hep.cms.text(text, loc=2, fontsize=20, ax=ax1)

        hep.histplot(hsig, histtype='errorbar', color='black', label='QCD SR', ax=ax1)
        hep.histplot(hbkg, histtype='fill', color='xkcd:pale gold', label='QCD Bkg Est', ax=ax1)


        ratio_plot =  hsig / hbkg.values()
        hep.histplot(ratio_plot, ax=ax2, histtype='errorbar', color='black')
        ax2.set_ylim(0,2)
        ax2.axhline(1, color='black', ls='--')
        ax2.set_ylabel('Data/Bkg')


        ax1.legend()
        ax1.set_ylabel('Events')
        ax1.set_yscale('log')
        ax1.set_ylim(1e-1, 1e13)
        ax1.set_xlim(900,6000)
        ax1.set_xlabel('')


        savefigname = f'images/png/closureTest/{IOV}/closureQCD_{cat}.png'
        plt.savefig(savefigname)
        plt.savefig(savefigname.replace('png', 'pdf'))

        print('saving '+savefigname)
        print('saving '+savefigname.replace('png', 'pdf'))





plotSystematics()
plotClosureTest()
plotClosureTestQCD()


   
