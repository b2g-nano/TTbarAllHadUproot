import numpy as np
from coffea import util
import glob
import os
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")


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


    

def getLabelMap():
    
    coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/outputs/'
    coffea_file = max(glob.glob(coffea_dir+'*.coffea'), key=os.path.getctime)
    label_map = util.load(coffea_file)['analysisCategories']
    
    return label_map
    
    

def getRapidity(p4):
    
    return 0.5 * np.log(( p4.energy + p4.pz ) / ( p4.energy - p4.pz ))


def getCoffeaFilenames():
    
    coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/outputs/'
#     coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/local/history/transferfxn/coffea/'
    
    
    
    coffeaFiles = {
        "JetHT":{
            "unweighted": {
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
            "weighted": {
                "2016APV": {
                    "B": coffea_dir+'JetHT_2016APVB_bkgest.coffea',
                    "C": coffea_dir+'JetHT_2016APVC_bkgest.coffea',
                    "D": coffea_dir+'JetHT_2016APVD_bkgest.coffea',
                    "E": coffea_dir+'JetHT_2016APVE_bkgest.coffea',
                },
                "2016": {
                    "F": coffea_dir+'JetHT_2016F_bkgest.coffea',
                    "G": coffea_dir+'JetHT_2016G_bkgest.coffea',
                    "H": coffea_dir+'JetHT_2016H_bkgest.coffea',

                },
                "2017": '',
                "2018": ''
            }
        },

        "TTbar": {
            "unweighted": {
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
            },
            "weighted": {
                "2016APV": {
                    "700to1000": coffea_dir+'TTbar_2016APV_700to1000_bkgest.coffea',
                    "1000toInf": coffea_dir+'TTbar_2016APV_1000toInf_bkgest.coffea',
                },
                "2016": {
                    "700to1000": coffea_dir+'TTbar_2016_700to1000_bkgest.coffea',
                    "1000toInf": coffea_dir+'TTbar_2016_1000toInf_bkgest.coffea',
                },
                "2017": {
                    "700to1000": '',
                    "1000toInf": '',
                },
                "2018": {
                    "700to1000": '',
                    "1000toInf": '',
                }
                
            },
        },
        
        "QCD": {
            "unweighted": {
                "2016APV": coffea_dir+'QCD_2016APV.coffea',
                "2016": coffea_dir+'QCD_2016.coffea',
                "2017": '',
                "2018": ''
            },
            "weighted": {
                "2016APV": coffea_dir+'QCD_2016APV_bkgest.coffea',
                "2016": coffea_dir+'QCD_2016_bkgest.coffea',
                "2017": '',
                "2018": ''
            }
        },
        
        "ZPrime": {
            "2016": {
                "1000": coffea_dir+'ZPrime1000_2016.coffea',
                "1500": coffea_dir+'ZPrime1500_2016.coffea',
                "2000": coffea_dir+'ZPrime2000_2016.coffea',
                "2500": coffea_dir+'ZPrime2500_2016.coffea',
                "3000": coffea_dir+'ZPrime3000_2016.coffea',
                "3500": coffea_dir+'ZPrime3500_2016.coffea',
                "4000": coffea_dir+'ZPrime4000_2016.coffea',
                "4500": coffea_dir+'ZPrime4500_2016.coffea',
                "5000": coffea_dir+'ZPrime5000_2016.coffea'
            }
        }
    }
    
    
        
    return coffeaFiles
    
    
    
def loadCoffeaFile(dataset='QCD', year='2016', tag='', bkgest=False):
    
    coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/outputs/'
    
    coffeaFiles = getCoffeaFilenames()
        
    if bkgest: 
        bkgest_str = 'weighted'
    else: 
        bkgest_str = 'unweighted'

    if len(tag) > 0:
        file = coffeaFiles[dataset][bkgest_str][year][tag]
    else:
        file = coffeaFiles[dataset][bkgest_str][year]
        
        
    return util.load(file)   
    
    
    
    
def getHist(hname, ds, bkgest, year, sum_axes=['anacat'], integrate_axes={}):
    
    ######################################################################################
    # hname = histogram name (example: 'ttbarmass')                                      #
    # ds = dataset name (example: 'JetHT')                                               #
    # bkgest = boolean, True if bkg estimate applied                                     #
    # year = '2016APV' or '2016' or '2017' or '2018'                                     #
    # sum_axes = names of axes to sum over for scikit-hep/hist histogram                 #
    # integrate_axes = range to integrate over axis (example: {'anacat': [0,1,2,3,4,5]}) #
    ######################################################################################    

    
    # load histograms and get scale factors
    coffeaFiles = getCoffeaFilenames()
    
    
   
    cfiles = []
    sf = []
    bkgest_str = 'weighted' if bkgest else 'unweighted'
    
    for key, file in coffeaFiles[ds][bkgest_str][year].items():
        loaded_file = util.load(file)
        cfiles.append(loaded_file)
        
        if 'TTbar' in ds and '700to1000' in key:
            sf.append(lumi[year] * ttbar_xs1 * toptag_sf**2 / loaded_file['cutflow']['sumw'])
        elif 'TTbar' in ds and '1000toInf' in key:
            sf.append(lumi[year] * ttbar_xs2 * toptag_sf**2 / loaded_file['cutflow']['sumw'])     
        elif 'QCD' in ds:
            sf.append(lumi[IOV] * qcd_xs / loaded_file['cutflow']['sumw'])  
        else:
            sf.append(1.)
    
    # sum or integrate axes for all hists from dataset eras or pt bins
    sum_axes_dict = {ax:sum for ax in sum_axes}
    
    hists = []
    for cfile in cfiles:
            hists.append(cfile[hname][integrate_axes][sum_axes_dict])
    
    
    # sum all hists from dataset eras or pt bins
    histo = hists[0]*sf[0]
    if len(hists) > 1:
        for i in range(len(hists) - 1): 
            histo = histo + hists[i+1]*sf[i+1]
            
            
    return histo
    
            
            
def plotBackgroundEstimate(hdata, hntmj, httbar, year, text=''):
    
    hbkg = hntmj + httbar
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])

    
    hep.cms.label('', data=True, lumi='{0:0.1f}'.format(lumi[year]/1000.), year=year, loc=2, fontsize=20, ax=ax1)
    hep.cms.text(text, loc=2, fontsize=20, ax=ax1)

    hep.histplot(hdata,  ax=ax1, histtype='errorbar', color='black', label='Data')
    hep.histplot(hbkg,   ax=ax1, histtype='fill', color='xkcd:pale gold', label='NTMJ')
    hep.histplot(httbar, ax=ax1, histtype='fill', color='xkcd:deep red', label='TTbar')


    ratio_plot =  hdata / hbkg.values()
    hep.histplot(ratio_plot, ax=ax2, histtype='errorbar', color='black')
    ax2.set_ylim(0,2)
    ax2.axhline(1, color='black', ls='--')
    ax2.set_ylabel('Data/Bkg')

    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_ylabel('Events')
    ax1.set_xlabel('')
    ax1.set_ylim(1e-1, 1e7)
    ax1.set_xlim(900, 6000)
    ax2.set_xlim(900, 6000)            
    

    
def makePlotDirectories():
    
        
    plots_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/plots'
       
    directories = [
       
        # QCD closure test
        plots_dir+'/images/png/closureTestQCD/2016all',
        plots_dir+'/images/png/closureTestQCD/2016APV',
        plots_dir+'/images/png/closureTestQCD/2016',
        plots_dir+'/images/png/closureTestQCD/2017',
        plots_dir+'/images/png/closureTestQCD/2018',
        plots_dir+'/images/pdf/closureTestQCD/2016all',
        plots_dir+'/images/pdf/closureTestQCD/2016APV',
        plots_dir+'/images/pdf/closureTestQCD/2016',
        plots_dir+'/images/pdf/closureTestQCD/2017',
        plots_dir+'/images/pdf/closureTestQCD/2018',

        # Closure test
        plots_dir+'/images/png/closureTest/2016all',
        plots_dir+'/images/png/closureTest/2016APV',
        plots_dir+'/images/png/closureTest/2016',
        plots_dir+'/images/png/closureTest/2017',
        plots_dir+'/images/png/closureTest/2018',
        plots_dir+'/images/pdf/closureTest/2016all',
        plots_dir+'/images/pdf/closureTest/2016APV',
        plots_dir+'/images/pdf/closureTest/2016',
        plots_dir+'/images/pdf/closureTest/2017',
        plots_dir+'/images/pdf/closureTest/2018',

        # systematics
        plots_dir+'/images/png/systematics/2016all',
        plots_dir+'/images/png/systematics/2016APV',
        plots_dir+'/images/png/systematics/2016',
        plots_dir+'/images/png/systematics/2017',
        plots_dir+'/images/png/systematics/2018',
        plots_dir+'/images/pdf/systematics/2016all',
        plots_dir+'/images/pdf/systematics/2016APV',
        plots_dir+'/images/pdf/systematics/2016',
        plots_dir+'/images/pdf/systematics/2017',
        plots_dir+'/images/pdf/systematics/2018',
        
        # ttbarmass
        plots_dir+'/images/png/ttbarmass/2016all',
        plots_dir+'/images/png/ttbarmass/2016APV',
        plots_dir+'/images/png/ttbarmass/2016',
        plots_dir+'/images/png/ttbarmass/2017',
        plots_dir+'/images/png/ttbarmass/2018',
        plots_dir+'/images/pdf/ttbarmass/2016all',
        plots_dir+'/images/pdf/ttbarmass/2016APV',
        plots_dir+'/images/pdf/ttbarmass/2016',
        plots_dir+'/images/pdf/ttbarmass/2017',
        plots_dir+'/images/pdf/ttbarmass/2018',
        
    ]


    for path in directories:
        if not os.path.exists(path):
            os.makedirs(path)




    

    