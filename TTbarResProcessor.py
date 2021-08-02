#!/usr/bin/env python
# coding: utf-8

import copy
import scipy.stats as ss
from coffea import hist, processor, nanoevents
from coffea import util
from coffea.btag_tools import BTagScaleFactor
import numpy as np
import itertools
import pandas as pd
from numpy.random import RandomState

import awkward as ak
#from coffea.nanoevents.methods import nanoaod
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import vector

#ak.behavior.update(nanoaod.behavior)
ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)

# --- Define 'Manual bins' to use for mistag plots for aesthetic purposes--- #
manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]
#manual_etabins = []

"""@TTbarResAnaHadronic Package to perform the data-driven mistag-rate-based ttbar hadronic analysis. 
"""
class TTbarResProcessor(processor.ProcessorABC):
    def __init__(self, prng=RandomState(1234567890), htCut=950., minMSD=105., maxMSD=210.,
                 tau32Cut=0.65, ak8PtMin=400., bdisc=0.8484,
                 writePredDist=True,isData=True,year=2019, UseLookUpTables=False, lu=None, 
                 ModMass=False, RandomDebugMode=False, CalcEff_MC=True, ApplySF=False, UseEfficiencies=False):
        
        self.prng = prng
        self.htCut = htCut
        self.minMSD = minMSD
        self.maxMSD = maxMSD
        self.tau32Cut = tau32Cut
        self.ak8PtMin = ak8PtMin
        self.bdisc = bdisc
        self.writePredDist = writePredDist
        self.writeHistFile = True
        self.isData = isData
        self.year=year
        self.UseLookUpTables = UseLookUpTables
        self.ModMass = ModMass
        self.RandomDebugMode = RandomDebugMode
        self.CalcEff_MC = CalcEff_MC # Only for first run of the processor
        self.ApplySF = ApplySF # Only apply scale factors when MC efficiencies are being imported in second run of processor
        self.UseEfficiencies = UseEfficiencies
        self.lu = lu # Look Up Tables
        
        # --- anti-tag+probe, anti-tag, pre-tag, 0, 1, >=1, 2 ttags, any t-tag --- #
        self.ttagcats = ["Probet", "at", "pret", "0t", "1t", "1t+2t", "2t", "0t+1t+2t"] 
        
        # --- 0, 1, or 2 b-tags --- #
        self.btagcats = ["0b", "1b", "2b"]
        
        # --- Central and forward --- #
        self.ycats = ['cen', 'fwd']
        
        # --- Combine categories like "0bcen", "0bfwd", etc: --- #
        self.anacats = [ t+b+y for t,b,y in itertools.product( self.ttagcats, self.btagcats, self.ycats) ]
        #print(self.anacats)
        
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        cats_axis = hist.Cat("anacat", "Analysis Category")
        
        jetmass_axis = hist.Bin("jetmass", r"Jet $m$ [GeV]", 50, 0, 500)
        jetpt_axis = hist.Bin("jetpt", r"Jet $p_{T}$ [GeV]", 50, 0, 5000)
        ttbarmass_axis = hist.Bin("ttbarmass", r"$m_{t\bar{t}}$ [GeV]", 50, 0, 5000)
        jeteta_axis = hist.Bin("jeteta", r"Jet $\eta$", 50, -3, 3)
        jetphi_axis = hist.Bin("jetphi", r"Jet $\phi$", 50, -np.pi, np.pi)
        jety_axis = hist.Bin("jety", r"Jet $y$", 50, -3, 3)
        jetdy_axis = hist.Bin("jetdy", r"Jet $\Delta y$", 50, 0, 5)
        manual_axis = hist.Bin("jetp", r"Jet Momentum [GeV]", manual_bins)
        tagger_axis = hist.Bin("tagger", r"deepTag", 50, 0, 1)
        tau32_axis = hist.Bin("tau32", r"$\tau_3/\tau_2$", 50, 0, 2)
        
        subjetmass_axis = hist.Bin("subjetmass", r"SubJet $m$ [GeV]", 50, 0, 500)
        subjetpt_axis = hist.Bin("subjetpt", r"SubJet $p_{T}$ [GeV]", 50, 0, 2000)
        subjetpt_laxis = hist.Bin("subjetpt", r"SubJet $p_{T}$ [GeV]", 10, 0, 2000) #Larger bins
        subjeteta_axis = hist.Bin("subjeteta", r"SubJet $\eta$", 50, -3, 3)
        subjeteta_laxis = hist.Bin("subjeteta", r"SubJet $\eta$", 10, -3, 3) #Larger bins
        subjetphi_axis = hist.Bin("subjetphi", r"SubJet $\phi$", 50, -np.pi, np.pi)

        self._accumulator = processor.dict_accumulator({
            'ttbarmass': hist.Hist("Counts", dataset_axis, cats_axis, ttbarmass_axis),
            
            'jetmass':         hist.Hist("Counts", dataset_axis, cats_axis, jetmass_axis),
            'SDmass':          hist.Hist("Counts", dataset_axis, cats_axis, jetmass_axis),
            'SDmass_precat':   hist.Hist("Counts", dataset_axis, jetpt_axis, jetmass_axis), # What was this for again?
            
            'jetpt':     hist.Hist("Counts", dataset_axis, cats_axis, jetpt_axis),
            'jeteta':    hist.Hist("Counts", dataset_axis, cats_axis, jeteta_axis),
            'jetphi':    hist.Hist("Counts", dataset_axis, cats_axis, jetphi_axis),
            
            'probept':   hist.Hist("Counts", dataset_axis, cats_axis, jetpt_axis),
            'probep':    hist.Hist("Counts", dataset_axis, cats_axis, manual_axis),
            
            'jety':      hist.Hist("Counts", dataset_axis, cats_axis, jety_axis),
            'jetdy':     hist.Hist("Counts", dataset_axis, cats_axis, jetdy_axis),
            
            'deepTag_TvsQCD':   hist.Hist("Counts", dataset_axis, cats_axis, jetpt_axis, tagger_axis),
            'deepTagMD_TvsQCD': hist.Hist("Counts", dataset_axis, cats_axis, jetpt_axis, tagger_axis),
            
            'tau32':          hist.Hist("Counts", dataset_axis, cats_axis, tau32_axis),
            'tau32_2D':       hist.Hist("Counts", dataset_axis, cats_axis, jetpt_axis, tau32_axis),
            'tau32_precat': hist.Hist("Counts", dataset_axis, jetpt_axis, tau32_axis),
            
            'subjetmass':   hist.Hist("Counts", dataset_axis, cats_axis, subjetmass_axis), # not yet used
            'subjetpt':     hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
            'subjeteta':    hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
            'subjetphi':    hist.Hist("Counts", dataset_axis, cats_axis, subjetphi_axis), # not yet used
            
            'numerator':   hist.Hist("Counts", dataset_axis, cats_axis, manual_axis),
            'denominator': hist.Hist("Counts", dataset_axis, cats_axis, manual_axis),
            
#********************************************************************************************************************#
            
#             # ---- SubJet b-tag Efficiency as function of pT ---- #
#             'b_eff_numerator_pt_s01': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis), 
#             'b_eff_numerator_pt_s02': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis), 
#             'b_eff_numerator_pt_s11': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis), 
#             'b_eff_numerator_pt_s12': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis), 
            
#             'b_eff_denominator_pt_s01': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis), 
#             'b_eff_denominator_pt_s02': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis), 
#             'b_eff_denominator_pt_s11': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis), 
#             'b_eff_denominator_pt_s12': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis), 
            
#             # ---- SubJet b-tag Efficiency as function of eta ---- #
#             'b_eff_numerator_eta_s01': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'b_eff_numerator_eta_s02': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'b_eff_numerator_eta_s11': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'b_eff_numerator_eta_s12': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
            
#             'b_eff_denominator_eta_s01': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'b_eff_denominator_eta_s02': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'b_eff_denominator_eta_s11': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'b_eff_denominator_eta_s12': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
            
#             # ---- SubJet c-tag Efficiency as function of pT ---- #
#             'c_eff_numerator_pt_s01': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
#             'c_eff_numerator_pt_s02': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
#             'c_eff_numerator_pt_s11': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
#             'c_eff_numerator_pt_s12': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
            
#             'c_eff_denominator_pt_s01': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
#             'c_eff_denominator_pt_s02': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
#             'c_eff_denominator_pt_s11': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
#             'c_eff_denominator_pt_s12': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
            
#             # ---- SubJet c-tag Efficiency as function of eta ---- #
#             'c_eff_numerator_eta_s01': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'c_eff_numerator_eta_s02': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'c_eff_numerator_eta_s11': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'c_eff_numerator_eta_s12': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
            
#             'c_eff_denominator_eta_s01': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'c_eff_denominator_eta_s02': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'c_eff_denominator_eta_s11': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'c_eff_denominator_eta_s12': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
            
#             # ---- SubJet light quark-tag Efficiency as function of pT ---- #
#             'udsg_eff_numerator_pt_s01': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
#             'udsg_eff_numerator_pt_s02': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
#             'udsg_eff_numerator_pt_s11': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
#             'udsg_eff_numerator_pt_s12': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
            
#             'udsg_eff_denominator_pt_s01': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
#             'udsg_eff_denominator_pt_s02': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
#             'udsg_eff_denominator_pt_s11': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
#             'udsg_eff_denominator_pt_s12': hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
            
#             # ---- SubJet light quark-tag Efficiency as function of eta ---- #
#             'udsg_eff_numerator_eta_s01': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'udsg_eff_numerator_eta_s02': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'udsg_eff_numerator_eta_s11': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'udsg_eff_numerator_eta_s12': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
            
#             'udsg_eff_denominator_eta_s01': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'udsg_eff_denominator_eta_s02': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'udsg_eff_denominator_eta_s11': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
#             'udsg_eff_denominator_eta_s12': hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
            
            # ---- 2D SubJet b-tag Efficiency ---- #
            'b_eff_numerator_s01': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'b_eff_numerator_s02': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'b_eff_numerator_s11': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'b_eff_numerator_s12': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            
            'b_eff_denominator_s01': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'b_eff_denominator_s02': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'b_eff_denominator_s11': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'b_eff_denominator_s12': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            
            'b_eff_numerator_s01_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'b_eff_numerator_s02_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'b_eff_numerator_s11_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'b_eff_numerator_s12_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            
            'b_eff_denominator_s01_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'b_eff_denominator_s02_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'b_eff_denominator_s11_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'b_eff_denominator_s12_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            
            # ---- 2D SubJet c-tag Efficiency ---- #
            'c_eff_numerator_s01': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'c_eff_numerator_s02': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'c_eff_numerator_s11': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'c_eff_numerator_s12': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            
            'c_eff_denominator_s01': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'c_eff_denominator_s02': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'c_eff_denominator_s11': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'c_eff_denominator_s12': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            
            'c_eff_numerator_s01_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'c_eff_numerator_s02_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'c_eff_numerator_s11_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'c_eff_numerator_s12_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            
            'c_eff_denominator_s01_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'c_eff_denominator_s02_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'c_eff_denominator_s11_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'c_eff_denominator_s12_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            
            # ---- 2D SubJet light quark-tag Efficiency ---- #
            'udsg_eff_numerator_s01': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'udsg_eff_numerator_s02': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'udsg_eff_numerator_s11': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'udsg_eff_numerator_s12': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            
            'udsg_eff_denominator_s01': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'udsg_eff_denominator_s02': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'udsg_eff_denominator_s11': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'udsg_eff_denominator_s12': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            
            'udsg_eff_numerator_s01_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'udsg_eff_numerator_s02_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'udsg_eff_numerator_s11_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'udsg_eff_numerator_s12_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            
            'udsg_eff_denominator_s01_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'udsg_eff_denominator_s02_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'udsg_eff_denominator_s11_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'udsg_eff_denominator_s12_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            
            'cutflow': processor.defaultdict_accumulator(int),
            
        })
        
    def BtagUpdater(subjet, b_eff, ScaleFactorFilename, FittingPoint, OperatingPoint):  
        """
        subjet (Flattened Awkward Array) ---> One of the Four preselected subjet awkward arrays (e.g. SubJet01)
        b_eff (2D Array)                 ---> The imported b-tagging efficiency of the selected subjet
        ScaleFactorFilename (string)     ---> CSV file containing info to evaluate scale factors with
        FittingPoint (string)            ---> "loose"  , "medium", "tight"
        OperatingPoint (string)          ---> "central", "up"    , "down"
        """
        ###############  Btag Update Method ##################
        #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods
        #https://github.com/rappoccio/usercode/blob/Dev_53x/EDSHyFT/plugins/BTagSFUtil_tprime.h
        
        coin = np.random.uniform(0,1,len(subjet)) # used for randomly deciding which jets' btag status to update or not
        subjet_btag_status = (subjet.btagCSVV2 > self.bdisc) # does this subjet pass the btagger requirement
        btag_sf = BTagScaleFactor(ScaleFactorFilename, FittingPoint)
        BSF = btag_sf.eval(OperatingPoint, subjet.hadronFlavour, abs(subjet.eta), subjet.pt, ignore_missing=True)
        f_less = 1. - BSF # fraction of subjets to be downgraded
        f_greater = f_less/(1. - 1./b_eff) # fraction of subjets to be upgraded
        
        """
*******************************************************************************************************************        
                        Does the Subjet Pass the Discriminator Cut?
                       ---------------------------------------------
                      True                                      False
                      ----                                      -----
        | SF = 1  |  SF < 1  |  SF > 1  |         |  SF = 1  |  SF < 1  |  SF > 1  |

        |    O    |Downgrade?|    O     |         |     X    |    X     | Upgrade? |
                   ----------                                             --------
        |         |True|False|          |         |          |          |True|False|
                    ---  ---                                              ---  ---
        |         |  X |  O  |          |         |          |          |  O |  X  |

        ---------------------------------------------------------------------------------

        KEY:
             O ---> btagged subjet     (boolean 'value' = True)
             X ---> non btagged subjet (boolean 'value' = False)

        Track all conditions where elements of 'btag_update' will be true (4 conditions marked with 'O')
*******************************************************************************************************************        
        """ 
        
        subjet_new_btag_status = np.where( 
            (subjet_btag_status == True & (BSF == 1. ^ (BSF < 1.0 & coin < BSF ) ^ BSF > 1.))
            ^
            (subjet_btag_status == False & (BSF > 1. & coin < f_greater)), 
            True, False )

        return subjet_new_btag_status
            
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        
        # ---- Define dataset ---- #
        dataset = events.metadata['dataset']
        
        # ---- Get triggers from Dataset_info ---- #
        #triggers = [itrig for itrig in Dataset_info if 'HLT_PFHT' in itrig]
        #AK8triggers = [itrig for itrig in Dataset_info if 'HLT_AK8PFHT' in itrig]

        # ---- Find numeric values in trigger strings ---- #
        #triggers_cut1 = [sub.split('PFHT')[1] for sub in triggers] # Remove string characters from left of number
        #triggers_cut2 = [sub.split('_')[0] for sub in triggers_cut1] # Remove string characters from right of number
        #isTriggerValue = [val.isnumeric() for val in triggers_cut2] # Boolean -> if string is only a number
        #triggers_cut2 = np.where(isTriggerValue, triggers_cut2, 0) # If string is not a number, replace with 0
        #triggers_vals = [int(val) for val in triggers_cut2] # Convert string numbers to integers
        
        #AK8triggers_cut1 = [sub.split('HT')[1] for sub in AK8triggers]
        #AK8triggers_cut2 = [sub.split('_')[0] for sub in AK8triggers_cut1]
        #isAK8TriggerValue = [val.isnumeric() for val in AK8triggers_cut2]
        #AK8triggers_cut2 = np.where(isAK8TriggerValue, AK8triggers_cut2, 0)
        #AK8triggers_vals = [int(val) for val in AK8triggers_cut2]
        
        # ---- Find Largest and Second Largest Value ---- #
        #triggers_vals.sort(reverse = True)
        #AK8triggers_vals.sort(reverse = True)
        
        #triggers_vals1 = str(triggers_vals[0])
        #triggers_vals2 = str(triggers_vals[1])
        #AK8triggers_vals1 = str(AK8triggers_vals[0])
        #AK8triggers_vals2 = str(AK8triggers_vals[1])
        
        # ---- Define strings for the selected triggers ---- #
        #HLT_trig1_str = [itrig for itrig in triggers if (triggers_vals1) in itrig][0]
        #HLT_trig2_str = [itrig for itrig in triggers if (triggers_vals2) in itrig][0]
        #HLT_AK8_trig1_str = [itrig for itrig in AK8triggers if (AK8triggers_vals1) in itrig][0]
        #HLT_AK8_trig2_str = [itrig for itrig in AK8triggers if (AK8triggers_vals2) in itrig][0]
        
        # ---- Define HLT triggers to be used ---- #
        #HLT_trig1 = df[HLT_trig1_str]
        #HLT_trig2 = df[HLT_trig2_str]
        #HLT_AK8_trig1 = df[HLT_AK8_trig1_str]
        #HLT_AK8_trig2 = df[HLT_AK8_trig2_str]
       
        
        # ---- Define AK8 Jets as FatJets ---- #
        #FatJets = events.FatJet # Everything should already be defined in here.  example) df['FatJet_pt] -> events.FatJet.pt
        FatJets = ak.zip({
            "nFatJet": events.nFatJet,
            "pt": events.FatJet_pt,
            "eta": events.FatJet_eta,
            "phi": events.FatJet_phi,
            "mass": events.FatJet_mass,
            "area": events.FatJet_area,
            "msoftdrop": events.FatJet_msoftdrop,
            "jetId": events.FatJet_jetId,
            "tau1": events.FatJet_tau1,
            "tau2": events.FatJet_tau2,
            "tau3": events.FatJet_tau3,
            "tau4": events.FatJet_tau4,
            "n3b1": events.FatJet_n3b1,
            "btagDeepB": events.FatJet_btagDeepB,
            "btagCSVV2": events.FatJet_btagCSVV2, # Use as a prior probability of containing a bjet?
            "deepTag_TvsQCD": events.FatJet_deepTag_TvsQCD,
            "deepTagMD_TvsQCD": events.FatJet_deepTagMD_TvsQCD,
            "subJetIdx1": events.FatJet_subJetIdx1,
            "subJetIdx2": events.FatJet_subJetIdx2,
            "p4": ak.zip({
                "pt": events.FatJet_pt,
                "eta": events.FatJet_eta,
                "phi": events.FatJet_phi,
                "mass": events.FatJet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })

        # ---- Define AK4 jets as Jets ---- #
        #Jets = events.Jet
        Jets = ak.zip({
            "pt": events.Jet_pt,
            "eta": events.Jet_eta,
            "phi": events.Jet_phi,
            "mass": events.Jet_mass,
            "area": events.Jet_area,
            "hadronFlavour": events.Jet_hadronFlavour,
            "p4": ak.zip({
                "pt": events.Jet_pt,
                "eta": events.Jet_eta,
                "phi": events.Jet_phi,
                "mass": events.Jet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })

        # ---- Define SubJets ---- #
        #SubJets = events.SubJet
        SubJets = ak.zip({
            "pt": events.SubJet_pt,
            "eta": events.SubJet_eta,
            "phi": events.SubJet_phi,
            "mass": events.SubJet_mass,
            "btagDeepB": events.SubJet_btagDeepB,
            "btagCSVV2": events.SubJet_btagCSVV2,
            "hadronFlavour": events.SubJet_hadronFlavour,
            "p4": ak.zip({
                "pt": events.SubJet_pt,
                "eta": events.SubJet_eta,
                "phi": events.SubJet_phi,
                "mass": events.SubJet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })
        
        # ---- Get event weights from dataset ---- #
        if 'JetHT' in dataset: # If data is used...
            evtweights = np.ones(ak.to_awkward0(FatJets).size) # set all "data weights" to one
        else: # if Monte Carlo dataset is used...
            evtweights = events.Generator_weight
        # ---- Show all events ---- #
        output['cutflow']['all events'] += ak.to_awkward0(FatJets).size

        # ---- Apply Trigger(s) ---- #
        #FatJets = FatJets[HLT_AK8_trig1]
        #evtweights = evtweights[HLT_AK8_trig1]
        #Jets = Jets[HLT_AK8_trig1]
        #SubJets = SubJets[HLT_AK8_trig1]
        
        # ---- Jets that satisfy Jet ID ---- #
        jet_id = (FatJets.jetId > 0) # Loose jet ID
        FatJets = FatJets[jet_id]
        output['cutflow']['jet id'] += ak.to_awkward0(jet_id).any().sum()
        
        # ---- Apply pT Cut and Rapidity Window ---- #
        FatJets_rapidity = .5*np.log( (FatJets.p4.energy + FatJets.p4.pz)/(FatJets.p4.energy - FatJets.p4.pz) )
        jetkincut_index = (FatJets.pt > self.ak8PtMin) & (np.abs(FatJets_rapidity) < 2.4)
        FatJets = FatJets[ jetkincut_index ]
        output['cutflow']['jet kin'] += ak.to_awkward0(jetkincut_index).any().sum()
        
        # ---- Find two AK8 Jets ---- #
        twoFatJetsKin = (ak.num(FatJets, axis=-1) == 2)
        FatJets = FatJets[twoFatJetsKin]
        evtweights = evtweights[twoFatJetsKin]
        Jets = Jets[twoFatJetsKin]
        SubJets = SubJets[twoFatJetsKin]
        output['cutflow']['two FatJets and jet kin'] += ak.to_awkward0(twoFatJetsKin).sum()
        
        # ---- Apply HT Cut ---- #
        hT = ak.to_awkward0(Jets.pt).sum()
        passhT = (hT > self.htCut)
        evtweights = evtweights[passhT]
        FatJets = FatJets[passhT]
        SubJets = SubJets[passhT]
        
        # ---- Randomly Assign AK8 Jets as TTbar Candidates 0 and 1 --- #
        Counts = np.ones(len(FatJets), dtype='i') # Number 1 for each FatJet
        
        if self.RandomDebugMode == True: # 'Sudo' randomizer to test for consistent results
            highPhi = FatJets.phi[:,0] > FatJets.phi[:,1]
            highRandIndex = np.where(highPhi, 0, 1) # 1D array of 0's and 1's
            index = ak.unflatten( highRandIndex, Counts ) # Subtly confusing logic of what this does is shown below
            """
                For example:
                
                    FatJets = [[FatJet0, FatJet1], [AnotherFatJet0, AnotherFatJet1], ..., [LastFatJet0, LastFatJet1]]
                    Counts = [1, 1, 1, 1, 1, 1]
                    highRandIndex = [1, 1, 0, 1, 0, 0]
                    
                unflattening highRandIndex with Counts will group "Counts" number of highRandIndex elements together 
                in a new higher order array:
                
                    index = [[1], [1], [0], [1], [0], [0]]
                    1 - index = [[0], [0], [1], [0], [1], [1]]
                    
               where the index is used to slice (select) either the first FatJet (FatJets[:,0]) or the second (FatJets[:,1]):
        
                    FatJets[index] = [FatJet1, AnotherFatJet1, ..., LastFatJet0]
                    FatJets[1-index] = [FatJet0, AnotherFatJet0, ..., LastFatJet1]
            """
        else: # Truly randomize
            index = ak.unflatten( self.prng.randint(2, size=len(FatJets)), Counts )
        
        jet0 = FatJets[index] #J0
        jet1 = FatJets[1 - index] #J1
        
        ttbarcands = ak.cartesian([jet0, jet1]) # Re-group the randomized pairs in a similar fashion to how they were
        
        """ NOTE that ak.cartesian gives a shape with one more layer than FatJets """
        # ---- Make sure we have at least 1 TTbar candidate pair and re-broadcast releveant arrays  ---- #
        oneTTbar = (ak.num(ttbarcands, axis=-1) >= 1)
        output['cutflow']['>= one oneTTbar'] += ak.to_awkward0(oneTTbar).sum()
        ttbarcands = ttbarcands[oneTTbar]
        evtweights = evtweights[oneTTbar]
        FatJets = FatJets[oneTTbar]
        SubJets = SubJets[oneTTbar]
         
        # ---- Apply Delta Phi Cut for Back to Back Topology ---- #
        """ NOTE: Should find function for this; avoids 2pi problem """
        dPhiCut = ttbarcands.slot0.p4.delta_phi(ttbarcands.slot1.p4) > 2.1
        dPhiCut = ak.flatten(dPhiCut)
        output['cutflow']['dPhi > 2.1'] += ak.to_awkward0(dPhiCut).sum()
        ttbarcands = ttbarcands[dPhiCut]
        evtweights = evtweights[dPhiCut]
        FatJets = FatJets[dPhiCut] 
        SubJets = SubJets[dPhiCut] 
        
        # ---- Identify subjets according to subjet ID ---- #
        hasSubjets0 = ((ttbarcands.slot0.subJetIdx1 > -1) & (ttbarcands.slot0.subJetIdx2 > -1)) # 1st candidate has two subjets
        hasSubjets1 = ((ttbarcands.slot1.subJetIdx1 > -1) & (ttbarcands.slot1.subJetIdx2 > -1)) # 2nd candidate has two subjets
        GoodSubjets = ak.flatten(((hasSubjets0) & (hasSubjets1))) # Selection of 4 (leading) subjects
   
        ttbarcands = ttbarcands[GoodSubjets] # Choose only ttbar candidates with this selection of subjets
        SubJets = SubJets[GoodSubjets]
        evtweights = evtweights[GoodSubjets]
       
        SubJet01 = SubJets[ttbarcands.slot0.subJetIdx1] # ttbarcandidate 0's first subjet 
        SubJet02 = SubJets[ttbarcands.slot0.subJetIdx2] # ttbarcandidate 0's second subjet
        SubJet11 = SubJets[ttbarcands.slot1.subJetIdx1] # ttbarcandidate 1's first subjet 
        SubJet12 = SubJets[ttbarcands.slot1.subJetIdx2] # ttbarcandidate 1's second subjet
        
        # ---- Define Rapidity Regions ---- #
        """ NOTE that ttbarcands.i0.p4.energy no longer works after ttbarcands is defined as an old awkward array """
        s0_energy = ttbarcands.slot0.p4.energy
        s1_energy = ttbarcands.slot1.p4.energy
        s0_pz = ttbarcands.slot0.p4.pz
        s1_pz = ttbarcands.slot1.p4.pz
        ttbarcands_s0_rapidity = 0.5*np.log( (s0_energy+s0_pz)/(s0_energy-s0_pz) ) # rapidity as function of eta
        ttbarcands_s1_rapidity = 0.5*np.log( (s1_energy+s1_pz)/(s1_energy-s1_pz) ) # rapidity as function of eta
        cen = np.abs(ttbarcands_s0_rapidity - ttbarcands_s1_rapidity) < 1.0
        fwd = (~cen)
        


#    TTTTTTT     TTTTTTT    A    GGGGGGG GGGGGGG EEEEEEE RRRRRR  
#       T           T      A A   G       G       E       R     R 
#       T           T     A   A  G       G       E       R     R 
#       T           T     AAAAA  G  GGGG G  GGGG EEEEEEE RRRRRR  
#       T           T    A     A G     G G     G E       R   R   
#       T           T    A     A G     G G     G E       R    R   
#       T           T    A     A  GGGGG   GGGGG  EEEEEEE R     R

        # ---- CMS Top Tagger Version 2 (SD and Tau32 Cuts) ---- #
        tau32_s0 = np.where(ttbarcands.slot0.tau2>0,ttbarcands.slot0.tau3/ttbarcands.slot0.tau2, 0 )
        tau32_s1 = np.where(ttbarcands.slot1.tau2>0,ttbarcands.slot1.tau3/ttbarcands.slot1.tau2, 0 )
        taucut_s0 = tau32_s0 < self.tau32Cut
        taucut_s1 = tau32_s1 < self.tau32Cut
        mcut_s0 = (self.minMSD < ttbarcands.slot0.msoftdrop) & (ttbarcands.slot0.msoftdrop < self.maxMSD) 
        mcut_s1 = (self.minMSD < ttbarcands.slot1.msoftdrop) & (ttbarcands.slot1.msoftdrop < self.maxMSD) 

        ttag_s0 = (taucut_s0) & (mcut_s0)
        ttag_s1 = (taucut_s1) & (mcut_s1)
        
        # ---- Define "Top Tag" Regions ---- #
        antitag = (~taucut_s0) & (mcut_s0) #Probe will always be ttbarcands.i1 (at)
        antitag_probe = np.logical_and(antitag, ttag_s1) #Found an antitag and ttagged probe pair for mistag rate (Probet)
        pretag =  ttag_s0 # Only jet0 (pret)
        ttag0 =   (~ttag_s0) & (~ttag_s1) # No tops tagged (0t)
        ttag1 =   ttag_s0 ^ ttag_s1 # Exclusively one top tagged (1t)
        ttagI =   ttag_s0 | ttag_s1 # At least one top tagged ('I' for 'inclusive' tagger; >=1t; 1t+2t)
        ttag2 =   ttag_s0 & ttag_s1 # Both jets top tagged (2t)
        Alltags = ttag0 | ttagI #Either no tag or at least one tag (0t+1t+2t)
        
        
#    BBBBBB      TTTTTTT    A    GGGGGGG GGGGGGG EEEEEEE RRRRRR  
#    B     B        T      A A   G       G       E       R     R 
#    B     B        T     A   A  G       G       E       R     R 
#    BBBBBB         T     AAAAA  G  GGGG G  GGGG EEEEEEE RRRRRR  
#    B     B        T    A     A G     G G     G E       R   R   
#    B     B        T    A     A G     G G     G E       R    R   
#    BBBBBB         T    A     A  GGGGG   GGGGG  EEEEEEE R     R
        
        # ---- Pick FatJet that passes btag discriminator cut based on its subjet with the highest btag value ---- #
        btag_s0 = ( np.maximum(SubJet01.btagCSVV2 , SubJet02.btagCSVV2) > self.bdisc )
        btag_s1 = ( np.maximum(SubJet11.btagCSVV2 , SubJet12.btagCSVV2) > self.bdisc )
        
        # --- Define "B Tag" Regions ---- #
        btag0 = (~btag_s0) & (~btag_s1) #(0b)
        btag1 = btag_s0 ^ btag_s1 #(1b)
        btag2 = btag_s0 & btag_s1 #(2b)
        
        
        if self.CalcEff_MC == True: # Get 'flavor' tagging efficiency from MC
            if 'JetHT' not in dataset:
                # --- Define pT and Eta for Both Candidates' Subjets (for simplicity) --- #
                pT_s01 = ak.flatten(SubJet01.pt) # pT of 1st subjet in ttbarcand 0
                eta_s01 = ak.flatten(SubJet01.eta) # eta of 1st subjet in ttbarcand 0
                flav_s01 = np.abs(ak.flatten(SubJet01.hadronFlavour)) # either 'normal' or 'anti'
                
                pT_s02 = ak.flatten(SubJet02.pt) # pT of 2nd subjet in ttbarcand 0
                eta_s02 = ak.flatten(SubJet02.eta) # eta of 2nd subjet in ttbarcand 0
                flav_s02 = np.abs(ak.flatten(SubJet02.hadronFlavour))
                
                pT_s11 = ak.flatten(SubJet11.pt) # pT of 1st subjet in ttbarcand 1
                eta_s11 = ak.flatten(SubJet11.eta) # eta of 1st subjet in ttbarcand 1
                flav_s11 = np.abs(ak.flatten(SubJet11.hadronFlavour))
                
                pT_s12 = ak.flatten(SubJet12.pt) # pT of 2nd subjet in ttbarcand 1
                eta_s12 = ak.flatten(SubJet12.eta) # eta of 2nd subjet in ttbarcand 1
                flav_s12 = np.abs(ak.flatten(SubJet12.hadronFlavour))
        
                # --- For Efficiency Calculations, check efficiency of all four subjets passing the discriminant ---- #
                s01_btagged = (SubJet01.btagCSVV2 > self.bdisc)
                s02_btagged = (SubJet02.btagCSVV2 > self.bdisc)
                s11_btagged = (SubJet11.btagCSVV2 > self.bdisc)
                s12_btagged = (SubJet12.btagCSVV2 > self.bdisc)

                # --- Calculate MC Flavor Effeciencies (Defining Numerator and Denominator) --- #
                # --- Numerators and Denominators to be placed in both 1D and 2D output files --- #
                # --- Denominators are independant of subjet flavor --- #
                
                # ---- b-tagging eff. numerators ---- #
                Eff_b_Num_pT_s01 = np.where(s01_btagged & (flav_s01 == 5), pT_s01, -1)
                Eff_b_Num_eta_s01 = np.where(s01_btagged & (flav_s01 == 5), eta_s01, -1)
                
                Eff_b_Num_pT_s02 = np.where(s02_btagged & (flav_s02 == 5), pT_s02, -1)
                Eff_b_Num_eta_s02 = np.where(s02_btagged & (flav_s02 == 5), eta_s02, -1)
                
                Eff_b_Num_pT_s11 = np.where(s11_btagged & (flav_s11 == 5), pT_s11, -1)
                Eff_b_Num_eta_s11 = np.where(s11_btagged & (flav_s11 == 5), eta_s11, -1)
                
                Eff_b_Num_pT_s12 = np.where(s12_btagged & (flav_s12 == 5), pT_s12, -1)
                Eff_b_Num_eta_s12 = np.where(s12_btagged & (flav_s12 == 5), eta_s12, -1)
                
                # ---- c-tagging eff. numerators ---- #
                Eff_c_Num_pT_s01 = np.where(s01_btagged & (flav_s01 == 4), pT_s01, -1)
                Eff_c_Num_eta_s01 = np.where(s01_btagged & (flav_s01 == 4), eta_s01, -1)
                
                Eff_c_Num_pT_s02 = np.where(s02_btagged & (flav_s02 == 4), pT_s02, -1)
                Eff_c_Num_eta_s02 = np.where(s02_btagged & (flav_s02 == 4), eta_s02, -1)
                
                Eff_c_Num_pT_s11 = np.where(s11_btagged & (flav_s11 == 4), pT_s11, -1)
                Eff_c_Num_eta_s11 = np.where(s11_btagged & (flav_s11 == 4), eta_s11, -1)
                
                Eff_c_Num_pT_s12 = np.where(s12_btagged & (flav_s12 == 4), pT_s12, -1)
                Eff_c_Num_eta_s12 = np.where(s12_btagged & (flav_s12 == 4), eta_s12, -1)
                
                # ---- light parton-tagging eff. numerators ---- #
                if_s01_isLightParton = (flav_s01 != 5) & (flav_s01 != 4)
                if_s02_isLightParton = (flav_s01 != 5) & (flav_s01 != 4)
                if_s11_isLightParton = (flav_s01 != 5) & (flav_s01 != 4)
                if_s12_isLightParton = (flav_s01 != 5) & (flav_s01 != 4)
                
                Eff_udsg_Num_pT_s01 = np.where(s01_btagged & (if_s01_isLightParton), pT_s01, -1)
                Eff_udsg_Num_eta_s01 = np.where(s01_btagged & (if_s01_isLightParton), eta_s01, -1)
                
                Eff_udsg_Num_pT_s02 = np.where(s02_btagged & (if_s02_isLightParton), pT_s02, -1)
                Eff_udsg_Num_eta_s02 = np.where(s02_btagged & (if_s02_isLightParton), eta_s02, -1)
                
                Eff_udsg_Num_pT_s11 = np.where(s11_btagged & (if_s11_isLightParton), pT_s11, -1)
                Eff_udsg_Num_eta_s11 = np.where(s11_btagged & (if_s11_isLightParton), eta_s11, -1)
                
                Eff_udsg_Num_pT_s12 = np.where(s12_btagged & (if_s12_isLightParton), pT_s12, -1)
                Eff_udsg_Num_eta_s12 = np.where(s12_btagged & (if_s12_isLightParton), eta_s12, -1)
                
                # ---- b-tagging eff. denominators ---- #
                Eff_b_Denom_pT_s01 = np.where(flav_s01 == 5, pT_s01, -1)
                Eff_b_Denom_eta_s01 = np.where(flav_s01 == 5, eta_s01, -1)
                
                Eff_b_Denom_pT_s02 = np.where(flav_s01 == 5, pT_s02, -1)
                Eff_b_Denom_eta_s02 = np.where(flav_s01 == 5, eta_s02, -1)
                
                Eff_b_Denom_pT_s11 = np.where(flav_s01 == 5, pT_s11, -1)
                Eff_b_Denom_eta_s11 = np.where(flav_s01 == 5, eta_s11, -1)
                
                Eff_b_Denom_pT_s12 = np.where(flav_s01 == 5, pT_s12, -1)
                Eff_b_Denom_eta_s12 = np.where(flav_s01 == 5, eta_s12, -1)
                
                # ---- c-tagging eff. denominators ---- #
                Eff_c_Denom_pT_s01 = np.where(flav_s01 == 4, pT_s01, -1)
                Eff_c_Denom_eta_s01 = np.where(flav_s01 == 4, eta_s01, -1)
                
                Eff_c_Denom_pT_s02 = np.where(flav_s01 == 4, pT_s02, -1)
                Eff_c_Denom_eta_s02 = np.where(flav_s01 == 4, eta_s02, -1)
                
                Eff_c_Denom_pT_s11 = np.where(flav_s01 == 4, pT_s11, -1)
                Eff_c_Denom_eta_s11 = np.where(flav_s01 == 4, eta_s11, -1)
                
                Eff_c_Denom_pT_s12 = np.where(flav_s01 == 4, pT_s12, -1)
                Eff_c_Denom_eta_s12 = np.where(flav_s01 == 4, eta_s12, -1)
                
                # ---- light parton-tagging eff. denominators ---- #
                Eff_udsg_Denom_pT_s01 = np.where(if_s01_isLightParton, pT_s01, -1)
                Eff_udsg_Denom_eta_s01 = np.where(if_s01_isLightParton, eta_s01, -1)
                
                Eff_udsg_Denom_pT_s02 = np.where(if_s02_isLightParton, pT_s02, -1)
                Eff_udsg_Denom_eta_s02 = np.where(if_s02_isLightParton, eta_s02, -1)
                
                Eff_udsg_Denom_pT_s11 = np.where(if_s11_isLightParton, pT_s11, -1)
                Eff_udsg_Denom_eta_s11 = np.where(if_s11_isLightParton, eta_s11, -1)
                
                Eff_udsg_Denom_pT_s12 = np.where(if_s12_isLightParton, pT_s12, -1)
                Eff_udsg_Denom_eta_s12 = np.where(if_s12_isLightParton, eta_s12, -1)
                
                # --- Flatten all numerators --- #
                Eff_b_Num_pT_s01 = ak.flatten(Eff_b_Num_pT_s01)
                Eff_b_Num_eta_s01 = ak.flatten(Eff_b_Num_eta_s01)
                
                Eff_b_Num_pT_s02 = ak.flatten(Eff_b_Num_pT_s02)
                Eff_b_Num_eta_s02 = ak.flatten(Eff_b_Num_eta_s02)
                
                Eff_b_Num_pT_s11 = ak.flatten(Eff_b_Num_pT_s11)
                Eff_b_Num_eta_s11 = ak.flatten(Eff_b_Num_eta_s11)
                
                Eff_b_Num_pT_s12 = ak.flatten(Eff_b_Num_pT_s12)
                Eff_b_Num_eta_s12 = ak.flatten(Eff_b_Num_eta_s12)
                
                Eff_c_Num_pT_s01 = ak.flatten(Eff_c_Num_pT_s01)
                Eff_c_Num_eta_s01 = ak.flatten(Eff_c_Num_eta_s01)
                
                Eff_c_Num_pT_s02 = ak.flatten(Eff_c_Num_pT_s02)
                Eff_c_Num_eta_s02 = ak.flatten(Eff_c_Num_eta_s02)
                
                Eff_c_Num_pT_s11 = ak.flatten(Eff_c_Num_pT_s11)
                Eff_c_Num_eta_s11 = ak.flatten(Eff_c_Num_eta_s11)
                
                Eff_c_Num_pT_s12 = ak.flatten(Eff_c_Num_pT_s12)
                Eff_c_Num_eta_s12 = ak.flatten(Eff_c_Num_eta_s12)
                
                Eff_udsg_Num_pT_s01 = ak.flatten(Eff_udsg_Num_pT_s01)
                Eff_udsg_Num_eta_s01 = ak.flatten(Eff_udsg_Num_eta_s01)
                
                Eff_udsg_Num_pT_s02 = ak.flatten(Eff_udsg_Num_pT_s02)
                Eff_udsg_Num_eta_s02 = ak.flatten(Eff_udsg_Num_eta_s02)
                
                Eff_udsg_Num_pT_s11 = ak.flatten(Eff_udsg_Num_pT_s11)
                Eff_udsg_Num_eta_s11 = ak.flatten(Eff_udsg_Num_eta_s11)
                
                Eff_udsg_Num_pT_s12 = ak.flatten(Eff_udsg_Num_pT_s12)
                Eff_udsg_Num_eta_s12 = ak.flatten(Eff_udsg_Num_eta_s12)
                
                # **************************************************************************************** #
                # ----------------------------- 2-D B-tagging Efficiencies ------------------------------- #
                # **************************************************************************************** #
                output['b_eff_numerator_s01'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Num_pT_s01),
                                                  subjeteta = ak.to_numpy(Eff_b_Num_eta_s01),
                                                   weight = ak.to_numpy(evtweights))
                output['b_eff_numerator_s02'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Num_pT_s02),
                                                  subjeteta = ak.to_numpy(Eff_b_Num_eta_s02),
                                                   weight = ak.to_numpy(evtweights))
                output['b_eff_numerator_s11'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Num_pT_s11),
                                                  subjeteta = ak.to_numpy(Eff_b_Num_eta_s11),
                                                  weight = ak.to_numpy(evtweights))
                output['b_eff_numerator_s12'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Num_pT_s12),
                                                  subjeteta = ak.to_numpy(Eff_b_Num_eta_s12),
                                                  weight = ak.to_numpy(evtweights))

                output['b_eff_denominator_s01'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Denom_pT_s01),
                                                  subjeteta = ak.to_numpy(Eff_b_Denom_eta_s01),
                                                  weight = ak.to_numpy(evtweights))
                output['b_eff_denominator_s02'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Denom_pT_s02),
                                                  subjeteta = ak.to_numpy(Eff_b_Denom_eta_s02),
                                                  weight = ak.to_numpy(evtweights))
                output['b_eff_denominator_s11'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Denom_pT_s11),
                                                  subjeteta = ak.to_numpy(Eff_b_Denom_eta_s11),
                                                  weight = ak.to_numpy(evtweights))
                output['b_eff_denominator_s12'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Denom_pT_s12),
                                                  subjeteta = ak.to_numpy(Eff_b_Denom_eta_s12),
                                                  weight = ak.to_numpy(evtweights))

                output['b_eff_numerator_s01_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Num_pT_s01),
                                                  subjeteta = ak.to_numpy(Eff_b_Num_eta_s01),
                                                  weight = ak.to_numpy(evtweights))
                output['b_eff_numerator_s02_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Num_pT_s02),
                                                  subjeteta = ak.to_numpy(Eff_b_Num_eta_s02),
                                                  weight = ak.to_numpy(evtweights))
                output['b_eff_numerator_s11_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Num_pT_s11),
                                                  subjeteta = ak.to_numpy(Eff_b_Num_eta_s11),
                                                  weight = ak.to_numpy(evtweights))
                output['b_eff_numerator_s12_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Num_pT_s12),
                                                  subjeteta = ak.to_numpy(Eff_b_Num_eta_s12),
                                                  weight = ak.to_numpy(evtweights))

                output['b_eff_denominator_s01_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Denom_pT_s01),
                                                  subjeteta = ak.to_numpy(Eff_b_Denom_eta_s01),
                                                  weight = ak.to_numpy(evtweights))
                output['b_eff_denominator_s02_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Denom_pT_s02),
                                                  subjeteta = ak.to_numpy(Eff_b_Denom_eta_s02),
                                                  weight = ak.to_numpy(evtweights))
                output['b_eff_denominator_s11_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Denom_pT_s11),
                                                  subjeteta = ak.to_numpy(Eff_b_Denom_eta_s11),
                                                  weight = ak.to_numpy(evtweights))
                output['b_eff_denominator_s12_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_b_Denom_pT_s12),
                                                  subjeteta = ak.to_numpy(Eff_b_Denom_eta_s12),
                                                  weight = ak.to_numpy(evtweights))

                # **************************************************************************************** #
                # ----------------------------- 2-D C-tagging Efficiencies ------------------------------- #
                # **************************************************************************************** #
                output['c_eff_numerator_s01'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Num_pT_s01),
                                                  subjeteta = ak.to_numpy(Eff_c_Num_eta_s01),
                                                  weight = ak.to_numpy(evtweights))
                output['c_eff_numerator_s02'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Num_pT_s02),
                                                  subjeteta = ak.to_numpy(Eff_c_Num_eta_s02),
                                                  weight = ak.to_numpy(evtweights))
                output['c_eff_numerator_s11'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Num_pT_s11),
                                                  subjeteta = ak.to_numpy(Eff_c_Num_eta_s11),
                                                  weight = ak.to_numpy(evtweights))
                output['c_eff_numerator_s12'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Num_pT_s12),
                                                  subjeteta = ak.to_numpy(Eff_c_Num_eta_s12),
                                                  weight = ak.to_numpy(evtweights))

                output['c_eff_denominator_s01'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Denom_pT_s01),
                                                  subjeteta = ak.to_numpy(Eff_c_Denom_eta_s01),
                                                  weight = ak.to_numpy(evtweights))
                output['c_eff_denominator_s02'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Denom_pT_s02),
                                                  subjeteta = ak.to_numpy(Eff_c_Denom_eta_s02),
                                                  weight = ak.to_numpy(evtweights))
                output['c_eff_denominator_s11'].fill(dataset = dataset,
                                                     subjetpt = ak.to_numpy(Eff_c_Denom_pT_s11),
                                                  subjeteta = ak.to_numpy(Eff_c_Denom_eta_s11),
                                                  weight = ak.to_numpy(evtweights))
                output['c_eff_denominator_s12'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Denom_pT_s12),
                                                  subjeteta = ak.to_numpy(Eff_c_Denom_eta_s12),
                                                  weight = ak.to_numpy(evtweights))

                output['c_eff_numerator_s01_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Num_pT_s01),
                                                  subjeteta = ak.to_numpy(Eff_c_Num_eta_s01),
                                                  weight = ak.to_numpy(evtweights))
                output['c_eff_numerator_s02_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Num_pT_s02),
                                                  subjeteta = ak.to_numpy(Eff_c_Num_eta_s02),
                                                  weight = ak.to_numpy(evtweights))
                output['c_eff_numerator_s11_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Num_pT_s11),
                                                  subjeteta = ak.to_numpy(Eff_c_Num_eta_s11),
                                                  weight = ak.to_numpy(evtweights))
                output['c_eff_numerator_s12_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Num_pT_s12),
                                                  subjeteta = ak.to_numpy(Eff_c_Num_eta_s12),
                                                  weight = ak.to_numpy(evtweights))

                output['c_eff_denominator_s01_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Denom_pT_s01),
                                                  subjeteta = ak.to_numpy(Eff_c_Denom_eta_s01),
                                                  weight = ak.to_numpy(evtweights))
                output['c_eff_denominator_s02_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Denom_pT_s02),
                                                  subjeteta = ak.to_numpy(Eff_c_Denom_eta_s02),
                                                  weight = ak.to_numpy(evtweights))
                output['c_eff_denominator_s11_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Denom_pT_s11),
                                                  subjeteta = ak.to_numpy(Eff_c_Denom_eta_s11),
                                                  weight = ak.to_numpy(evtweights))
                output['c_eff_denominator_s12_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_c_Denom_pT_s12),
                                                  subjeteta = ak.to_numpy(Eff_c_Denom_eta_s12),
                                                  weight = ak.to_numpy(evtweights))
                
                # **************************************************************************************** #
                # ------------------------ 2-D Light Parton-tagging Efficiencies ------------------------- #
                # **************************************************************************************** #
                output['udsg_eff_numerator_s01'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Num_pT_s01),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Num_eta_s01),
                                                  weight = ak.to_numpy(evtweights))
                output['udsg_eff_numerator_s02'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Num_pT_s02),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Num_eta_s02),
                                                  weight = ak.to_numpy(evtweights))
                output['udsg_eff_numerator_s11'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Num_pT_s11),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Num_eta_s11),
                                                  weight = ak.to_numpy(evtweights))
                output['udsg_eff_numerator_s12'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Num_pT_s12),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Num_eta_s12),
                                                  weight = ak.to_numpy(evtweights))

                output['udsg_eff_denominator_s01'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Denom_pT_s01),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Denom_eta_s01),
                                                  weight = ak.to_numpy(evtweights))
                output['udsg_eff_denominator_s02'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Denom_pT_s02),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Denom_eta_s02),
                                                  weight = ak.to_numpy(evtweights))
                output['udsg_eff_denominator_s11'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Denom_pT_s11),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Denom_eta_s11),
                                                  weight = ak.to_numpy(evtweights))
                output['udsg_eff_denominator_s12'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Denom_pT_s12),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Denom_eta_s12),
                                                  weight = ak.to_numpy(evtweights))

                output['udsg_eff_numerator_s01_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Num_pT_s01),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Num_eta_s01),
                                                  weight = ak.to_numpy(evtweights))
                output['udsg_eff_numerator_s02_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Num_pT_s02),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Num_eta_s02),
                                                  weight = ak.to_numpy(evtweights))
                output['udsg_eff_numerator_s11_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Num_pT_s11),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Num_eta_s11),
                                                  weight = ak.to_numpy(evtweights))
                output['udsg_eff_numerator_s12_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Num_pT_s12),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Num_eta_s12),
                                                  weight = ak.to_numpy(evtweights))

                output['udsg_eff_denominator_s01_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Denom_pT_s01),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Denom_eta_s01),
                                                  weight = ak.to_numpy(evtweights))
                output['udsg_eff_denominator_s02_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Denom_pT_s02),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Denom_eta_s02),
                                                  weight = ak.to_numpy(evtweights))
                output['udsg_eff_denominator_s11_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Denom_pT_s11),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Denom_eta_s11),
                                                  weight = ak.to_numpy(evtweights))
                output['udsg_eff_denominator_s12_largerbins'].fill(dataset = dataset,
                                                  subjetpt = ak.to_numpy(Eff_udsg_Denom_pT_s12),
                                                  subjeteta = ak.to_numpy(Eff_udsg_Denom_eta_s12),
                                                  weight = ak.to_numpy(evtweights))
                
        
        if self.ApplySF == True: # Apply b Tag Scale Factors and redefine btag_s0 and btag_s1
            if 'JetHT' not in dataset:
                if self.UseEfficiencies == False: # Define weights solely from BSF to weight each btag category region

                    # ---- Temporarily define the 'outline' of the collection of weights to calculate ---- # 
                    btag_wgts = {'0b':np.array([None]),
                                 '1b':np.array([None, None]),
                                 '2b':np.array([None, None, None])}
                    
                    """
                    btag_wgts['mb'][n] --> w(n|m) --> "Probability" of n number of b-tags given m number of b-tagged jets
                    -----------------------------------------------------------------------------------------
                    w(0|0) = 1
                    w(0|1), w(1|1) = 1 - BSF, 
                                   = BSF
                    w(0|2), w(1|2), w(2|2) = (1 - BSF_s0)(1 - BSF_s1), 
                                           = (1 - BSF_s0)BSF_s1 + BSF_s0(1 - BSF_s1),
                                           = (BSF_s0)(BSF_s1)
                    w(1|0), w(2|0), w(2|1) = Undef.
                    """
                    
                    # ---- Use the leading subjet again to get the scale factors ---- #
                    LeadingSubjet_s0 = np.where(SubJet01.btagCSVV2>SubJet02.btagCSVV2, SubJet01, SubJet02)
                    LeadingSubjet_s1 = np.where(SubJet11.btagCSVV2>SubJet12.btagCSVV2, SubJet11, SubJet12)
                    
                    # ---- Define the BSF for each of the two fatjets ---- #
                    btag_sf = BTagScaleFactor("DeepCSV_106XUL17SF_V2.csv", "tight")
                    
                    BSF_s0 = btag_sf.eval("central", 
                                          LeadingSubjet_s0.hadronFlavour, abs(LeadingSubjet_s0.eta), LeadingSubjet_s0.pt,
                                          ignore_missing=True)
                    BSF_s1 = btag_sf.eval("central", 
                                          LeadingSubjet_s1.hadronFlavour, abs(LeadingSubjet_s1.eta), LeadingSubjet_s1.pt,
                                          ignore_missing=True)
                    
                    # ---- w(0|0) ---- #
                    btag_wgts['0b'][0] = np.where(btag0, np.ones_like(BSF_s0), 0.)
                    
                    # ---- w(0|1) and w(1|1) ---- # 
                    btag_wgts['1b'][0] = np.where(btag1, np.where(btag_s0, 1.-BSF_s0, 1.-BSF_s1), 0.)
                    btag_wgts['1b'][1] = np.where(btag1, np.where(btag_s0, BSF_s0, BSF_s1), 0.)
                    
                    # ---- w(0|2), w(1|2), w(2|2) ---- # 
                    btag_wgts['2b'][0] = np.where(btag2, (1 - BSF_s0)*(1 - BSF_s1), 0.) 
                    btag_wgts['2b'][1] = np.where(btag2, (1 - BSF_s0)*BSF_s1 + BSF_s0*(1 - BSF_s1), 0.) 
                    btag_wgts['2b'][2] = np.where(btag2, BSF_s0*BSF_s1, 0.) 
                    
                    # ---- 'Matrix Multiplied' weights to apply to each b-tag region ---- #
                    Wgts_to_0btag_region = ak.flatten(btag_wgts['0b'][0] + btag_wgts['1b'][0] + btag_wgts['2b'][0])
                    Wgts_to_1btag_region = ak.flatten(btag_wgts['1b'][1] + btag_wgts['2b'][1])
                    Wgts_to_2btag_region = ak.flatten(btag_wgts['2b'][2])
                    
                    # ---- 'Matrix Multiplied' non-zero weights to apply to each b-tag region ---- #
                    Wgts_to_0btag_region_nonzero = np.where(Wgts_to_0btag_region==0., 1., Wgts_to_0btag_region)
                    Wgts_to_1btag_region_nonzero = np.where(Wgts_to_1btag_region==0., 1., Wgts_to_1btag_region)
                    Wgts_to_2btag_region_nonzero = np.where(Wgts_to_2btag_region==0., 1., Wgts_to_2btag_region)
                    
                    
                else: # Upgrade or Downgrade btag status based on btag efficiency of all four subjets per event
                                # ---- Import MC 'flavor' efficiencies ---- #
                    """
                    e.g.)
                        b_eff_s01 = imported b tagging efficiency for 1st subjet in ttbar candidate slot 0
                        c_eff_s12 = imported c tagging efficiency for 2nd subjet in ttbar candidate slot 1
                    """

                    # -- Scale Factor File -- #
                    SF_filename = "DeepCSV_106XUL17SF_V2.csv"    
                    Fitting = "medium"

                    # -- Does Subjet pass the discriminator cut and is it updated -- #
                    SubJet01_isBtagged_central = BtagUpdater(SubJet01, b_eff, SF_filename, Fitting, "central")
                    SubJet02_isBtagged_central = BtagUpdater(SubJet02, b_eff, SF_filename, Fitting, "central")
                    SubJet11_isBtagged_central = BtagUpdater(SubJet11, b_eff, SF_filename, Fitting, "central")
                    SubJet12_isBtagged_central = BtagUpdater(SubJet12, b_eff, SF_filename, Fitting, "central")

                    # If either subjet 1 or 2 in FatJet 0 and 1 is btagged after update, then that FatJet is considered btagged #
                    btag_s0 = (SubJet01_isBtagged) ^ (SubJet02_isBtagged)  
                    btag_s1 = (SubJet11_isBtagged) ^ (SubJet12_isBtagged)
     
                    # --- Re-Define b-Tag Regions with "Updated" Tags ---- #
                    btag0 = (~btag_s0) & (~btag_s1) #(0b)
                    btag1 = btag_s0 ^ btag_s1 #(1b)
                    btag2 = btag_s0 & btag_s1 #(2b)

        # ---- Get Analysis Categories ---- # 
        # ---- They are (central, forward) cross (0b,1b,2b) cross (Probet,at,0t,1t,>=1t,2t) ---- #
        regs = [cen,fwd]
        btags = [btag0,btag1,btag2]
        ttags = [antitag_probe,antitag,pretag,ttag0,ttag1,ttagI,ttag2,Alltags]
        cats = [ ak.to_awkward0(ak.flatten(t&b&y)) for t,b,y in itertools.product( ttags, btags, regs) ]
        labels_and_categories = dict(zip( self.anacats, cats ))
        #print(labels_and_categories)
        
        # ---- Variables for Kinematic Histograms ---- #
        # ---- "slot0" is the control jet, "slot1" is the probe jet ---- #
        jetpt = ak.flatten(ttbarcands.slot1.pt)
        jeteta = ak.flatten(ttbarcands.slot1.eta)
        jetphi = ak.flatten(ttbarcands.slot1.phi)
        jetmass = ak.flatten(ttbarcands.slot1.mass)
        SDmass = ak.flatten(ttbarcands.slot1.msoftdrop)
        Tau32 = ak.flatten((ttbarcands.slot1.tau3/ttbarcands.slot1.tau2))

        """ Add 4-vectors and get its total mass """
        ttbarp4sum = ttbarcands.slot0.p4.add(ttbarcands.slot1.p4)
        ttbarmass = ak.flatten(ttbarp4sum.mass)
        
        """ Use previously defined definitions for rapidity (until/unless better method is found) """
        jety = ak.flatten(ttbarcands_s0_rapidity)
        jetdy = np.abs(ak.flatten(ttbarcands_s0_rapidity) - ak.flatten(ttbarcands_s1_rapidity))

        # ---- Variables for Deep Tagger Analysis ---- #
        deepTag = ak.flatten(ttbarcands.slot1.deepTag_TvsQCD)
        deepTagMD = ak.flatten(ttbarcands.slot1.deepTagMD_TvsQCD)
        
        weights = evtweights

        # ---- Define the SumW2 for MC Datasets ---- #
        output['cutflow']['sumw'] += np.sum(weights)
        output['cutflow']['sumw2'] += np.sum(weights**2)
        
        # ---- Define Momentum p of probe jet as the Mistag Rate variable; M(p) ---- #
        # ---- Transverse Momentum pT can also be used instead; M(pT) ---- #
        pT = ak.flatten(ttbarcands.slot1.pt)
        pz = ak.flatten(ttbarcands.slot1.p4.pz)
        p = np.absolute(np.sqrt(pT**2 + pz**2))
        
        # ---- Define the Numerator and Denominator for Mistag Rate ---- #
        numerator = np.where(antitag_probe, p, -1) # If no antitag and tagged probe, move event to useless bin
        denominator = np.where(antitag, p, -1) # If no antitag, move event to useless bin
        
        numerator = ak.flatten(numerator)
        denominator = ak.flatten(denominator)
        
        df = pd.DataFrame({"momentum":p}) # Used for finding values in LookUp Tables
        
        for ilabel,icat in labels_and_categories.items():
            ### ------------------------------------ Mistag Scaling ------------------------------------ ###
            if self.UseLookUpTables == True:
                # ---- Weight ttbar M.C. and data by mistag from data (corresponding to its year) ---- #
                if 'TTbar_' in dataset:
                    file_df = self.lu['JetHT' + dataset[-4:] + '_Data']['at' + str(ilabel[-5:])] #Pick out proper JetHT year mistag for TTbar sim.
                elif dataset == 'TTbar':
                    file_df = self.lu['JetHT']['at' + str(ilabel[-5:])] # All JetHT years mistag for TTbar sim.
                else:
                    file_df = self.lu[dataset]['at' + str(ilabel[-5:])] # get mistag (lookup) filename for 'at'
                
                bin_widths = file_df['p'].values # collect bins as written in .csv file
                mtr = file_df['M(p)'].values # collect mistag rate as function of p as written in file
                wgts = mtr # Define weights based on mistag rates
                
                BinKeys = np.arange(bin_widths.size) # Use as label for BinNumber column in the new dataframe
                
                Bins = np.array(manual_bins)
                
                df['BinWidth'] = pd.cut(np.asarray(p), bins=Bins) # new dataframe column
                df['BinNumber'] = pd.cut(np.asarray(p), bins=Bins, labels=BinKeys)
                
                BinNumber = df['BinNumber'].values # Collect the Bin Numbers into a numpy array
                BinNumber = BinNumber.astype('int64') # Insures the bin numbers are integers
            
                WeightMatching = wgts[BinNumber] # Match 'wgts' with corresponding p bin using the bin number
                Weights = weights*WeightMatching # Include 'wgts' with the previously defined 'weights'
            else:
                Weights = weights # No mistag rates, no change to weights
                
            ###---------------------------------------------------------------------------------------------###
            ### ----------------------------------- Mod-mass Procedure ------------------------------------ ###
            if self.ModMass == True:
                QCD_unweighted = util.load('CoffeaOutputs/UnweightedOutputs/TTbarResCoffea_QCD_unweighted_output_futures_3-10-21_trial.coffea') 
    
                # ---- Extract event counts from QCD MC hist in signal region ---- #
                QCD_hist = QCD_unweighted['jetmass'].integrate('anacat', '2t' + str(ilabel[-5:])).integrate('dataset', 'QCD')
                data = QCD_hist.values() # Dictionary of values
                QCD_data = [i for i in data.values()][0] # place every element of the dictionary into a numpy array

                # ---- Re-create Bins from QCD_hist as Numpy Array ---- #
                bins = np.arange(510) #Re-make bins from the jetmass_axis starting with the appropriate range
                QCD_bins = bins[::10] #Finish re-making bins by insuring exactly 50 bins like the jetmass_axis

                # ---- Define Mod Mass Distribution ---- #
                ModMass_hist_dist = ss.rv_histogram([QCD_data,QCD_bins])
                jet1_modp4 = copy.copy(jet1.p4) #J1's Lorentz four vector that can be safely modified
                jet1_modp4["fMass"] = ModMass_hist_dist.rvs(size=ak.to_awkward0(jet1_modp4).size) #Replace J1's mass with random value of mass from mm hist
                #ttbarcands_modmass = jet0.p4.cross(jet1_modp4) #J0's four vector x modified J1's four vector
                ttbarcands_modmass = ak.cartesian([jet0.p4, jet1_modp4])

                # ---- Apply Necessary Selections to new modmass version ---- #
                ttbarcands_modmass = ttbarcands_modmass[oneTTbar]
                ttbarcands_modmass = ttbarcands_modmass[dPhiCut]
                ttbarcands_modmass = ttbarcands_modmass[GoodSubjets]
                
                # ---- Manually sum the modmass p4 candidates (Coffea technicality) ---- #
                #ttbarcands_modmass_p4_sum = (ttbarcands_modmass.i0 + ttbarcands_modmass.i1)
                ttbarcands_modmassp4sum = ttbarcands.slot0.p4.add(ttbarcands.slot1.p4)
                
                # ---- Re-define Mass Variables for ModMass Procedure (pt, eta, phi are redundant to change) ---- #
                #ttbarmass = ttbarcands_modmass_p4_sum.flatten().mass
                #jetmass = ttbarcands_modmass.i1.mass.flatten()
                ttbarmass = ak.flatten(ttbarcands_modmassp4sum.mass)
                jetmass = ak.flatten(ttbarcands_modmass.slot1.mass)
                
            ###---------------------------------------------------------------------------------------------###
            ### ------------------------------ B-Tag Weighting (S.F. Only) -------------------------------- ###
            if (self.ApplySF == True) and (self.UseEfficiencies == False):
 
                if '0b' in ilabel:
                    Weights = Weights*Wgts_to_0btag_region_nonzero
                elif '1b' in ilabel:
                    Weights = Weights*Wgts_to_1btag_region_nonzero 
                else:
                    Weights = Weights*Wgts_to_2btag_region_nonzero 
  
            ###---------------------------------------------------------------------------------------------###
            output['cutflow'][ilabel] += np.sum(icat)
          
            output['ttbarmass'].fill(dataset = dataset, anacat = ilabel, 
                                ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['jetpt'].fill(dataset = dataset, anacat = ilabel, 
                                jetpt = ak.to_numpy(jetpt[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['probept'].fill(dataset = dataset, anacat = ilabel, 
                                jetpt = ak.to_numpy(pT[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['probep'].fill(dataset = dataset, anacat = ilabel, 
                                jetp = ak.to_numpy(p[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['jeteta'].fill(dataset = dataset, anacat = ilabel, 
                                jeteta = ak.to_numpy(jeteta[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['jetphi'].fill(dataset = dataset, anacat = ilabel, 
                                jetphi = ak.to_numpy(jetphi[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['jety'].fill(dataset = dataset, anacat = ilabel, 
                                jety = ak.to_numpy(jety[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['jetdy'].fill(dataset = dataset, anacat = ilabel, 
                                jetdy = ak.to_numpy(jetdy[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['numerator'].fill(dataset = dataset, anacat = ilabel, 
                                jetp = ak.to_numpy(numerator[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['denominator'].fill(dataset = dataset, anacat = ilabel, 
                                jetp = ak.to_numpy(denominator[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['jetmass'].fill(dataset = dataset, anacat = ilabel, 
                                   jetmass = ak.to_numpy(jetmass[icat]),
                                   weight = ak.to_numpy(Weights[icat]))
            output['SDmass'].fill(dataset = dataset, anacat = ilabel, 
                                   jetmass = ak.to_numpy(SDmass[icat]),
                                   weight = ak.to_numpy(Weights[icat]))
            output['tau32'].fill(dataset = dataset, anacat = ilabel,
                                          tau32 = ak.to_numpy(Tau32[icat]),
                                          weight = ak.to_numpy(Weights[icat]))
            output['tau32_2D'].fill(dataset = dataset, anacat = ilabel,
                                          jetpt = ak.to_numpy(pT[icat]),
                                          tau32 = ak.to_numpy(Tau32[icat]),
                                          weight = ak.to_numpy(Weights[icat]))
            output['deepTag_TvsQCD'].fill(dataset = dataset, anacat = ilabel,
                                          jetpt = ak.to_numpy(pT[icat]),
                                          tagger = ak.to_numpy(deepTag[icat]),
                                          weight = ak.to_numpy(Weights[icat]))
            output['deepTagMD_TvsQCD'].fill(dataset = dataset, anacat = ilabel,
                                            jetpt = ak.to_numpy(pT[icat]),
                                            tagger = ak.to_numpy(deepTagMD[icat]),
                                            weight = ak.to_numpy(Weights[icat]))
            
# ************************************************************************************************************ #            
            
            
#             # ---- 1-D B-tagging Efficiencies ---- #
#             output['b_eff_numerator_pt_s01'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_b_Num_pT_s01[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['b_eff_numerator_pt_s02'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_b_Num_pT_s02[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['b_eff_numerator_pt_s11'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_b_Num_pT_s11[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['b_eff_numerator_pt_s12'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_b_Num_pT_s12[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
            
#             output['b_eff_denominator_pt_s01'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_b_Denom_pT_s01[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['b_eff_denominator_pt_s02'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_b_Denom_pT_s02[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['b_eff_denominator_pt_s11'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_b_Denom_pT_s11[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['b_eff_denominator_pt_s12'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_b_Denom_pT_s12[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
            
#             output['b_eff_numerator_eta_s01'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_b_Num_eta_s01[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['b_eff_numerator_eta_s02'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_b_Num_eta_s02[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['b_eff_numerator_eta_s11'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_b_Num_eta_s11[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['b_eff_numerator_eta_s12'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_b_Num_eta_s12[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
            
#             output['b_eff_denominator_eta_s01'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_b_Denom_eta_s01[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['b_eff_denominator_eta_s02'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_b_Denom_eta_s02[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['b_eff_denominator_eta_s11'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_b_Denom_eta_s11[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['b_eff_denominator_eta_s12'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_b_Denom_eta_s12[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
            
#             # ---- 1-D C-tagging Efficiencies ---- #
#             output['c_eff_numerator_pt_s01'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_c_Num_pT_s01[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['c_eff_numerator_pt_s02'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_c_Num_pT_s02[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['c_eff_numerator_pt_s11'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_c_Num_pT_s11[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['c_eff_numerator_pt_s12'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_c_Num_pT_s12[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
            
#             output['c_eff_denominator_pt_s01'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_c_Denom_pT_s01[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['c_eff_denominator_pt_s02'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_c_Denom_pT_s02[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['c_eff_denominator_pt_s11'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_c_Denom_pT_s11[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['c_eff_denominator_pt_s12'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_c_Denom_pT_s12[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
            
#             output['c_eff_numerator_eta_s01'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_c_Num_eta_s01[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['c_eff_numerator_eta_s02'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_c_Num_eta_s02[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['c_eff_numerator_eta_s11'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_c_Num_eta_s11[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['c_eff_numerator_eta_s12'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_c_Num_eta_s12[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
            
#             output['c_eff_denominator_eta_s01'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_c_Denom_eta_s01[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['c_eff_denominator_eta_s02'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_c_Denom_eta_s02[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['c_eff_denominator_eta_s11'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_c_Denom_eta_s11[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['c_eff_denominator_eta_s12'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_c_Denom_eta_s12[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
            
#             # ---- 1-D Light Parton-tagging Efficiencies ---- #
#             output['udsg_eff_numerator_pt_s01'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_udsg_Num_pT_s01[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['udsg_eff_numerator_pt_s02'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_udsg_Num_pT_s02[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['udsg_eff_numerator_pt_s11'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_udsg_Num_pT_s11[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['udsg_eff_numerator_pt_s12'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_udsg_Num_pT_s12[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
            
#             output['udsg_eff_denominator_pt_s01'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_udsg_Denom_pT_s01[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['udsg_eff_denominator_pt_s02'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_udsg_Denom_pT_s02[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['udsg_eff_denominator_pt_s11'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_udsg_Denom_pT_s11[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['udsg_eff_denominator_pt_s12'].fill(dataset = dataset, anacat = ilabel,
#                                               subjetpt = ak.to_numpy(Eff_udsg_Denom_pT_s12[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
            
#             output['udsg_eff_numerator_eta_s01'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_udsg_Num_eta_s01[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['udsg_eff_numerator_eta_s02'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_udsg_Num_eta_s02[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['udsg_eff_numerator_eta_s11'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_udsg_Num_eta_s11[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['udsg_eff_numerator_eta_s12'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_udsg_Num_eta_s12[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
            
#             output['udsg_eff_denominator_eta_s01'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_udsg_Denom_eta_s01[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['udsg_eff_denominator_eta_s02'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_udsg_Denom_eta_s02[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['udsg_eff_denominator_eta_s11'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_udsg_Denom_eta_s11[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
#             output['udsg_eff_denominator_eta_s12'].fill(dataset = dataset, anacat = ilabel,
#                                               subjeteta = ak.to_numpy(Eff_udsg_Denom_eta_s12[icat]),
#                                               weight = ak.to_numpy(Weights[icat]))
            
            
        
        return output

    def postprocess(self, accumulator):
        return accumulator

