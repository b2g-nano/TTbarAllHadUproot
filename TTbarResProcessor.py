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

        distance_axis = hist.Bin("delta_r", r"$\Delta r$", 50, 0, 5)

        self._accumulator = processor.dict_accumulator({
#    ===================================================================================================================
#    K     K IIIIIII N     N EEEEEEE M     M    A    TTTTTTT IIIIIII   CCCC      H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
#    K   K      I    NN    N E       MM   MM   A A      T       I     C          H     H    I     S         T     S          
#    K K        I    N N   N E       M M M M  A   A     T       I    C           H     H    I    S          T    S           
#    KKk        I    N  N  N EEEEEEE M  M  M  AAAAA     T       I    C           HHHHHHH    I     SSSSS     T     SSSSS      
#    K  K       I    N   N N E       M     M A     A    T       I    C           H     H    I          S    T          S     
#    K   K      I    N    NN E       M     M A     A    T       I     C          H     H    I         S     T         S      
#    K   K   IIIIIII N     N EEEEEEE M     M A     A    T    IIIIIII   CCCC      H     H IIIIIII SSSSS      T    SSSSS 
#    ===================================================================================================================
            
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
            
#    ====================================================================
#    EEEEEEE FFFFFFF FFFFFFF      H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
#    E       F       F            H     H    I     S         T     S          
#    E       F       F            H     H    I    S          T    S           
#    EEEEEEE FFFFFFF FFFFFFF      HHHHHHH    I     SSSSS     T     SSSSS      
#    E       F       F            H     H    I          S    T          S     
#    E       F       F            H     H    I         S     T         S      
#    EEEEEEE F       F       *    H     H IIIIIII SSSSS      T    SSSSS
#    ====================================================================
            
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
            
            #********************************************************************************************************************#
            
#    =============================================================================      
#   " TTTTTTT EEEEEEE   SSSSS TTTTTTT "    H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
#        T    E        S         T         H     H    I     S         T     S          
#        T    E       S          T         H     H    I    S          T    S           
#        T    EEEEEEE  SSSSS     T         HHHHHHH    I     SSSSS     T     SSSSS      
#        T    E             S    T         H     H    I          S    T          S     
#        T    E            S     T         H     H    I         S     T         S      
#        T    EEEEEEE SSSSS      T         H     H IIIIIII SSSSS      T    SSSSS 
#    =============================================================================  
            
            # ---- 4 Subjet's pt independant of category (testing purposes only) ---- #
            'subjet01_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet02_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet11_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet12_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            
            # ---- Distance between subjet and Gen level quark ---- #
            'subjet01_bquark_distance': hist.Hist("Counts", dataset_axis, distance_axis),
            'subjet02_bquark_distance': hist.Hist("Counts", dataset_axis, distance_axis),
            'subjet11_bquark_distance': hist.Hist("Counts", dataset_axis, distance_axis),
            'subjet12_bquark_distance': hist.Hist("Counts", dataset_axis, distance_axis),
            
            'subjet01_cquark_distance': hist.Hist("Counts", dataset_axis, distance_axis),
            'subjet02_cquark_distance': hist.Hist("Counts", dataset_axis, distance_axis),
            'subjet11_cquark_distance': hist.Hist("Counts", dataset_axis, distance_axis),
            'subjet12_cquark_distance': hist.Hist("Counts", dataset_axis, distance_axis),
            
            'subjet01_lightquark_distance': hist.Hist("Counts", dataset_axis, distance_axis),
            'subjet02_lightquark_distance': hist.Hist("Counts", dataset_axis, distance_axis),
            'subjet11_lightquark_distance': hist.Hist("Counts", dataset_axis, distance_axis),
            'subjet12_lightquark_distance': hist.Hist("Counts", dataset_axis, distance_axis),
            
            # pt of subjet defined with 'hadronFlavour' and subjet defined with nearest flavoured genpart #
            'subjet01_bflavor_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet01_with_bquark_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet02_bflavor_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet02_with_bquark_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet11_bflavor_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet11_with_bquark_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet12_bflavor_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet12_with_bquark_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            
            'subjet01_cflavor_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet01_with_cquark_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet02_cflavor_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet02_with_cquark_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet11_cflavor_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet11_with_cquark_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet12_cflavor_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet12_with_cquark_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            
            'subjet01_lightflavor_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet01_with_lightquark_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet02_lightflavor_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet02_with_lightquark_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet11_lightflavor_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet11_with_lightquark_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet12_lightflavor_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            'subjet12_with_lightquark_pt': hist.Hist("Counts", dataset_axis, subjetpt_axis),
            
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
        
#    ================================================================
#    TTTTTTT RRRRRR  IIIIIII GGGGGGG GGGGGGG EEEEEEE RRRRRR    SSSSS     
#       T    R     R    I    G       G       E       R     R  S          
#       T    R     R    I    G       G       E       R     R S           
#       T    RRRRRR     I    G  GGGG G  GGGG EEEEEEE RRRRRR   SSSSS      
#       T    R   R      I    G     G G     G E       R   R         S     
#       T    R    R     I    G     G G     G E       R    R       S      
#       T    R     R IIIIIII  GGGGG   GGGGG  EEEEEEE R     R SSSSS  
#    ================================================================
        
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

#    ===========================================================================================    
#    N     N    A    N     N   OOO      A      OOO   DDDD          OOO   BBBBBB  JJJJJJJ   SSSSS     
#    NN    N   A A   NN    N  O   O    A A    O   O  D   D        O   O  B     B    J     S          
#    N N   N  A   A  N N   N O     O  A   A  O     O D    D      O     O B     B    J    S           
#    N  N  N  AAAAA  N  N  N O     O  AAAAA  O     O D     D     O     O BBBBBB     J     SSSSS      
#    N   N N A     A N   N N O     O A     A O     O D    D      O     O B     B J  J          S     
#    N    NN A     A N    NN  O   O  A     A  O   O  D   D        O   O  B     B J  J         S      
#    N     N A     A N     N   OOO   A     A   OOO   DDDD          OOO   BBBBBB   JJ     SSSSS 
#    =========================================================================================== 
        
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

        # ---- Define Generator Particles ---- #
        GenParts = ak.zip({
            "pdgId": events.GenPart_pdgId,
            "pt": events.GenPart_pt,
            "eta": events.GenPart_eta,
            "phi": events.GenPart_phi,
            "mass": events.GenPart_mass,
            "p4": ak.zip({
                "pt": events.GenPart_pt,
                "eta": events.GenPart_eta,
                "phi": events.GenPart_phi,
                "mass": events.GenPart_mass,
                }, with_name="Vector3D"),
            })

#    ===================================================================================
#    PPPPPP  RRRRRR  EEEEEEE L       IIIIIII M     M       CCCC  U     U TTTTTTT   SSSSS     
#    P     P R     R E       L          I    MM   MM      C      U     U    T     S          
#    P     P R     R E       L          I    M M M M     C       U     U    T    S           
#    PPPPPP  RRRRRR  EEEEEEE L          I    M  M  M     C       U     U    T     SSSSS      
#    P       R   R   E       L          I    M     M     C       U     U    T          S     
#    P       R    R  E       L          I    M     M      C       U   U     T         S      
#    P       R     R EEEEEEE LLLLLLL IIIIIII M     M       CCCC    UUU      T    SSSSS
#    ===================================================================================
        
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
        GenParts = GenParts[twoFatJetsKin]
        output['cutflow']['two FatJets and jet kin'] += ak.to_awkward0(twoFatJetsKin).sum()
        
        # ---- Apply HT Cut ---- #
        hT = ak.to_awkward0(Jets.pt).sum()
        passhT = (hT > self.htCut)
        evtweights = evtweights[passhT]
        FatJets = FatJets[passhT]
        SubJets = SubJets[passhT]
        GenParts = GenParts[passhT]
        
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
        GenParts = GenParts[oneTTbar]
         
        # ---- Apply Delta Phi Cut for Back to Back Topology ---- #
        """ NOTE: Should find function for this; avoids 2pi problem """
        dPhiCut = ttbarcands.slot0.p4.delta_phi(ttbarcands.slot1.p4) > 2.1
        dPhiCut = ak.flatten(dPhiCut)
        output['cutflow']['dPhi > 2.1'] += ak.to_awkward0(dPhiCut).sum()
        ttbarcands = ttbarcands[dPhiCut]
        evtweights = evtweights[dPhiCut]
        FatJets = FatJets[dPhiCut] 
        SubJets = SubJets[dPhiCut] 
        GenParts = GenParts[dPhiCut]
        
        # ---- Identify subjets according to subjet ID ---- #
        hasSubjets0 = ((ttbarcands.slot0.subJetIdx1 > -1) & (ttbarcands.slot0.subJetIdx2 > -1)) # 1st candidate has two subjets
        hasSubjets1 = ((ttbarcands.slot1.subJetIdx1 > -1) & (ttbarcands.slot1.subJetIdx2 > -1)) # 2nd candidate has two subjets
        GoodSubjets = ak.flatten(((hasSubjets0) & (hasSubjets1))) # Selection of 4 (leading) subjects
   
        ttbarcands = ttbarcands[GoodSubjets] # Choose only ttbar candidates with this selection of subjets
        SubJets = SubJets[GoodSubjets]
        GenParts = GenParts[GoodSubjets]
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
        

#    ============================================================
#    TTTTTTT     TTTTTTT    A    GGGGGGG GGGGGGG EEEEEEE RRRRRR  
#       T           T      A A   G       G       E       R     R 
#       T           T     A   A  G       G       E       R     R 
#       T           T     AAAAA  G  GGGG G  GGGG EEEEEEE RRRRRR  
#       T           T    A     A G     G G     G E       R   R   
#       T           T    A     A G     G G     G E       R    R   
#       T           T    A     A  GGGGG   GGGGG  EEEEEEE R     R
#    ============================================================

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
        
#    ============================================================        
#    BBBBBB      TTTTTTT    A    GGGGGGG GGGGGGG EEEEEEE RRRRRR  
#    B     B        T      A A   G       G       E       R     R 
#    B     B        T     A   A  G       G       E       R     R 
#    BBBBBB         T     AAAAA  G  GGGG G  GGGG EEEEEEE RRRRRR  
#    B     B        T    A     A G     G G     G E       R   R   
#    B     B        T    A     A G     G G     G E       R    R   
#    BBBBBB         T    A     A  GGGGG   GGGGG  EEEEEEE R     R
#    ============================================================
        
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
                    ******************************************************************************************************
                    btag_wgts['mb'][n] --> w(n|m) --> "Probability" of n number of b-tags given m number of b-tagged jets
                    -----------------------------------------------------------------------------------------
                    w(0|0) = 1
                    
                    w(0|1), w(1|1) = 1 - BSF, 
                                   = BSF
                                   
                    w(0|2), w(1|2), w(2|2) = (1 - BSF_s0)(1 - BSF_s1), 
                                           = (1 - BSF_s0)BSF_s1 + BSF_s0(1 - BSF_s1),
                                           = (BSF_s0)(BSF_s1)
                                           
                    w(1|0), w(2|0), w(2|1) = Undef.
                    ******************************************************************************************************
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
                    
#    ===========================================================================================
#    PPPPPP     A    IIIIIII RRRRRR  IIIIIII N     N GGGGGGG     TTTTTTT EEEEEEE   SSSSS TTTTTTT 
#    P     P   A A      I    R     R    I    NN    N G              T    E        S         T    
#    P     P  A   A     I    R     R    I    N N   N G              T    E       S          T    
#    PPPPPP   AAAAA     I    RRRRRR     I    N  N  N G  GGGG        T    EEEEEEE  SSSSS     T    
#    P       A     A    I    R   R      I    N   N N G     G        T    E             S    T    
#    P       A     A    I    R    R     I    N    NN G     G        T    E            S     T    
#    P       A     A IIIIIII R     R IIIIIII N     N  GGGGG         T    EEEEEEE SSSSS      T  
#    ===========================================================================================

# ---- Subjets with Nearest b quarks ---- #
        isGenPart_bquark = (np.abs(GenParts.pdgId) == 5)
        
        # Start with SubJet01 #
        
                        # 1.) Group the subjet with gen level bquark #
        pairing_b01 = ak.cartesian([SubJet01.p4, GenParts.p4[isGenPart_bquark]]) 
                        # 2.) Check if pair exists #
        keepEvents_b01 = np.where(ak.count(pairing_b01.slot0.pt,-1) == 0, False, True) 
                        # 3.) Only keep pairs if the event exists (avoid empty array elements) #
        pairing_b01 = pairing_b01[keepEvents_b01] 
                        # 4.) Distance between subjet and bquark #
        deltaR_b01 = pairing_b01.slot0.delta_r(pairing_b01.slot1) 
                        # 5.) Select (index of) smallest distance (also flattens deltaR_01) #
        minimumR_indexb01 = ak.argmin(deltaR_b01, axis=-1) 
                        # 6.) is nearest genpart within SubJet01 radius? #
        deltaR_b01_new = deltaR_b01[np.arange(ak.size(deltaR_b01,0)),ak.to_numpy(minimumR_indexb01)]
        isQuarkWithinRadiusb01 = deltaR_b01_new < 0.4 
                        # 7.) Choose pairs with closest quark (min. index) that passes subjet radius cut (pass radius) #
        SubJet01_and_nearby_bquark = pairing_b01.slot0[np.arange(ak.size(pairing_b01.slot0,0)),ak.to_numpy(minimumR_indexb01)]
                        # Finally.) Used for Test Output #
        SubJet01_with_bquark = SubJet01_and_nearby_bquark[isQuarkWithinRadiusb01] 
        deltaR_b01_lessthanAK4 = deltaR_b01_new[isQuarkWithinRadiusb01]
        
        # Repeat all 7 steps and select desired subjets with quarks for SubJets 02, 11, and 12 #
        pairing_b02 = ak.cartesian([SubJet02.p4, GenParts.p4[isGenPart_bquark]]) 
        keepEvents_b02 = np.where(ak.count(pairing_b02.slot0.pt,-1) == 0, False, True)
        pairing_b02 = pairing_b02[keepEvents_b02]
        deltaR_b02 = pairing_b02.slot0.delta_r(pairing_b02.slot1)
        minimumR_indexb02 = ak.argmin(deltaR_b02, axis=-1) 
        deltaR_b02_new = deltaR_b01[np.arange(ak.size(deltaR_b02,0)),ak.to_numpy(minimumR_indexb02)]
        isQuarkWithinRadiusb02 = deltaR_b02_new < 0.4 
        SubJet02_and_nearby_bquark = pairing_b02.slot0[np.arange(ak.size(pairing_b02.slot0,0)),ak.to_numpy(minimumR_indexb02)]
        SubJet02_with_bquark = SubJet02_and_nearby_bquark[isQuarkWithinRadiusb02]
        deltaR_b02_lessthanAK4 = deltaR_b02_new[isQuarkWithinRadiusb02]
        
        pairing_b11 = ak.cartesian([SubJet11.p4, GenParts.p4[isGenPart_bquark]]) 
        keepEvents_b11 = np.where(ak.count(pairing_b11.slot0.pt,-1) == 0, False, True)
        pairing_b11 = pairing_b11[keepEvents_b11]
        deltaR_b11 = pairing_b11.slot0.delta_r(pairing_b11.slot1)
        minimumR_indexb11 = ak.argmin(deltaR_b11, axis=-1) 
        deltaR_b11_new = deltaR_b01[np.arange(ak.size(deltaR_b11,0)),ak.to_numpy(minimumR_indexb11)]
        isQuarkWithinRadiusb11 = deltaR_b11_new < 0.4 
        SubJet11_and_nearby_bquark = pairing_b11.slot0[np.arange(ak.size(pairing_b11.slot0,0)),ak.to_numpy(minimumR_indexb11)]
        SubJet11_with_bquark = SubJet11_and_nearby_bquark[isQuarkWithinRadiusb11]
        deltaR_b11_lessthanAK4 = deltaR_b11_new[isQuarkWithinRadiusb11]
        
        pairing_b12 = ak.cartesian([SubJet12.p4, GenParts.p4[isGenPart_bquark]]) 
        keepEvents_b12 = np.where(ak.count(pairing_b12.slot0.pt,-1) == 0, False, True)
        pairing_b12 = pairing_b12[keepEvents_b12]
        deltaR_b12 = pairing_b12.slot0.delta_r(pairing_b12.slot1)
        minimumR_indexb12 = ak.argmin(deltaR_b12, axis=-1) 
        deltaR_b12_new = deltaR_b12[np.arange(ak.size(deltaR_b12,0)),ak.to_numpy(minimumR_indexb12)]
        isQuarkWithinRadiusb12 = deltaR_b12_new < 0.4 
        SubJet12_and_nearby_bquark = pairing_b12.slot0[np.arange(ak.size(pairing_b12.slot0,0)),ak.to_numpy(minimumR_indexb12)]
        SubJet12_with_bquark = SubJet12_and_nearby_bquark[isQuarkWithinRadiusb12]
        deltaR_b12_lessthanAK4 = deltaR_b12_new[isQuarkWithinRadiusb12]
        
        """ ---------------------------------------------------------------------------------------------------------- """
        
        # ---- Subjets with Nearest c quarks ---- #  ak.fill_none(array, 0)
        isGenPart_cquark = (np.abs(GenParts.pdgId) == 4)
        
        pairing_c01 = ak.cartesian([SubJet01.p4, GenParts.p4[isGenPart_cquark]]) 
        keepEvents_c01 = np.where(ak.count(pairing_c01.slot0.pt,-1) == 0, False, True)
        bquark_contamination01 = np.where(keepEvents_b01, False, True) # If bquarks are also in this subjet, don't count this
        pairing_c01 = pairing_c01[keepEvents_c01]#, pairing_c01 = pairing_c01[bquark_contamination01]
        deltaR_c01 = pairing_c01.slot0.delta_r(pairing_c01.slot1) 
        minimumR_indexc01 = ak.argmin(deltaR_c01, axis=-1) 
        deltaR_c01_new = deltaR_c01[np.arange(ak.size(deltaR_c01,0)),ak.to_numpy(minimumR_indexc01)]
        isQuarkWithinRadiusc01 = deltaR_c01_new < 0.4 
        SubJet01_and_nearby_cquark = pairing_c01.slot0[np.arange(ak.size(pairing_c01.slot0,0)),ak.to_numpy(minimumR_indexc01)]
        SubJet01_with_cquark = SubJet01_and_nearby_cquark[isQuarkWithinRadiusc01] 
        deltaR_c01_lessthanAK4 = deltaR_c01_new[isQuarkWithinRadiusc01]
        
        pairing_c02 = ak.cartesian([SubJet02.p4, GenParts.p4[isGenPart_cquark]]) 
        keepEvents_c02 = np.where(ak.count(pairing_c02.slot0.pt,-1) == 0, False, True)
        pairing_c02 = pairing_c02[keepEvents_c02]
        deltaR_c02 = pairing_c02.slot0.delta_r(pairing_c02.slot1)
        minimumR_indexc02 = ak.argmin(deltaR_c02, axis=-1) 
        deltaR_c02_new = deltaR_c02[np.arange(ak.size(deltaR_c02,0)),ak.to_numpy(minimumR_indexc02)]
        isQuarkWithinRadiusc02 = deltaR_c02_new < 0.4 
        SubJet02_and_nearby_cquark = pairing_c02.slot0[np.arange(ak.size(pairing_c02.slot0,0)),ak.to_numpy(minimumR_indexc02)]
        SubJet02_with_cquark = SubJet02_and_nearby_cquark[isQuarkWithinRadiusc02]
        deltaR_c02_lessthanAK4 = deltaR_c02_new[isQuarkWithinRadiusc02]
        
        pairing_c11 = ak.cartesian([SubJet11.p4, GenParts.p4[isGenPart_cquark]]) 
        keepEvents_c11 = np.where(ak.count(pairing_c11.slot0.pt,-1) == 0, False, True)
        pairing_c11 = pairing_c11[keepEvents_c11]
        deltaR_c11 = pairing_c11.slot0.delta_r(pairing_c11.slot1)
        minimumR_indexc11 = ak.argmin(deltaR_c11, axis=-1)
        deltaR_c11_new = deltaR_c11[np.arange(ak.size(deltaR_c11,0)),ak.to_numpy(minimumR_indexc11)]
        isQuarkWithinRadiusc11 = deltaR_c11_new < 0.4 
        SubJet11_and_nearby_cquark = pairing_c11.slot0[np.arange(ak.size(pairing_c11.slot0,0)),ak.to_numpy(minimumR_indexc11)]
        SubJet11_with_cquark = SubJet11_and_nearby_cquark[isQuarkWithinRadiusc11]
        deltaR_c11_lessthanAK4 = deltaR_c11_new[isQuarkWithinRadiusc11]
        
        pairing_c12 = ak.cartesian([SubJet12.p4, GenParts.p4[isGenPart_cquark]]) 
        keepEvents_c12 = np.where(ak.count(pairing_c12.slot0.pt,-1) == 0, False, True)
        pairing_c12 = pairing_c12[keepEvents_c12]
        deltaR_c12 = pairing_c12.slot0.delta_r(pairing_c12.slot1)
        minimumR_indexc12 = ak.argmin(deltaR_c12, axis=-1) 
        deltaR_c12_new = deltaR_c12[np.arange(ak.size(deltaR_c12,0)),ak.to_numpy(minimumR_indexc12)]
        isQuarkWithinRadiusc12 = deltaR_c12_new < 0.4 
        SubJet12_and_nearby_cquark = pairing_c12.slot0[np.arange(ak.size(pairing_c12.slot0,0)),ak.to_numpy(minimumR_indexc12)]
        SubJet12_with_cquark = SubJet12_and_nearby_cquark[isQuarkWithinRadiusc12]
        deltaR_c12_lessthanAK4 = deltaR_c12_new[isQuarkWithinRadiusc12]
        
        """ ---------------------------------------------------------------------------------------------------------- """
        
        # ---- Subjets with Nearest light quarks ---- #
        isGenPart_1or2  = np.logical_or(np.abs(GenParts.pdgId) == 1, np.abs(GenParts.pdgId) == 2)
        isGenPart_3or21 = np.logical_or(np.abs(GenParts.pdgId) == 3, np.abs(GenParts.pdgId) == 21)
        isGenPart_lightquark = np.logical_or(isGenPart_1or2, isGenPart_3or21)
        
        pairing_l01 = ak.cartesian([SubJet01.p4, GenParts.p4[isGenPart_lightquark]]) 
        keepEvents_l01 = np.where(ak.count(pairing_l01.slot0.pt,-1) == 0, False, True)
        pairing_l01 = pairing_l01[keepEvents_l01]
        deltaR_l01 = pairing_l01.slot0.delta_r(pairing_l01.slot1) 
        minimumR_indexl01 = ak.argmin(deltaR_l01, axis=-1) 
        deltaR_l01_new = deltaR_l01[np.arange(ak.size(deltaR_l01,0)),ak.to_numpy(minimumR_indexl01)]
        isQuarkWithinRadiusl01 = deltaR_l01_new < 0.4 
        SubJet01_and_nearby_lquark = pairing_l01.slot0[np.arange(ak.size(pairing_l01.slot0,0)),ak.to_numpy(minimumR_indexl01)]
        SubJet01_with_lquark = SubJet01_and_nearby_lquark[isQuarkWithinRadiusl01] 
        deltaR_l01_lessthanAK4 = deltaR_l01_new[isQuarkWithinRadiusl01]
        
        pairing_l02 = ak.cartesian([SubJet02.p4, GenParts.p4[isGenPart_lightquark]]) 
        keepEvents_l02 = np.where(ak.count(pairing_l02.slot0.pt,-1) == 0, False, True)
        pairing_l02 = pairing_l02[keepEvents_l02]
        deltaR_l02 = pairing_l02.slot0.delta_r(pairing_l02.slot1)
        minimumR_indexl02 = ak.argmin(deltaR_l02, axis=-1) 
        deltaR_l02_new = deltaR_l02[np.arange(ak.size(deltaR_l02,0)),ak.to_numpy(minimumR_indexl02)]
        isQuarkWithinRadiusl02 = deltaR_l02_new < 0.4 
        SubJet02_and_nearby_lquark = pairing_l02.slot0[np.arange(ak.size(pairing_l02.slot0,0)),ak.to_numpy(minimumR_indexl02)]
        SubJet02_with_lquark = SubJet02_and_nearby_lquark[isQuarkWithinRadiusl02]
        deltaR_l02_lessthanAK4 = deltaR_l02_new[isQuarkWithinRadiusl02]
        
        pairing_l11 = ak.cartesian([SubJet11.p4, GenParts.p4[isGenPart_lightquark]]) 
        keepEvents_l11 = np.where(ak.count(pairing_l11.slot0.pt,-1) == 0, False, True)
        pairing_l11 = pairing_l11[keepEvents_l11]
        deltaR_l11 = pairing_l11.slot0.delta_r(pairing_l11.slot1)
        minimumR_indexl11 = ak.argmin(deltaR_l11, axis=-1) 
        deltaR_l11_new = deltaR_l11[np.arange(ak.size(deltaR_l11,0)),ak.to_numpy(minimumR_indexl11)]
        isQuarkWithinRadiusl11 = deltaR_l11_new < 0.4 
        SubJet11_and_nearby_lquark = pairing_l11.slot0[np.arange(ak.size(pairing_l11.slot0,0)),ak.to_numpy(minimumR_indexl11)]
        SubJet11_with_lquark = SubJet11_and_nearby_lquark[isQuarkWithinRadiusl11]
        deltaR_l11_lessthanAK4 = deltaR_l11_new[isQuarkWithinRadiusl11]
        
        pairing_l12 = ak.cartesian([SubJet12.p4, GenParts.p4[isGenPart_lightquark]]) 
        keepEvents_l12 = np.where(ak.count(pairing_l12.slot0.pt,-1) == 0, False, True)
        pairing_l12 = pairing_l12[keepEvents_l12]
        deltaR_l12 = pairing_l12.slot0.delta_r(pairing_l12.slot1)
        minimumR_indexl12 = ak.argmin(deltaR_l12, axis=-1) 
        deltaR_l12_new = deltaR_l12[np.arange(ak.size(deltaR_l12,0)),ak.to_numpy(minimumR_indexl12)]
        isQuarkWithinRadiusl12 = deltaR_l12_new < 0.4 
        SubJet12_and_nearby_lquark = pairing_l12.slot0[np.arange(ak.size(pairing_l12.slot0,0)),ak.to_numpy(minimumR_indexl12)]
        SubJet12_with_lquark = SubJet12_and_nearby_lquark[isQuarkWithinRadiusl12]
        deltaR_l12_lessthanAK4 = deltaR_l12_new[isQuarkWithinRadiusl12]
        
        # ---- Compare these plots as a test ---- # 
        # ---- Flavors from s01 should agree well with genpart matching ---- #
        # ---- B flavor from all 4 subjets should agree well with genpart matching ---- #
        
        # Check the SubJets' SubJet_hadronFlavor #
        isSubJet01_bflavor = (flav_s01 == 5)
        isSubJet02_bflavor = (flav_s02 == 5)
        isSubJet11_bflavor = (flav_s11 == 5)
        isSubJet12_bflavor = (flav_s12 == 5)
        
        isSubJet01_cflavor = (flav_s01 == 4)
        isSubJet02_cflavor = (flav_s02 == 4)
        isSubJet11_cflavor = (flav_s11 == 4)
        isSubJet12_cflavor = (flav_s12 == 4)
        
        isSubJet01_lightflavor = if_s01_isLightParton # Defined previously...
        isSubJet02_lightflavor = if_s02_isLightParton # Defined previously...
        isSubJet11_lightflavor = if_s11_isLightParton # Defined previously...
        isSubJet12_lightflavor = if_s12_isLightParton # Defined previously...
        
        # SubJets #
        subjet01_pt = ak.flatten(SubJet01.p4.pt)
        subjet02_pt = ak.flatten(SubJet02.p4.pt)
        subjet11_pt = ak.flatten(SubJet11.p4.pt)
        subjet12_pt = ak.flatten(SubJet12.p4.pt)
        
        # SubJets that are given a SubJet_hadronFlavor #
        subjet01_bflavor_pt = ak.flatten(SubJet01[isSubJet01_bflavor].p4.pt)
        subjet02_bflavor_pt = ak.flatten(SubJet02[isSubJet02_bflavor].p4.pt)
        subjet11_bflavor_pt = ak.flatten(SubJet11[isSubJet11_bflavor].p4.pt)
        subjet12_bflavor_pt = ak.flatten(SubJet12[isSubJet12_bflavor].p4.pt)
        
        subjet01_cflavor_pt = ak.flatten(SubJet01[isSubJet01_cflavor].p4.pt)
        subjet02_cflavor_pt = ak.flatten(SubJet02[isSubJet02_cflavor].p4.pt)
        subjet11_cflavor_pt = ak.flatten(SubJet11[isSubJet11_cflavor].p4.pt)
        subjet12_cflavor_pt = ak.flatten(SubJet12[isSubJet12_cflavor].p4.pt)
        
        subjet01_lightflavor_pt = ak.flatten(SubJet01[isSubJet01_lightflavor].p4.pt)
        subjet02_lightflavor_pt = ak.flatten(SubJet02[isSubJet02_lightflavor].p4.pt)
        subjet11_lightflavor_pt = ak.flatten(SubJet11[isSubJet11_lightflavor].p4.pt)
        subjet12_lightflavor_pt = ak.flatten(SubJet12[isSubJet12_lightflavor].p4.pt)
        
        # SubJets that are paired with a GenPart_pdgId #
        subjet01_with_bquark_pt = SubJet01_with_bquark.pt
        subjet02_with_bquark_pt = SubJet02_with_bquark.pt
        subjet11_with_bquark_pt = SubJet11_with_bquark.pt
        subjet12_with_bquark_pt = SubJet12_with_bquark.pt
        
        subjet01_with_cquark_pt = SubJet01_with_cquark.pt
        subjet02_with_cquark_pt = SubJet02_with_cquark.pt
        subjet11_with_cquark_pt = SubJet11_with_cquark.pt
        subjet12_with_cquark_pt = SubJet12_with_cquark.pt
        
        subjet01_with_lightquark_pt = SubJet01_with_lquark.pt
        subjet02_with_lightquark_pt = SubJet02_with_lquark.pt
        subjet11_with_lightquark_pt = SubJet11_with_lquark.pt
        subjet12_with_lightquark_pt = SubJet12_with_lquark.pt
        
        # ******************************************************************************** #
        # ------------------------------- Outputs for Test ------------------------------- #
        # ******************************************************************************** #
        output['subjet01_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet01_pt),
                                  weight = ak.to_numpy(evtweights))
        output['subjet02_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet02_pt),
                                  weight = ak.to_numpy(evtweights))
        output['subjet11_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet11_pt),
                                  weight = ak.to_numpy(evtweights))
        output['subjet12_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet12_pt),
                                  weight = ak.to_numpy(evtweights))
        
        
        output['subjet01_bflavor_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet01_bflavor_pt),
                                  weight = ak.to_numpy(evtweights[isSubJet01_bflavor]))
        output['subjet01_with_bquark_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet01_with_bquark_pt),
                                  weight = ak.to_numpy(evtweights[isQuarkWithinRadiusb01]))
        output['subjet02_bflavor_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet02_bflavor_pt),
                                  weight = ak.to_numpy(evtweights[isSubJet02_bflavor]))
        output['subjet02_with_bquark_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet02_with_bquark_pt),
                                  weight = ak.to_numpy(evtweights[isQuarkWithinRadiusb02]))
        output['subjet11_bflavor_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet11_bflavor_pt),
                                  weight = ak.to_numpy(evtweights[isSubJet11_bflavor]))
        output['subjet11_with_bquark_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet11_with_bquark_pt),
                                  weight = ak.to_numpy(evtweights[isQuarkWithinRadiusb11]))
        output['subjet12_bflavor_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet12_bflavor_pt),
                                  weight = ak.to_numpy(evtweights[isSubJet12_bflavor]))
        output['subjet12_with_bquark_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet12_with_bquark_pt),
                                  weight = ak.to_numpy(evtweights[isQuarkWithinRadiusb12]))
        
        
        output['subjet01_cflavor_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet01_cflavor_pt),
                                  weight = ak.to_numpy(evtweights[isSubJet01_cflavor]))
        output['subjet01_with_cquark_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet01_with_cquark_pt),
                                  weight = ak.to_numpy(evtweights[isQuarkWithinRadiusc01]))
        output['subjet02_cflavor_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet02_cflavor_pt),
                                  weight = ak.to_numpy(evtweights[isSubJet02_cflavor]))
        output['subjet02_with_cquark_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet02_with_cquark_pt),
                                  weight = ak.to_numpy(evtweights[isQuarkWithinRadiusc02]))
        output['subjet11_cflavor_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet11_cflavor_pt),
                                  weight = ak.to_numpy(evtweights[isSubJet11_cflavor]))
        output['subjet11_with_cquark_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet11_with_cquark_pt),
                                  weight = ak.to_numpy(evtweights[isQuarkWithinRadiusc11]))
        output['subjet12_cflavor_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet12_cflavor_pt),
                                  weight = ak.to_numpy(evtweights[isSubJet12_cflavor]))
        output['subjet12_with_cquark_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet12_with_cquark_pt),
                                  weight = ak.to_numpy(evtweights[isQuarkWithinRadiusc12]))
        
        
        output['subjet01_lightflavor_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet01_lightflavor_pt),
                                  weight = ak.to_numpy(evtweights[isSubJet01_lightflavor]))
        output['subjet01_with_lightquark_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet01_with_lightquark_pt),
                                  weight = ak.to_numpy(evtweights[isQuarkWithinRadiusl01]))
        output['subjet02_lightflavor_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet02_lightflavor_pt),
                                  weight = ak.to_numpy(evtweights[isSubJet02_lightflavor]))
        output['subjet02_with_lightquark_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet02_with_lightquark_pt),
                                  weight = ak.to_numpy(evtweights[isQuarkWithinRadiusl02]))
        output['subjet11_lightflavor_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet11_lightflavor_pt),
                                  weight = ak.to_numpy(evtweights[isSubJet11_lightflavor]))
        output['subjet11_with_lightquark_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet11_with_lightquark_pt),
                                  weight = ak.to_numpy(evtweights[isQuarkWithinRadiusl11]))
        output['subjet12_lightflavor_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet12_lightflavor_pt),
                                  weight = ak.to_numpy(evtweights[isSubJet12_lightflavor]))
        output['subjet12_with_lightquark_pt'].fill(dataset = dataset,
                                  subjetpt = ak.to_numpy(subjet12_with_lightquark_pt),
                                  weight = ak.to_numpy(evtweights[isQuarkWithinRadiusl12]))
        
        
        # ---- Delta R to nearest b quark and c quark, for subjets that are identified as “5” or “4". ---- #
        
        output['subjet01_bquark_distance'].fill(dataset = dataset,
                                               delta_r = ak.to_numpy(deltaR_b01_lessthanAK4),
                                               weight = ak.to_numpy(evtweights[isQuarkWithinRadiusb01]))
        output['subjet02_bquark_distance'].fill(dataset = dataset,
                                               delta_r = ak.to_numpy(deltaR_b02_lessthanAK4),
                                               weight = ak.to_numpy(evtweights[isQuarkWithinRadiusb02]))
        output['subjet11_bquark_distance'].fill(dataset = dataset,
                                               delta_r = ak.to_numpy(deltaR_b11_lessthanAK4),
                                               weight = ak.to_numpy(evtweights[isQuarkWithinRadiusb11]))
        output['subjet12_bquark_distance'].fill(dataset = dataset,
                                               delta_r = ak.to_numpy(deltaR_b12_lessthanAK4),
                                               weight = ak.to_numpy(evtweights[isQuarkWithinRadiusb12]))
        
        output['subjet01_cquark_distance'].fill(dataset = dataset,
                                               delta_r = ak.to_numpy(deltaR_c01_lessthanAK4),
                                               weight = ak.to_numpy(evtweights[isQuarkWithinRadiusc01]))
        output['subjet02_cquark_distance'].fill(dataset = dataset,
                                               delta_r = ak.to_numpy(deltaR_c02_lessthanAK4),
                                               weight = ak.to_numpy(evtweights[isQuarkWithinRadiusc02]))
        output['subjet11_cquark_distance'].fill(dataset = dataset,
                                               delta_r = ak.to_numpy(deltaR_c11_lessthanAK4),
                                               weight = ak.to_numpy(evtweights[isQuarkWithinRadiusc11]))
        output['subjet12_cquark_distance'].fill(dataset = dataset,
                                               delta_r = ak.to_numpy(deltaR_c12_lessthanAK4),
                                               weight = ak.to_numpy(evtweights[isQuarkWithinRadiusc12]))
        
        output['subjet01_lightquark_distance'].fill(dataset = dataset,
                                               delta_r = ak.to_numpy(deltaR_l01_lessthanAK4),
                                               weight = ak.to_numpy(evtweights[isQuarkWithinRadiusl01]))
        output['subjet02_lightquark_distance'].fill(dataset = dataset,
                                               delta_r = ak.to_numpy(deltaR_l02_lessthanAK4),
                                               weight = ak.to_numpy(evtweights[isQuarkWithinRadiusl02]))
        output['subjet11_lightquark_distance'].fill(dataset = dataset,
                                               delta_r = ak.to_numpy(deltaR_l11_lessthanAK4),
                                               weight = ak.to_numpy(evtweights[isQuarkWithinRadiusl11]))
        output['subjet12_lightquark_distance'].fill(dataset = dataset,
                                               delta_r = ak.to_numpy(deltaR_l12_lessthanAK4),
                                               weight = ak.to_numpy(evtweights[isQuarkWithinRadiusl12]))
        
#    ================================================================
#       A    N     N    A    L       Y     Y   SSSSS IIIIIII   SSSSS 
#      A A   NN    N   A A   L        Y   Y   S         I     S      
#     A   A  N N   N  A   A  L         Y Y   S          I    S       
#     AAAAA  N  N  N  AAAAA  L          Y     SSSSS     I     SSSSS  
#    A     A N   N N A     A L          Y          S    I          S 
#    A     A N    NN A     A L          Y         S     I         S  
#    A     A N     N A     A LLLLLLL    Y    SSSSS   IIIIIII SSSSS   
#    ================================================================
        
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
        
        # ---- Weights ---- #
        weights = evtweights

        # ---- Define the SumW2 for MC Datasets (Probably unnecessary now) ---- #
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
        
        df = pd.DataFrame({"momentum":p}) # DataFrame used for finding values in LookUp Tables
        
        for ilabel,icat in labels_and_categories.items():
            ###------------------------------------------------------------------------------------------###
            ### ------------------------------------ Mistag Scaling ------------------------------------ ###
            ###------------------------------------------------------------------------------------------###
            if self.UseLookUpTables == True:
                # ---- Weight ttbar M.C. and data by mistag from data (corresponding to its year) ---- #
                # -- Pick out proper JetHT year mistag for TTbar sim. -- #
                if 'TTbar_' in dataset:
                    file_df = self.lu['JetHT' + dataset[-4:] + '_Data']['at' + str(ilabel[-5:])] 
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
            ###---------------------------------------------------------------------------------------------###
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
                
                ttbarcands_modmass = ak.cartesian([jet0.p4, jet1_modp4])

                # ---- Apply Necessary Selections to new modmass version ---- #
                ttbarcands_modmass = ttbarcands_modmass[oneTTbar]
                ttbarcands_modmass = ttbarcands_modmass[dPhiCut]
                ttbarcands_modmass = ttbarcands_modmass[GoodSubjets]
                
                # ---- Manually sum the modmass p4 candidates (Coffea technicality) ---- #
                ttbarcands_modmassp4sum = ttbarcands.slot0.p4.add(ttbarcands.slot1.p4)
                
                # ---- Re-define Mass Variables for ModMass Procedure (pt, eta, phi are redundant to change) ---- #
                ttbarmass = ak.flatten(ttbarcands_modmassp4sum.mass)
                jetmass = ak.flatten(ttbarcands_modmass.slot1.mass)
                
            ###---------------------------------------------------------------------------------------------###
            ### ------------------------------ B-Tag Weighting (S.F. Only) -------------------------------- ###
            ###---------------------------------------------------------------------------------------------###
            if (self.ApplySF == True) and (self.UseEfficiencies == False):
 
                if '0b' in ilabel:
                    Weights = Weights*Wgts_to_0btag_region_nonzero
                elif '1b' in ilabel:
                    Weights = Weights*Wgts_to_1btag_region_nonzero 
                else:
                    Weights = Weights*Wgts_to_2btag_region_nonzero 

# ************************************************************************************************************ #      
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

