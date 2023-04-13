#!/usr/bin/env python 
# coding: utf-8

from coffea import processor, nanoevents
from coffea import util
from coffea.btag_tools import BTagScaleFactor
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import vector
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor
from collections import defaultdict
import sys
import os, psutil
import copy
import scipy.stats as ss
import numpy as np
import itertools
import pandas as pd
from numpy.random import RandomState
import correctionlib
import hist
# from correctionlib.schemav2 import Correction
import re as regularexpressions

import awkward as ak

from cms_utils import getLumiMaskRun2

ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)

# --- Define 'Manual bins' to use for mistag plots for aesthetic purposes--- #
from constants import *

"""Package to collect MC flavor efficiencies used for btagging systematics in main analysis (TTbarResProcessor) """
class MCFlavorEfficiencyProcessor(processor.ProcessorABC):
    def __init__(self, prng=RandomState(1234567890), htCut=950., bdisc=0.8484, ak8PtMin=400., year=None, apv='', vfp='', RandomDebugMode=False):
        
        self.prng = prng
        self.htCut = htCut
        self.ak8PtMin = ak8PtMin
        self.year = year
        self.apv = apv
        self.vfp = vfp
        self.bdisc = bdisc
        self.RandomDebugMode = RandomDebugMode
        
        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary Dataset")

        ttbarmass_axis = hist.axis.Regular(50, 800, 8000, name="ttbarmass", label=r"$m_{t\bar{t}}$ [GeV]")
        
        subjetpt_axis = hist.axis.Regular(25, 0, 2000, name = "subjetpt", label = "SubJet $p_{T}$ [GeV]")
        subjetpt_laxis = hist.axis.Regular( 8, 0, 2000, name = "subjetpt", label = r"SubJet $p_{T}$ [GeV]") #Larger bins
        subjetpt_maxis = hist.axis.Variable(manual_subjetpt_bins, name = "subjetpt", label = r"SubJet $p_T$ [GeV]") #Manually defined bins for better statistics per bin
        
        subjeteta_axis = hist.axis.Regular(25, 0, 2.4, name = "subjeteta", label = r"SubJet $\eta$")
        subjeteta_laxis = hist.axis.Regular(8, 0, 2.4, name = "subjeteta", label = r"SubJet $\eta$") #Larger bins
        subjeteta_maxis = hist.axis.Variable(manual_subjeteta_bins, name = "subjeteta", label = r"SubJet $\eta$") #Manually defined bins for better statistics per bin

        
#    ====================================================================
#    EEEEEEE FFFFFFF FFFFFFF      H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
#    E       F       F            H     H    I     S         T     S          
#    E       F       F            H     H    I    S          T    S           
#    EEEEEEE FFFFFFF FFFFFFF      HHHHHHH    I     SSSSS     T     SSSSS      
#    E       F       F            H     H    I          S    T          S     
#    E       F       F            H     H    I         S     T         S      
#    EEEEEEE F       F       *    H     H IIIIIII SSSSS      T    SSSSS
#    ====================================================================

        self.histo_dict = {
            # ---- 2D SubJet b-tag Efficiency ---- #
            'b_eff_numerator_s01': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s02': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s11': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s12': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            
            'b_eff_denominator_s01': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s02': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s11': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s12': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            
            'b_eff_numerator_s01_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s02_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s11_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s12_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            
            'b_eff_denominator_s01_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s02_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s11_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s12_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            
            'b_eff_numerator_s01_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s02_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s11_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s12_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            
            'b_eff_denominator_s01_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s02_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s11_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s12_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            
            # ---- 2D SubJet c-tag Efficiency ---- #
            'c_eff_numerator_s01': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s02': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s11': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s12': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            
            'c_eff_denominator_s01': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s02': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s11': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s12': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            
            'c_eff_numerator_s01_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s02_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s11_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s12_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            
            'c_eff_denominator_s01_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s02_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s11_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s12_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            
            'c_eff_numerator_s01_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s02_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s11_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s12_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            
            'c_eff_denominator_s01_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s02_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s11_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s12_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            
            # ---- 2D SubJet light quark-tag Efficiency ---- #
            'udsg_eff_numerator_s01': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s02': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s11': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s12': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            
            'udsg_eff_denominator_s01': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s02': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s11': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s12': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            
            'udsg_eff_numerator_s01_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s02_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s11_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s12_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            
            'udsg_eff_denominator_s01_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s02_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s11_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s12_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            
            'udsg_eff_numerator_s01_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s02_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s11_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s12_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            
            'udsg_eff_denominator_s01_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s02_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s11_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s12_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            
            'cutflow': processor.defaultdict_accumulator(int),
        }
                                                          
#   =======================================================================
#   FFFFFFF U     U N     N   CCCC  TTTTTTT IIIIIII   OOO   N     N   SSSSS     
#   F       U     U NN    N  C         T       I     O   O  NN    N  S          
#   F       U     U N N   N C          T       I    O     O N N   N S           
#   FFFFFFF U     U N  N  N C          T       I    O     O N  N  N  SSSSS      
#   F       U     U N   N N C          T       I    O     O N   N N       S     
#   F        U   U  N    NN  C         T       I     O   O  N    NN      S      
#   F         UUU   N     N   CCCC     T    IIIIIII   OOO   N     N SSSSS
#   =======================================================================

    
    def GetFlavorEfficiency(self, Subjet, Flavor): # Return "Flavor" efficiency numerator and denominator
        '''
        Subjet --> awkward array object after preselection i.e. SubJetXY
        Flavor --> integer i.e 5, 4, or 0 (b, c, or udsg)
        '''
        # --- Define pT and Eta for Both Candidates' Subjets (for simplicity) --- #
        pT = ak.flatten(Subjet.pt) # pT of subjet in ttbarcand 
        eta = np.abs(ak.flatten(Subjet.eta)) # eta of 1st subjet in ttbarcand 
        flav = np.abs(ak.flatten(Subjet.hadronFlavour)) # either 'normal' or 'anti' quark
        
        subjet_btagged = (Subjet.btagCSVV2 > self.bdisc)
        
        Eff_Num_pT = np.where(subjet_btagged & (flav == Flavor), pT, -1) # if not collecting pT of subjet, then put non exisitent bin, i.e. -1
        Eff_Num_eta = np.where(subjet_btagged & (flav == Flavor), eta, -1) # if not collecting eta of subjet, then put non exisitent bin, i.e. 5
        
        Eff_Num_pT = ak.flatten(Eff_Num_pT) # extra step needed for numerator to gaurantee proper shape for filling hists
        Eff_Num_eta = ak.flatten(Eff_Num_eta)
        
        Eff_Denom_pT = np.where(flav == Flavor, pT, -1)
        Eff_Denom_eta = np.where(flav == Flavor, eta, -1)
        
        EffStuff = {
            'Num_pT' : Eff_Num_pT,
            'Num_eta' : Eff_Num_eta,
            'Denom_pT' : Eff_Denom_pT,
            'Denom_eta' : Eff_Denom_eta,
        }
        
        return EffStuff
            
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.histo_dict
        
        # ---- Define dataset ---- #
        dataset = events.metadata['dataset']

#    ===========================================================================================    
#    N     N    A    N     N   OOO      A      OOO   DDDD          OOO   BBBBBB  JJJJJJJ   SSSSS     
#    NN    N   A A   NN    N  O   O    A A    O   O  D   D        O   O  B     B    J     S          
#    N N   N  A   A  N N   N O     O  A   A  O     O D    D      O     O B     B    J    S           
#    N  N  N  AAAAA  N  N  N O     O  AAAAA  O     O D     D     O     O BBBBBB     J     SSSSS      
#    N   N N A     A N   N N O     O A     A O     O D    D      O     O B     B J  J          S     
#    N    NN A     A N    NN  O   O  A     A  O   O  D   D        O   O  B     B J  J         S      
#    N     N A     A N     N   OOO   A     A   OOO   DDDD          OOO   BBBBBB   JJ     SSSSS 
#    =========================================================================================== 

        isData = ('JetHT' in dataset) or ('SingleMu' in dataset)
        
        # ---- Define AK8 Jets as FatJets ---- #
        #FatJets = events.FatJet # Everything should already be defined in here.  example) df['FatJet_pt] -> events.FatJet.pt
        FatJets = ak.zip({
            "run": events.run,
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
        Jets = ak.zip({
            "run": events.run,
            "pt": events.Jet_pt,
            "eta": events.Jet_eta,
            "phi": events.Jet_phi,
            "mass": events.Jet_mass,
            "area": events.Jet_area,
            "p4": ak.zip({
                "pt": events.Jet_pt,
                "eta": events.Jet_eta,
                "phi": events.Jet_phi,
                "mass": events.Jet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })

        # ---- Define SubJets ---- #
        SubJets = ak.zip({
            "run": events.run,
            "pt": events.SubJet_pt,
            "eta": events.SubJet_eta,
            "phi": events.SubJet_phi,
            "mass": events.SubJet_mass,
            "btagDeepB": events.SubJet_btagDeepB,
            "btagCSVV2": events.SubJet_btagCSVV2,
            "p4": ak.zip({
                "pt": events.SubJet_pt,
                "eta": events.SubJet_eta,
                "phi": events.SubJet_phi,
                "mass": events.SubJet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })
        
        Jets['hadronFlavour'] = events.Jet_hadronFlavour
        SubJets['hadronFlavour'] = events.SubJet_hadronFlavour
            
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
        if isData: # If data is used...
            # print('if isData command works')
            evtweights = np.ones(len(events) ) # set all "data weights" to one
        else: # if Monte Carlo dataset is used...
            evtweights = events.Generator_weight
        # ---- Show all events ---- #
        output['cutflow']['all events'] += len(events)

        # ---- Apply HT Cut ---- #
        # ---- This gives the analysis 99.8% efficiency (see 2016 AN) ---- #
        hT = ak.sum(Jets.pt, axis=1)
        passhT = (hT > self.htCut)
        FatJets = FatJets[passhT]
        Jets = Jets[passhT] # this used to not be here
        SubJets = SubJets[passhT]
        evtweights = evtweights[passhT]
        output['cutflow']['HT Cut'] += len(FatJets)
           
        # ---- Jets that satisfy Jet ID ---- #
        jet_id = (FatJets.jetId > 0) # Loose jet ID
        FatJets = FatJets[jet_id]
        output['cutflow']['Loose Jet ID'] += len(FatJets)
        
        # ---- Apply pT Cut and Rapidity Window ---- #
        FatJets_rapidity = .5*np.log( (FatJets.p4.energy + FatJets.p4.pz)/(FatJets.p4.energy - FatJets.p4.pz) )
        jetkincut_index = (FatJets.pt > self.ak8PtMin) & (np.abs(FatJets_rapidity) < 2.4)
        FatJets = FatJets[ jetkincut_index ]
        output['cutflow']['pT,y Cut'] += len(FatJet)
        
        # ---- Find two AK8 Jets ---- #
        twoFatJetsKin = (ak.num(FatJets, axis=-1) > 1)
        FatJets = FatJets[twoFatJetsKin]
        SubJets = SubJets[twoFatJetsKin]
        Jets = Jets[twoFatJetsKin] # this used to not be here
        evtweights = evtweights[twoFatJetsKin]
        output['cutflow']['two FatJets'] += len(FatJets)
        
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
        oneTTbar = (ak.num(ttbarcands, axis=-1) > 0)
        ttbarcands = ttbarcands[oneTTbar]
        FatJets = FatJets[oneTTbar]
        Jets = Jets[oneTTbar] # this used to not be here
        SubJets = SubJets[oneTTbar]
        evtweights = evtweights[oneTTbar]
        output['cutflow']['>= oneTTbar'] += len(FatJets)            

        # ---- Apply Delta Phi Cut for Back to Back Topology ---- #
        """ NOTE: Should find function for this; avoids 2pi problem """
        dPhiCut = ttbarcands.slot0.p4.delta_phi(ttbarcands.slot1.p4) > 2.1
        dPhiCut = ak.flatten(dPhiCut)
        ttbarcands = ttbarcands[dPhiCut]
        FatJets = FatJets[dPhiCut] 
        Jets = Jets[dPhiCut] # this used to not be here
        SubJets = SubJets[dPhiCut] 
        evtweights = evtweights[dPhiCut]
        output['cutflow']['dPhi Cut'] += len(FatJets)
        
        # ---- Identify subjets according to subjet ID ---- #
        hasSubjets0 = ((ttbarcands.slot0.subJetIdx1 > -1) & (ttbarcands.slot0.subJetIdx2 > -1)) # 1st candidate has two subjets
        hasSubjets1 = ((ttbarcands.slot1.subJetIdx1 > -1) & (ttbarcands.slot1.subJetIdx2 > -1)) # 2nd candidate has two subjets
        GoodSubjets = ak.flatten(((hasSubjets0) & (hasSubjets1))) # Selection of 4 (leading) subjects
        FatJets = FatJets[GoodSubjets]
        ttbarcands = ttbarcands[GoodSubjets] # Choose only ttbar candidates with this selection of subjets
        SubJets = SubJets[GoodSubjets]
        Jets = Jets[GoodSubjets] # this used to not be here
        evtweights = evtweights[GoodSubjets]
        output['cutflow']['Good Subjets'] += len(FatJets)
        
        SubJet01 = SubJets[ttbarcands.slot0.subJetIdx1] # ttbarcandidate 0's first subjet 
        SubJet02 = SubJets[ttbarcands.slot0.subJetIdx2] # ttbarcandidate 0's second subjet
        SubJet11 = SubJets[ttbarcands.slot1.subJetIdx1] # ttbarcandidate 1's first subjet 
        SubJet12 = SubJets[ttbarcands.slot1.subJetIdx2] # ttbarcandidate 1's second subjet
        
#       ===============================================================================
#       M     M   CCCC      FFFFFFF L          A    V     V     EEEEEEE FFFFFFF FFFFFFF     
#       MM   MM  C          F       L         A A   V     V     E       F       F           
#       M M M M C           F       L        A   A  V     V     E       F       F           
#       M  M  M C           FFFFFFF L        AAAAA  V     V     EEEEEEE FFFFFFF FFFFFFF     
#       M     M C           F       L       A     A  V   V      E       F       F           
#       M     M  C          F       L       A     A   V V       E       F       F           
#       M     M   CCCC      F       LLLLLLL A     A    V        EEEEEEE F       F  
#       ===============================================================================
        
        if not isData:

            bFlavEff01 = self.GetFlavorEfficiency(SubJet01, 5)
            bFlavEff02 = self.GetFlavorEfficiency(SubJet02, 5)
            bFlavEff11 = self.GetFlavorEfficiency(SubJet11, 5)
            bFlavEff12 = self.GetFlavorEfficiency(SubJet12, 5)

            cFlavEff01 = self.GetFlavorEfficiency(SubJet01, 4)
            cFlavEff02 = self.GetFlavorEfficiency(SubJet02, 4)
            cFlavEff11 = self.GetFlavorEfficiency(SubJet11, 4)
            cFlavEff12 = self.GetFlavorEfficiency(SubJet12, 4)

            lFlavEff01 = self.GetFlavorEfficiency(SubJet01, 0)
            lFlavEff02 = self.GetFlavorEfficiency(SubJet02, 0)
            lFlavEff11 = self.GetFlavorEfficiency(SubJet11, 0)
            lFlavEff12 = self.GetFlavorEfficiency(SubJet12, 0)

            # **************************************************************************************** #
            # ----------------------------- 2-D B-tagging Efficiencies ------------------------------- #
            # **************************************************************************************** #
            output['b_eff_numerator_s01'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s02'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s11'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s12'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['b_eff_denominator_s01'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s02'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s11'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s12'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['b_eff_numerator_s01_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s02_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s11_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s12_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['b_eff_denominator_s01_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s02_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s11_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s12_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['b_eff_numerator_s01_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s02_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s11_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s12_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['b_eff_denominator_s01_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s02_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s11_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s12_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))

            # **************************************************************************************** #
            # ----------------------------- 2-D C-tagging Efficiencies ------------------------------- #
            # **************************************************************************************** #
            output['c_eff_numerator_s01'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s02'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s11'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s12'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['c_eff_denominator_s01'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s02'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s11'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s12'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['c_eff_numerator_s01_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s02_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s11_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s12_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['c_eff_denominator_s01_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s02_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s11_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s12_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['c_eff_numerator_s01_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s02_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s11_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s12_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['c_eff_denominator_s01_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s02_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s11_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s12_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))

            # **************************************************************************************** #
            # ------------------------ 2-D Light Parton-tagging Efficiencies ------------------------- #
            # **************************************************************************************** #
            output['udsg_eff_numerator_s01'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s02'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s11'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s12'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['udsg_eff_denominator_s01'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s02'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s11'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s12'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['udsg_eff_numerator_s01_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s02_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s11_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s12_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['udsg_eff_denominator_s01_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s02_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s11_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s12_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['udsg_eff_numerator_s01_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s02_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s11_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s12_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['udsg_eff_denominator_s01_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s02_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s11_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s12_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
 
        return output

    def postprocess(self, accumulator):
        return accumulator    
    
#===================================================================================================================================================    
#=================================================================================================================================================== 
#=================================================================================================================================================== 
#=================================================================================================================================================== 
#=================================================================================================================================================== 
#=================================================================================================================================================== 
