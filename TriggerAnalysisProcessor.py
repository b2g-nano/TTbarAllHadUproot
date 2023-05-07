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

from constants import *
       
"""Package to perform trigger analysis before applying triggers in main analysis (TTbarResProcessor)"""
class TriggerAnalysisProcessor(processor.ProcessorABC):
    def __init__(self, prng=RandomState(1234567890), htCut=950., minMSD=105., maxMSD=210.,
                 tau32Cut=0.65, ak8PtMin=400., bdisc=0.8484, deepAK8Cut=0.435,
                 year=None, apv='', vfp='', RandomDebugMode=False):
        
        self.prng = prng
        self.htCut = htCut
        self.minMSD = minMSD
        self.maxMSD = maxMSD
        self.tau32Cut = tau32Cut
        self.ak8PtMin = ak8PtMin
        self.bdisc = bdisc
        self.deepAK8Cut = deepAK8Cut
        self.year = year
        self.apv = apv
        self.vfp = vfp
        self.RandomDebugMode = RandomDebugMode

        
        # --- 0, >=1 ttags --- #
        self.ttagcats_forTriggerAnalysis = ["NoCut", ">=1t"]                                     
        self.label_dict = {i: label for i, label in enumerate(self.ttagcats_forTriggerAnalysis)}
                                                          
        dataset_axis = hist.axis.StrCategory([], growth=True, name = "dataset", label = "Primary Dataset")
        cats_axis = hist.axis.IntCategory(range(2), name = "anacat", label = "Analysis Category")
        jetht_axis = hist.axis.Variable(manual_jetht_bins, name = "Jet_HT", label = r'$AK4\ Jet\ HT$') # Used for Trigger Analysis
        jetht_axis_zoom = hist.axis.Variable(manual_jetht_bins_zoom, name = "Jet_HT", label = r'$AK4\ Jet\ HT$') # Used for Trigger Analysis
        sdMass_axis = hist.axis.Variable(manual_sdMass_bins, name = "Jet_sdMass", label = r'$AK4\ M_{SD}$')
        
#    ===================================================================================================
#    TTTTTTT RRRRRR  IIIIIII GGGGGGG GGGGGGG EEEEEEE RRRRRR      H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
#       T    R     R    I    G       G       E       R     R     H     H    I     S         T     S          
#       T    R     R    I    G       G       E       R     R     H     H    I    S          T    S           
#       T    RRRRRR     I    G  GGGG G  GGGG EEEEEEE RRRRRR      HHHHHHH    I     SSSSS     T     SSSSS      
#       T    R   R      I    G     G G     G E       R   R       H     H    I          S    T          S     
#       T    R    R     I    G     G G     G E       R    R      H     H    I         S     T         S      
#       T    R     R IIIIIII  GGGGG   GGGGG  EEEEEEE R     R     H     H IIIIIII SSSSS      T    SSSSS
#    ===================================================================================================
        
        self.histo_dict = {
           'condition1_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'condition2_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'condition3_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'condition4_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'condition5_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'condition_denominator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
            
           'trigger1_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'trigger2_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'trigger3_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'trigger4_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'trigger5_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
            
           'condition1_numerator_zoom': hist.Hist(dataset_axis, cats_axis, jetht_axis_zoom, sdMass_axis, storage = "weight", name = "Counts"),
           'condition2_numerator_zoom': hist.Hist(dataset_axis, cats_axis, jetht_axis_zoom, sdMass_axis, storage = "weight", name = "Counts"),
           'condition3_numerator_zoom': hist.Hist(dataset_axis, cats_axis, jetht_axis_zoom, sdMass_axis, storage = "weight", name = "Counts"),
           'condition4_numerator_zoom': hist.Hist(dataset_axis, cats_axis, jetht_axis_zoom, sdMass_axis, storage = "weight", name = "Counts"),
           'condition5_numerator_zoom': hist.Hist(dataset_axis, cats_axis, jetht_axis_zoom, sdMass_axis, storage = "weight", name = "Counts"),
           'condition_denominator_zoom': hist.Hist(dataset_axis, cats_axis, jetht_axis_zoom, sdMass_axis, storage = "weight", name = "Counts"),
            
           'trigger1_numerator_zoom': hist.Hist(dataset_axis, cats_axis, jetht_axis_zoom, sdMass_axis, storage = "weight", name = "Counts"),
           'trigger2_numerator_zoom': hist.Hist(dataset_axis, cats_axis, jetht_axis_zoom, sdMass_axis, storage = "weight", name = "Counts"),
           'trigger3_numerator_zoom': hist.Hist(dataset_axis, cats_axis, jetht_axis_zoom, sdMass_axis, storage = "weight", name = "Counts"),
           'trigger4_numerator_zoom': hist.Hist(dataset_axis, cats_axis, jetht_axis_zoom, sdMass_axis, storage = "weight", name = "Counts"),
           'trigger5_numerator_zoom': hist.Hist(dataset_axis, cats_axis, jetht_axis_zoom, sdMass_axis, storage = "weight", name = "Counts"),
            
           'cutflow': processor.defaultdict_accumulator(int),
        }
        
            
    @property
    def accumulator(self):
        return self._accumulator
    
    def ConvertLabelToInt(self, mapping, str_label):
        for intkey, string in mapping.items():
            if str_label == string:
                return intkey

        return "The label has not been found :("

    def process(self, events):
        output = self.histo_dict
        
        # ---- Define dataset ---- #
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']

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
            "msoftdrop": events.FatJet_msoftdrop,
            "jetId": events.FatJet_jetId,
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
            # "PU": events.Pileup_nPU,
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
            
            
#    ================================================================
#    TTTTTTT RRRRRR  IIIIIII GGGGGGG GGGGGGG EEEEEEE RRRRRR    SSSSS     
#       T    R     R    I    G       G       E       R     R  S          
#       T    R     R    I    G       G       E       R     R S           
#       T    RRRRRR     I    G  GGGG G  GGGG EEEEEEE RRRRRR   SSSSS      
#       T    R   R      I    G     G G     G E       R   R         S     
#       T    R    R     I    G     G G     G E       R    R       S      
#       T    R     R IIIIIII  GGGGG   GGGGG  EEEEEEE R     R SSSSS  
#    ================================================================
        
#         Dataset_info = events.fields # All nanoaod events
#         listOfTriggers = np.array([name for name in Dataset_info if 'HLT' in name]) # Find event name info that have HLT to find all relevant triggers
        
#         isHLT_PF = np.array(['HLT_PF' in i for i in listOfTriggers])
#         isHLT_AK8 = np.array(['HLT_AK8' in i for i in listOfTriggers])
        
#         HLT_PF_triggers = listOfTriggers[isHLT_PF]
#         HLT_AK8_triggers = listOfTriggers[isHLT_AK8]
        
        # print(Jets.PU, '\n')
        trigDenom = events.HLT_Mu50 | events.HLT_IsoMu24
        # print(HLT_PF_triggers)
        # print(HLT_AK8_triggers)
        # print(events.HLT_IsoMu24)
        # print(trigDenom)
        # print('-----------------------------------------------')
        
        # is2017Bun = False # Switch to isolate 2017 B chunks
        
        trigger1 = None
        trigger2 = None
        trigger3 = None
        
        condition = None
        Triggers = []
        
        if self.year == 2016 and isData: 
            
            try:
                Triggers.append(events.HLT_PFHT900)
            except AttributeError as AE:
                print(f'\nTrigger HLT_PFHT900 not found in {filename}\nEvent run {events.run}\n', flush=True)
                return output
            try:
                Triggers.append(events.HLT_AK8PFJet450)
            except AttributeError as AE:
                print(f'\nTrigger HLT_AK8PFJet450 not found in {filename}\nEvent run {events.run}\n', flush=True)
                return output
            try:
                Triggers.append(events.HLT_AK8PFJet360_TrimMass30)
            except AttributeError as AE:
                print(f'\nTrigger HLT_AK8PFJet360_TrimMass30 not found in {filename}\nEvent run {events.run}\n', flush=True)
                return output
            
        elif self.year == 2017 and isData and 'Run2017B' not in filename: 
            
            try:
                Triggers.append(events.HLT_PFHT1050)
            except AttributeError as AE:
                print(f'\nTrigger HLT_PFHT1050 not found in {filename}\nEvent run {events.run}\n', flush=True)
                return output
            try:
                Triggers.append(events.HLT_AK8PFJet550)
            except AttributeError as AE:
                print(f'\nTrigger HLT_AK8PFJet550 not found in {filename}\nEvent run {events.run}\n', flush=True)
                return output
            try:
                Triggers.append(events.HLT_AK8PFHT800_TrimMass50 ) # This IS prescaled!!
            except AttributeError as AE:
                print(f'\nTrigger HLT_AK8PFHT800_TrimMass50 not found in {filename}\nEvent run {events.run}\n', flush=True)
                return output
            
        elif self.year == 2017 and isData and 'Run2017B' in filename: 
            
            try:
                Triggers.append(events.HLT_PFHT1050)
            except AttributeError as AE:
                print(f'\nTrigger HLT_PFHT1050 not found in {filename}\nEvent run {events.run}\n', flush=True)
                return output
            try:
                Triggers.append(events.HLT_AK8PFJet550)
            except AttributeError as AE:
                print(f'\nTrigger HLT_AK8PFJet550 not found in {filename}\nEvent run {events.run}\n', flush=True)
                return output
            
        else:
            
            try:
                Triggers.append(events.HLT_PFHT1050)
            except AttributeError as AE:
                print(f'\nTrigger HLT_PFHT1050 not found in {filename}\nEvent run {events.run}\n', flush=True)
                return output
            try:
                Triggers.append(events.HLT_AK8PFHT800_TrimMass50 )
            except AttributeError as AE:
                print(f'\nTrigger HLT_AK8PFHT800_TrimMass50 not found in {filename}\nEvent run {events.run}\n', flush=True)
                return output
            try:
                Triggers.append(events.HLT_AK8PFJet400_TrimMass30)
            except AttributeError as AE:
                print(f'\nTrigger HLT_AK8PFJet400_TrimMass30 not found in {filename}\nEvent run {events.run}\n', flush=True)
                return output
                
            
                
        # condition = ak.flatten(ak.any(Triggers, axis=0, keepdims=True))

        trigger1 = Triggers[0]
        trigger2 = Triggers[1]

        if 'Run2017B' not in filename:
            trigger3 = Triggers[2]
            
        Trigger1 = trigger1 & trigDenom
        Trigger2 = trigger2 & trigDenom
        
        if 'Run2017B' not in filename:
            Trigger3 = trigger3 & trigDenom
            

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
            evtweights = np.ones(len(FatJets)) # set all "data weights" to one
        else: # if Monte Carlo dataset is used...
            evtweights = events.Generator_weight
        # ---- Show all events ---- #
        output['cutflow']['all events'] += len(FatJets)
        
        # ---- Setup Trigger Analysis Conditions in higher scope ---- #
        condition1 = None
        condition2 = None
        condition3 = None
        
        # ---- Apply Trigger(s) ---- # 
        condition1 = trigger1 & trigDenom 
        condition2 = (trigger1 | trigger2) & trigDenom
        if 'Run2017B' not in filename:
            condition3 = ((trigger1 | trigger2) | trigger3) & trigDenom
        
            
        # ---- Jets that satisfy Jet ID ---- #
        jet_id = (FatJets.jetId > 0) # Loose jet ID
        FatJets = FatJets[jet_id]
        output['cutflow']['events with Loose Jet ID'] += len(FatJets)
        
        # ---- Apply pT Cut and Rapidity Window ---- #
        FatJets_rapidity = .5*np.log( (FatJets.p4.energy + FatJets.p4.pz)/(FatJets.p4.energy - FatJets.p4.pz) )
        jetkincut_index = (FatJets.pt > self.ak8PtMin) & (np.abs(FatJets_rapidity) < 2.4)
        FatJets = FatJets[ jetkincut_index ]
        output['cutflow']['events with pT,y Cut'] += len(FatJets)
        
        # ---- Find two AK8 Jets ---- #
        twoFatJetsKin = (ak.num(FatJets, axis=-1) >= 2)
        FatJets = FatJets[twoFatJetsKin]
        SubJets = SubJets[twoFatJetsKin]
        Jets = Jets[twoFatJetsKin] # this used to not be here
        
        Trigger1 = Trigger1[twoFatJetsKin]
        Trigger2 = Trigger2[twoFatJetsKin]
        if 'Run2017B' not in filename:
            Trigger3 = Trigger3[twoFatJetsKin]
        condition1 = condition1[twoFatJetsKin]
        condition2 = condition2[twoFatJetsKin]
        if 'Run2017B' not in filename:
            condition3 = condition3[twoFatJetsKin]
        
            
        trigDenom = trigDenom[twoFatJetsKin]
        evtweights = evtweights[twoFatJetsKin]
        output['cutflow']['events with two FatJets'] += len(FatJets)
        
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
        ttbarcands = ttbarcands[oneTTbar]
        FatJets = FatJets[oneTTbar]
        Jets = Jets[oneTTbar] # this used to not be here
        output['cutflow']['events with >= oneTTbar'] += len(FatJets)
        
        Trigger1 = Trigger1[oneTTbar]
        Trigger2 = Trigger2[oneTTbar]
        if 'Run2017B' not in filename:
            Trigger3 = Trigger3[oneTTbar]
        condition1 = condition1[oneTTbar]
        condition2 = condition2[oneTTbar]
        if 'Run2017B' not in filename:
            condition3 = condition3[oneTTbar]
        
        trigDenom = trigDenom[oneTTbar]
        SubJets = SubJets[oneTTbar]
        evtweights = evtweights[oneTTbar]
            
        # ---- Apply Delta Phi Cut for Back to Back Topology ---- #
        """ NOTE: Should find function for this; avoids 2pi problem """
        dPhiCut = np.abs(ttbarcands.slot0.p4.delta_phi(ttbarcands.slot1.p4)) > 2.1
        dPhiCut = ak.flatten(dPhiCut)
        ttbarcands = ttbarcands[dPhiCut]
        FatJets = FatJets[dPhiCut] 
        Jets = Jets[dPhiCut] # this used to not be here
        output['cutflow']['events with dPhi Cut'] += len(FatJets)
        
        Trigger1 = Trigger1[dPhiCut]
        Trigger2 = Trigger2[dPhiCut]
        if 'Run2017B' not in filename:
            Trigger3 = Trigger3[dPhiCut]
        condition1 = condition1[dPhiCut]
        condition2 = condition2[dPhiCut]
        if 'Run2017B' not in filename:
            condition3 = condition3[dPhiCut]
            
        trigDenom = trigDenom[dPhiCut]
        SubJets = SubJets[dPhiCut] 
        evtweights = evtweights[dPhiCut]
        
        # ---- Identify subjets according to subjet ID ---- #
        hasSubjets0 = ((ttbarcands.slot0.subJetIdx1 > -1) & (ttbarcands.slot0.subJetIdx2 > -1)) # 1st candidate has two subjets
        hasSubjets1 = ((ttbarcands.slot1.subJetIdx1 > -1) & (ttbarcands.slot1.subJetIdx2 > -1)) # 2nd candidate has two subjets
        GoodSubjets = ak.flatten(((hasSubjets0) & (hasSubjets1))) # Selection of 4 (leading) subjects
        FatJets = FatJets[GoodSubjets]
        output['cutflow']['events with Good Subjets'] += len(FatJets)
        ttbarcands = ttbarcands[GoodSubjets] # Choose only ttbar candidates with this selection of subjets
        SubJets = SubJets[GoodSubjets]
        Jets = Jets[GoodSubjets] # this used to not be here
        
        Trigger1 = Trigger1[GoodSubjets]
        Trigger2 = Trigger2[GoodSubjets]
        if 'Run2017B' not in filename:
            Trigger3 = Trigger3[GoodSubjets]
        condition1 = condition1[GoodSubjets]
        condition2 = condition2[GoodSubjets]
        if 'Run2017B' not in filename:
            condition3 = condition3[GoodSubjets]
            
        trigDenom = trigDenom[GoodSubjets]
        evtweights = evtweights[GoodSubjets]
        
        SubJet01 = SubJets[ttbarcands.slot0.subJetIdx1] # ttbarcandidate 0's first subjet 
        SubJet02 = SubJets[ttbarcands.slot0.subJetIdx2] # ttbarcandidate 0's second subjet
        SubJet11 = SubJets[ttbarcands.slot1.subJetIdx1] # ttbarcandidate 1's first subjet 
        SubJet12 = SubJets[ttbarcands.slot1.subJetIdx2] # ttbarcandidate 1's second subjet
        
        

#    ============================================================
#    TTTTTTT     TTTTTTT    A    GGGGGGG GGGGGGG EEEEEEE RRRRRR  
#       T           T      A A   G       G       E       R     R 
#       T           T     A   A  G       G       E       R     R 
#       T           T     AAAAA  G  GGGG G  GGGG EEEEEEE RRRRRR  
#       T           T    A     A G     G G     G E       R   R   
#       T           T    A     A G     G G     G E       R    R   
#       T           T    A     A  GGGGG   GGGGG  EEEEEEE R     R
#    ============================================================

        # ----------- CMS Top Tagger Version 2 (SD and Tau32 Cuts) ----------- #
        # ---- NOTE: Must Change This to DeepAK8 Top Tag Discriminator Cut ----#
        # ---- Maybe we should ignore tau32 cut(s) when performing trigger analysis ---- #
        
        mcut_s0 = (self.minMSD < ttbarcands.slot0.msoftdrop) & (ttbarcands.slot0.msoftdrop < self.maxMSD) 
        mcut_s1 = (self.minMSD < ttbarcands.slot1.msoftdrop) & (ttbarcands.slot1.msoftdrop < self.maxMSD) 
        
        ttag_s0 = mcut_s0
        ttag_s1 = mcut_s1
        
        # ---- Define "Top Tag" Regions ---- #
        ttag0 =   (~ttag_s0) & (~ttag_s1) # No tops tagged (0t) (will not store 2D hist values of SD within softdrop window)
        ttagI =   ttag_s0 | ttag_s1 # At least one top tagged ('I' for 'inclusive' tagger; >=1t; 1t+2t)
        ttagAny = ttag0 | ttagI # Any tag region (0, 1, or 2)
        
#    ===========================================================================================================================        
#    TTTTTTT RRRRRR  IIIIIII GGGGGGG GGGGGGG EEEEEEE RRRRRR         A    N     N    A    L       Y     Y   SSSSS IIIIIII   SSSSS     
#       T    R     R    I    G       G       E       R     R       A A   NN    N   A A   L        Y   Y   S         I     S          
#       T    R     R    I    G       G       E       R     R      A   A  N N   N  A   A  L         Y Y   S          I    S           
#       T    RRRRRR     I    G  GGGG G  GGGG EEEEEEE RRRRRR       AAAAA  N  N  N  AAAAA  L          Y     SSSSS     I     SSSSS      
#       T    R   R      I    G     G G     G E       R   R       A     A N   N N A     A L          Y          S    I          S     
#       T    R    R     I    G     G G     G E       R    R      A     A N    NN A     A L          Y         S     I         S      
#       T    R     R IIIIIII  GGGGG   GGGGG  EEEEEEE R     R     A     A N     N A     A LLLLLLL    Y    SSSSS   IIIIIII SSSSS
#    ===========================================================================================================================  
        
        TriggersDict = {
            '1': Trigger1,
            '2': Trigger2
        }   
    
        ConditionsDict = {
            '1': condition1,
            '2': condition2
        } 
        
        if 'Run2017B' not in filename:
            TriggersDict['3'] = Trigger3
            ConditionsDict['3'] = condition3
        
            

        # ---- Defining Jet Collections for Trigger Analysis Numerator and Denominator ---- #
        Jets_NumTrigger1 = Jets[Trigger1] # contains jets to be used as numerator for trigger eff
        Jets_NumTrigger2 = Jets[Trigger2]
        if 'Run2017B' not in filename:
            Jets_NumTrigger3 = Jets[Trigger3]
        Jets_NumCondition1 = Jets[condition1] # contains jets to be used as numerator for trigger eff
        Jets_NumCondition2 = Jets[condition2]
        if 'Run2017B' not in filename:
            Jets_NumCondition3 = Jets[condition3]
        
            
        Jets_DenomCondition = Jets[trigDenom] # contains jets to be used as denominator for trigger eff
        
        output['cutflow']['events with jets cond1'] +=  ak.sum(condition1, axis=-1)
        output['cutflow']['events with jets cond2'] +=  ak.sum(condition2, axis=-1)
        if 'Run2017B' not in filename:
            output['cutflow']['events with jets cond3'] +=  ak.sum(condition3, axis=-1)
        
        output['cutflow']['events with jets Denom cond'] +=  ak.sum(trigDenom, axis=-1)
        
        # ---- Must pass this cut before calculating HT variables for analysis ---- #
        passAK4_num1_trig = (Jets_NumTrigger1.pt > 30.) & (np.abs(Jets_NumTrigger1.eta) < 3.0) 
        passAK4_num2_trig = (Jets_NumTrigger2.pt > 30.) & (np.abs(Jets_NumTrigger2.eta) < 3.0)
        if 'Run2017B' not in filename:
            passAK4_num3_trig = (Jets_NumTrigger3.pt > 30.) & (np.abs(Jets_NumTrigger3.eta) < 3.0) 
        passAK4_num1 = (Jets_NumCondition1.pt > 30.) & (np.abs(Jets_NumCondition1.eta) < 3.0) 
        passAK4_num2 = (Jets_NumCondition2.pt > 30.) & (np.abs(Jets_NumCondition2.eta) < 3.0)
        if 'Run2017B' not in filename:
            passAK4_num3 = (Jets_NumCondition3.pt > 30.) & (np.abs(Jets_NumCondition3.eta) < 3.0)
        
        
        passAK4_denom = (Jets_DenomCondition.pt > 30.) & (np.abs(Jets_DenomCondition.eta) < 3.0) 
        
        # ---------------------------------------------------------------------------------------------#
        # ---- Remember to have weights array that is consistently the same size as num and denom ---- #
        # --------- This is only because input later in the code expects an array of weights --------- #
        # ------------- despite this array being simply an array of ones for the sake of ------------- #
        # ----------------------------------- not altering the data ---------------------------------- #
        # ---------------------------------------------------------------------------------------------#
        
        Num1Wgt_trig = evtweights[Trigger1]
        Num2Wgt_trig = evtweights[Trigger2]
        if 'Run2017B' not in filename:
            Num3Wgt_trig = evtweights[Trigger3]
        Num1Wgt = evtweights[condition1]
        Num2Wgt = evtweights[condition2]
        if 'Run2017B' not in filename:
            Num3Wgt = evtweights[condition3]
        
        
        NumWgtTrigDict = {
            '1': Num1Wgt_trig,
            '2': Num2Wgt_trig
        }
        NumWgtDict = {
            '1': Num1Wgt,
            '2': Num2Wgt
        }
        
        if 'Run2017B' not in filename:
            NumWgtTrigDict['3'] = Num3Wgt_trig
            NumWgtDict['3'] = Num3Wgt
            
        DenomWgt = evtweights[trigDenom]
        
        # ---- Defining Trigger Analysis Numerator(s) and Denominator as function of HT ---- #
        jet_HT_numerator1_trig = ak.sum(Jets_NumTrigger1[passAK4_num1_trig].pt, axis=-1) # Sum over each AK4 Jet per event
        jet_HT_numerator2_trig = ak.sum(Jets_NumTrigger2[passAK4_num2_trig].pt, axis=-1)
        if 'Run2017B' not in filename:
            jet_HT_numerator3_trig = ak.sum(Jets_NumTrigger3[passAK4_num3_trig].pt, axis=-1)
        jet_HT_numerator1 = ak.sum(Jets_NumCondition1[passAK4_num1].pt, axis=-1) # Sum over each AK4 Jet per event
        jet_HT_numerator2 = ak.sum(Jets_NumCondition2[passAK4_num2].pt, axis=-1)
        if 'Run2017B' not in filename:
            jet_HT_numerator3 = ak.sum(Jets_NumCondition3[passAK4_num3].pt, axis=-1)
        
        
        jet_HT_numeratorTrigDict = {
            '1': jet_HT_numerator1_trig,
            '2': jet_HT_numerator2_trig
        }
        
        jet_HT_numeratorDict = {
            '1': jet_HT_numerator1,
            '2': jet_HT_numerator2
        }
        if 'Run2017B' not in filename:
            jet_HT_numeratorTrigDict['3'] = jet_HT_numerator3_trig
            jet_HT_numeratorDict['3'] = jet_HT_numerator3
            
        jet_HT_denominator = ak.sum(Jets_DenomCondition[passAK4_denom].pt, axis=-1) # Sum over each AK4 Jet per event
        
        # ---- Defining Trigger Analysis Numerator(s) and Denominator as function of SD ---- #
        sdMass = ak.flatten(ttbarcands.slot0.msoftdrop)
        
        jet_SD_numerator1_trig = sdMass[Trigger1]
        jet_SD_numerator2_trig = sdMass[Trigger2]
        if 'Run2017B' not in filename:
            jet_SD_numerator3_trig = sdMass[Trigger3]
        jet_SD_numerator1 = sdMass[condition1]
        jet_SD_numerator2 = sdMass[condition2]
        if 'Run2017B' not in filename:
            jet_SD_numerator3 = sdMass[condition3]
        
            
        jet_SD_denominator = sdMass[trigDenom]
        
        jet_SD_numeratorTrigDict = {
            '1': jet_SD_numerator1_trig,
            '2': jet_SD_numerator2_trig
        }
        
        jet_SD_numeratorDict = {
            '1': jet_SD_numerator1,
            '2': jet_SD_numerator2
        }
        if 'Run2017B' not in filename:
            jet_SD_numeratorTrigDict['3'] = jet_SD_numerator3_trig
            jet_SD_numeratorDict['3'] = jet_SD_numerator3
                
        # ----------------- Keep track of cutflow for individual bins ---------------- #
        # ---- [200, 800, 840, 880, 920, 960, 1000, 1200, 1400, 1600, 1800, 2000] ---- #
        # num1_inBin1 = (200. < jet_HT_numerator1) & (jet_HT_numerator1 < 800.)
        # num2_inBin1 = (200. < jet_HT_numerator2) & (jet_HT_numerator2 < 800.)
        # num3_inBin1 = (200. < jet_HT_numerator3) & (jet_HT_numerator3 < 800.)
        # num4_inBin1 = (200. < jet_HT_numerator4) & (jet_HT_numerator4 < 800.)
        # denom_inBin1 = (200. < jet_HT_denominator) & (jet_HT_denominator < 800.)
        # num1_inBin11 = (1800. < jet_HT_numerator1) & (jet_HT_numerator1 < 2000.)
        # num2_inBin11 = (1800. < jet_HT_numerator2) & (jet_HT_numerator2 < 2000.)
        # num3_inBin11 = (1800. < jet_HT_numerator3) & (jet_HT_numerator3 < 2000.)
        # num4_inBin11 = (1800. < jet_HT_numerator4) & (jet_HT_numerator4 < 2000.)
        # denom_inBin11 = (1800. < jet_HT_denominator) & (jet_HT_denominator < 2000.)
        # output['cutflow']['numerator 1 in bin [200, 800]'] += ak.to_awkward0(num1_inBin1).sum()
        # output['cutflow']['numerator 2 in bin [200, 800]'] += ak.to_awkward0(num2_inBin1).sum()
        # output['cutflow']['numerator 3 in bin [200, 800]'] += ak.to_awkward0(num3_inBin1).sum()
        # output['cutflow']['numerator 4 in bin [200, 800]'] += ak.to_awkward0(num4_inBin1).sum()
        # output['cutflow']['denominator in bin [200, 800]'] += ak.to_awkward0(denom_inBin1).sum()
        # output['cutflow']['numerator 1 in bin [1800, 2000]'] += ak.to_awkward0(num1_inBin11).sum()
        # output['cutflow']['numerator 2 in bin [1800, 2000]'] += ak.to_awkward0(num2_inBin11).sum()
        # output['cutflow']['numerator 3 in bin [1800, 2000]'] += ak.to_awkward0(num3_inBin11).sum()
        # output['cutflow']['numerator 4 in bin [1800, 2000]'] += ak.to_awkward0(num4_inBin11).sum()
        # output['cutflow']['denominator in bin [1800, 2000]'] += ak.to_awkward0(denom_inBin11).sum()
        
        # ---- Define Categories for Trigger Analysis Denominator and Fill Hists ---- #
        ttags = [ttagAny[trigDenom],ttagI[trigDenom]]
        cats = [ ak.to_awkward0(ak.flatten(t)) for t in ttags ]
        labels_and_categories = dict(zip(self.ttagcats_forTriggerAnalysis, cats))
        for ilabel,icat in labels_and_categories.items():
            output['condition_denominator'].fill(dataset = dataset, anacat = self.ConvertLabelToInt(self.label_dict, ilabel), 
                                                Jet_HT = ak.to_numpy(jet_HT_denominator[icat]),
                                                Jet_sdMass = ak.to_numpy(jet_SD_denominator[icat]),
                                                weight = ak.to_numpy(DenomWgt[icat]))
            output['condition_denominator_zoom'].fill(dataset = dataset, anacat = self.ConvertLabelToInt(self.label_dict, ilabel), 
                                                Jet_HT = ak.to_numpy(jet_HT_denominator[icat]),
                                                Jet_sdMass = ak.to_numpy(jet_SD_denominator[icat]),
                                                weight = ak.to_numpy(DenomWgt[icat]))
        # ---- Define Categories for Trigger Analysis Numerators and Fill Hists---- #
        for i in range(1, len(ConditionsDict)+1):
            c = ConditionsDict[str(i)]
            ct = TriggersDict[str(i)]
            n_HT = jet_HT_numeratorDict[str(i)]
            n_SD = jet_SD_numeratorDict[str(i)]
            nt_HT = jet_HT_numeratorTrigDict[str(i)]
            nt_SD = jet_SD_numeratorTrigDict[str(i)]
            w = NumWgtDict[str(i)]
            wt = NumWgtTrigDict[str(i)]
            ttags_cond = [ttagAny[c],ttagI[c]]
            ttags_trig = [ttagAny[ct],ttagI[ct]]
            cats_cond = [ ak.to_awkward0(ak.flatten(t)) for t in ttags_cond]
            cats_trig = [ ak.to_awkward0(ak.flatten(t)) for t in ttags_trig]
            labels_and_categories_cond = dict(zip(self.ttagcats_forTriggerAnalysis, cats_cond))
            labels_and_categories_trig = dict(zip(self.ttagcats_forTriggerAnalysis, cats_trig))
            # print(labels_and_categories_cond)
            for ilabel,icat in labels_and_categories_cond.items(): 
                output['condition' + str(i) + '_numerator'].fill(dataset = dataset, anacat = self.ConvertLabelToInt(self.label_dict, ilabel), 
                                                                Jet_HT = ak.to_numpy(n_HT[icat]),
                                                                Jet_sdMass = ak.to_numpy(n_SD[icat]),
                                                                weight = ak.to_numpy(w[icat]))
                output['condition' + str(i) + '_numerator_zoom'].fill(dataset = dataset, anacat = self.ConvertLabelToInt(self.label_dict, ilabel), 
                                                                Jet_HT = ak.to_numpy(n_HT[icat]),
                                                                Jet_sdMass = ak.to_numpy(n_SD[icat]),
                                                                weight = ak.to_numpy(w[icat]))
            for ilabel,icat in labels_and_categories_trig.items():
                output['trigger' + str(i) + '_numerator'].fill(dataset = dataset, anacat = self.ConvertLabelToInt(self.label_dict, ilabel), 
                                                                Jet_HT = ak.to_numpy(nt_HT[icat]),
                                                                Jet_sdMass = ak.to_numpy(nt_SD[icat]),
                                                                weight = ak.to_numpy(wt[icat]))
                output['trigger' + str(i) + '_numerator_zoom'].fill(dataset = dataset, anacat = self.ConvertLabelToInt(self.label_dict, ilabel), 
                                                                Jet_HT = ak.to_numpy(nt_HT[icat]),
                                                                Jet_sdMass = ak.to_numpy(nt_SD[icat]),
                                                                weight = ak.to_numpy(wt[icat]))
 
        return output

    def postprocess(self, accumulator):
        return accumulator

    
    
    