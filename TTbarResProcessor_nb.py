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
manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]

"""@TTbarResAnaHadronic Package to perform the data-driven mistag-rate-based ttbar hadronic analysis. 
"""
class TTbarResProcessor(processor.ProcessorABC):
    def __init__(self, prng=RandomState(1234567890), htCut=950., minMSD=105., maxMSD=210.,
                 tau32Cut=0.65, ak8PtMin=400., bdisc=0.8484,
                 writePredDist=True,isData=True,year=2019, UseLookUpTables=False, lu=None, 
                 ModMass=False, RandomDebugMode=False):
        
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
        self.lu = lu # Look Up Tables
        
        self.ttagcats = ["Probet", "at", "pret", "0t", "1t", "1t+2t", "2t", "0t+1t+2t"] #anti-tag+probe, anti-tag, pre-tag, 0, 1, >=1, 2 ttags, any t-tag
        self.btagcats = ["0b", "1b", "2b"]   # 0, 1, >=2 btags
        self.ycats = ['cen', 'fwd']          # Central and forward
        # Combine categories like "0bcen", "0bfwd", etc:
        self.anacats = [ t+b+y for t,b,y in itertools.product( self.ttagcats, self.btagcats, self.ycats) ]
        #print(self.anacats)
        
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        cats_axis = hist.Cat("anacat", "Analysis Category")
        
        jetmass_axis = hist.Bin("jetmass", r"Jet $m$ [GeV]", 50, 0, 500)
        jetpt_axis = hist.Bin("jetpt", r"Jet $p_{T}$ [GeV]", 50, 0, 5000)
        ttbarmass_axis = hist.Bin("ttbarmass", r"$m_{t\bar{t}}$ [GeV]", 50, 0, 5000)
        jeteta_axis = hist.Bin("jeteta", r"Jet $\eta$", 50, -5, 5)
        jetphi_axis = hist.Bin("jetphi", r"Jet $\phi$", 50, -np.pi, np.pi)
        jety_axis = hist.Bin("jety", r"Jet $y$", 50, -3, 3)
        jetdy_axis = hist.Bin("jetdy", r"Jet $\Delta y$", 50, 0, 5)
        manual_axis = hist.Bin("jetp", r"Jet Momentum [GeV]", manual_bins)
        tagger_axis = hist.Bin("tagger", r"deepTag", 50, 0, 1)
        tau32_axis = hist.Bin("tau32", r"$\tau_3/\tau_2$", 50, 0, 2)
        
        subjetmass_axis = hist.Bin("subjetmass", r"SubJet $m$ [GeV]", 50, 0, 500)
        subjetpt_axis = hist.Bin("subjetpt", r"SubJet $p_{T}$ [GeV]", 50, 0, 2000)
        subjeteta_axis = hist.Bin("subjeteta", r"SubJet $\eta$", 50, -4, 4)
        subjetphi_axis = hist.Bin("subjetphi", r"SubJet $\phi$", 50, -np.pi, np.pi)

        self._accumulator = processor.dict_accumulator({
            'ttbarmass': hist.Hist("Counts", dataset_axis, cats_axis, ttbarmass_axis),
            
            'jetmass':         hist.Hist("Counts", dataset_axis, cats_axis, jetmass_axis),
            'SDmass':          hist.Hist("Counts", dataset_axis, cats_axis, jetmass_axis),
            'SDmass_precat':   hist.Hist("Counts", dataset_axis, jetpt_axis, jetmass_axis),
            
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
            
            'subjetmass':   hist.Hist("Counts", dataset_axis, cats_axis, subjetmass_axis),
            'subjetpt':     hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
            'subjeteta':    hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
            'subjetphi':    hist.Hist("Counts", dataset_axis, cats_axis, subjetphi_axis),
            
            'numerator':   hist.Hist("Counts", dataset_axis, cats_axis, manual_axis),
            'denominator': hist.Hist("Counts", dataset_axis, cats_axis, manual_axis),
            
            'cutflow': processor.defaultdict_accumulator(int),
            
        })

            
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
       
        SubJet01 = SubJets[ttbarcands.slot0.subJetIdx1] # 1st candidate's subjet 1
        SubJet02 = SubJets[ttbarcands.slot0.subJetIdx2] # 1st candidate's subjet 2
        SubJet11 = SubJets[ttbarcands.slot1.subJetIdx1] # 2nd candidate's subjet 1
        SubJet12 = SubJets[ttbarcands.slot1.subJetIdx2] # 2nd candidate's subjet 2
        
        # ---- Define Rapidity Regions ---- #
        """ NOTE that ttbarcands.i0.p4.energy no longer works after ttbarcands is defined as an old awkward array """
        #i0_p = ttbarcands.i0.pt*np.cosh( ttbarcands.i0.eta ) # 3-momentum magnitude
        #i1_p = ttbarcands.i1.pt*np.cosh( ttbarcands.i1.eta ) # 3-momentum magnitude
        #i0_energy = np.sqrt( ttbarcands.i0.mass**2 + i0_p**2 )
        #i1_energy = np.sqrt( ttbarcands.i1.mass**2 + i1_p**2 )
        #i0_pz = ttbarcands.i0.pt*np.sinh( ttbarcands.i0.eta )
        #i1_pz = ttbarcands.i1.pt*np.sinh( ttbarcands.i1.eta )
        s0_energy = ttbarcands.slot0.p4.energy
        s1_energy = ttbarcands.slot1.p4.energy
        s0_pz = ttbarcands.slot0.p4.pz
        s1_pz = ttbarcands.slot1.p4.pz
        ttbarcands_s0_rapidity = 0.5*np.log( (s0_energy+s0_pz)/(s0_energy-s0_pz) ) # rapidity as function of eta
        ttbarcands_s1_rapidity = 0.5*np.log( (s1_energy+s1_pz)/(s1_energy-s1_pz) ) # rapidity as function of eta
        cen = np.abs(ttbarcands_s0_rapidity - ttbarcands_s1_rapidity) < 1.0
        fwd = (~cen)
        
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
        
        # ---- Pick FatJet that passes btag cut based on its subjet with the highest btag value ---- #
        btag_s0 = ( np.maximum(SubJet01.btagCSVV2 , SubJet02.btagCSVV2) > self.bdisc )
        btag_s1 = ( np.maximum(SubJet11.btagCSVV2 , SubJet12.btagCSVV2) > self.bdisc )
        
        # --- Define "B Tag" Regions ---- #
        btag0 = (~btag_s0) & (~btag_s1) #(0b)
        btag1 = btag_s0 ^ btag_s1 #(1b)
        btag2 = btag_s0 & btag_s1 #(2b)
        
        # ---- Probabilities of finding/not finding a bjet ---- #
        Prob_yes_s0, Prob_yes_s1 = ttbarcands.slot0.btagCSVV2, ttbarcands.slot1.btagCSVV2
        Prob_no_s0, Prob_no_s1 = (1.-Prob_yes_s0), (1.-Prob_yes_s1) 
        
        Prob_btag0 = Prob_no_s0 * Prob_no_s1 # P(0 tags | 2 jets)
        Prob_btag1 = (Prob_yes_s0*Prob_no_s1) + (Prob_no_s0*Prob_yes_s1) # P(1 tag | 2 jets)
        Prob_btag2 = Prob_yes_s0 * Prob_yes_s1 # P(2 tags | 2 jets)
        
        # ---- Probabilities weighted by Scale Factors ---- #
        
                    # -- Subjets to use for extracting scale factors -- #
        SubJet_s0 = np.where( np.maximum(SubJet01.btagCSVV2 , SubJet02.btagCSVV2) == SubJet01.btagCSVV2, SubJet01, SubJet02 )
        SubJet_s1 = np.where( np.maximum(SubJet11.btagCSVV2 , SubJet12.btagCSVV2) == SubJet11.btagCSVV2, SubJet11, SubJet12 )
        
                    # -- Scale Factors -- #
        btag_sf = BTagScaleFactor("DeepCSV_106XUL17SF_V2.csv", "tight")
        BSF_s0 = btag_sf.eval("central", SubJet_s0.hadronFlavour, abs(SubJet_s0.eta), SubJet_s0.pt, ignore_missing=True)
        BSF_s1 = btag_sf.eval("central", SubJet_s1.hadronFlavour, abs(SubJet_s1.eta), SubJet_s1.pt, ignore_missing=True)
        
                    # -- Re-Define Probs with BSF's -- #
        Prob_yes_s0_sf, Prob_yes_s1_sf = (Prob_yes_s0*BSF_s0), (Prob_yes_s1*BSF_s1)
        Prob_no_s0_sf, Prob_no_s1_sf = (Prob_no_s0*BSF_s0), (Prob_no_s1*BSF_s1)
        
        Prob_btag0_sf = Prob_no_s0_sf * Prob_no_s1_sf # P(0 tags | 2 jets)
        Prob_btag1_sf = (Prob_yes_s0_sf*Prob_no_s1_sf) + (Prob_no_s0_sf*Prob_yes_s1_sf) # P(1 tag | 2 jets)
        Prob_btag2_sf = Prob_yes_s0_sf * Prob_yes_s1_sf # P(2 tags | 2 jets)
        
        print("SF0:", BSF_s0)
        print("\nType:", type(BSF_s0))
        print("\nLength = ", len(BSF_s0))
        print()
        print("SF1:", BSF_s1)
        print("\nType:", type(BSF_s1))
        print("\nLength = ", len(BSF_s1))
        print()
        print("Weights", evtweights)
        print("\nType:", type(evtweights))
        print("\nLength = ", len(evtweights))
        print()
        print('_____________________________________________________________________')
        print()
        
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
        
        return output

    def postprocess(self, accumulator):
        return accumulator

