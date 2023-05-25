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
from coffea.analysis_tools import Weights, PackedSelection
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

import awkward as ak




# get corrections

# for dask, `from corrections.corrections import` does not work
sys.path.append(os.getcwd()+'/corrections/')
from corrections import (    
    GetFlavorEfficiency,
    HEMCleaning,
    GetJECUncertainties,
    GetPDFWeights,
    GetPUSF,
    getLumiMaskRun2,
    getMETFilter,
    pTReweighting,
)
from btagCorrections import btagCorrections
from functions import calcRapidity



#ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)


# --- Define 'Manual bins' to use for mistag plots for aesthetic purposes--- #
manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]

# --- Define 'Manual pT bins' to use for mc flavor efficiency plots for higher stats per bin--- # 
manual_subjetpt_bins = [0, 300, 600, 1200] # Used on 6/17/22 for ttbar (3 bins)
manual_subjeteta_bins = [0., 0.6, 1.2, 2.4] # Used on 6/17/22 for ttbar (3 bins)
manual_jetht_bins = [200, 800, 840, 880, 920, 960, 1000, 1200, 1400, 1600, 1800, 2000]
manual_sdMass_bins = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

"""Package to perform the data-driven mistag-rate-based ttbar hadronic analysis. """
class TTbarResProcessor(processor.ProcessorABC):
    def __init__(self, htCut=950., ak8PtMin=400.,
                 minMSD=105., maxMSD=210.,
                 tau32Cut=0.65,
                 bdisc=0.5847,
                 deepAK8Cut=0.435, useDeepAK8=False,
                 iov='2016APV',
                 bkgEst=False,
                ):
                 
        self.iov = iov
        self.htCut = htCut
        self.minMSD = minMSD
        self.maxMSD = maxMSD
        self.tau32Cut = tau32Cut
        self.ak8PtMin = ak8PtMin
        self.bdisc = bdisc
        self.deepAK8Cut = deepAK8Cut
        self.useDeepAK8 = useDeepAK8
        self.means_stddevs = defaultdict()
        self.bkgEst = bkgEst
        
        
        
        # analysis categories #
        self.ttagcats = ["AT&Pt", "at", "pret", "0t", "1t", ">=1t", "2t", ">=0t"] 
        self.btagcats = ["0b", "1b", "2b"]
        self.ycats = ['cen', 'fwd']
        
        self.anacats = [ t+b+y for t,b,y in itertools.product( self.ttagcats, self.btagcats, self.ycats) ]
        self.label_dict = {i: label for i, label in enumerate(self.anacats)}
        self.label_to_int_dict = {label: i for i, label in enumerate(self.anacats)}
        
        
        # axes
        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary Dataset")
        ttbarmass_axis = hist.axis.Regular(50, 800, 8000, name="ttbarmass", label=r"$m_{t\bar{t}}$ [GeV]")
        jetmass_axis   = hist.axis.Regular(50, 0, 500, name="jetmass", label=r"Jet $m$ [GeV]")
        jetpt_axis     = hist.axis.Regular(50, 400, 2000, name="jetpt", label=r"Jet $p_{T}$ [GeV]")
        jeteta_axis    = hist.axis.Regular(50, -2.4, 2.4, name="jeteta", label=r"Jet $\eta$")
        jetphi_axis    = hist.axis.Regular(50, -np.pi, np.pi, name="jetphi", label=r"Jet $\phi$")
        cats_axis = hist.axis.IntCategory(range(48), name="anacat", label="Analysis Category")
        manual_axis = hist.axis.Variable(manual_bins, name="jetp", label=r"Jet Momentum [GeV]")

        # output
        self.histo_dict = {

            
            # histograms
            'ttbarmass'  : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'numerator'  : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            'denominator': hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            'jetmass' : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            'jetpt'  : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            'jeteta'  : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            'jetphi'  : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
                        
            # accumulators
            'cutflow': processor.defaultdict_accumulator(int),

        }
        
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
                
        output = self.histo_dict        
        
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']
        
        isData = ('JetHT' in dataset) or ('SingleMu' in dataset)
            
        # Remove events with large weights
        if "QCD_Pt-15to7000" in filename: 
            events = events[ events.Generator.binvar > 400 ] 
        
        # lumi mask #
        if (isData):
            lumimasks = getLumiMaskRun2()
            lumi_mask = np.array(lumimasks[self.iov](events.run, events.luminosityBlock), dtype=bool)
            events = events[lumi_mask]
        elif ('TTbar' in dataset) or ('QCD' in dataset) : 
            if dataset not in self.means_stddevs : 
                average = np.average( events.genWeight )
                stddev = np.std( events.genWeight )
                self.means_stddevs[dataset] = (average, stddev)            
            average,stddev = self.means_stddevs[dataset]
            vals = (events.genWeight - average ) / stddev
            events = events[(np.abs(vals) < 2)]
        
        
        
        
        # blinding #
        if isData and (('2017' in self.iov) or ('2018' in self.iov)):
            events = events[::10]
            

        
        # event selection #
        selection = PackedSelection()
        

        # trigger cut #
#         if isData:
            
#             Triggers = { 
            
#             "2016APV": ["PFHT900"],
#             "2016" : ["PFHT900"],
#             "2017" : ["PFHT1050"],
#             "2018" : ["PFHT1050"],
        
#             }
                
            
# #             print(events.HLT["PFHT900"])
                
# #             for f in events.HLT.fields:
# #                 print(f)
            
#             triggers = [events.HLT[trig] for trig in Triggers[self.iov]]
            
            
#             selection.add('trigger', ak.flatten(ak.any(Triggers, axis=0, keepdims=True)))
#             deltriggers
            
            
#             Triggers = { 
            
#             "2016APV": ["HLT_PFHT900"],
#             "2016" : ["HLT_PFHT900"],
#             "2017" : ["HLT_PFHT1050"],
#             "2018" : ["HLT_PFHT1050"],
        
#             }
                
                        
        
                
        
        # objects #
        
        FatJets = events.FatJet
        SubJets = events.SubJet
        Jets    = events.Jet

        FatJets["p4"] = ak.with_name(FatJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        SubJets["p4"] = ak.with_name(SubJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        Jets["p4"]    = ak.with_name(Jets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")

        if not isData:
            GenJets = events.GenJet
            GenJets["p4"] = ak.with_name(GenJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                    
        
        # ---- Get event weights from dataset ---- #
        if isData:
            evtweights = np.ones( len(FatJets) ) 
        else: 
            if "LHEWeight_originalXWGTUP" not in events.fields: 
                evtweights = events.genWeight
            else: 
                evtweights = events.LHEWeight_originalXWGTUP
        
        output['cutflow']['all events'] += len(FatJets)
        output['cutflow']['sumw'] += np.sum(evtweights)
        output['cutflow']['sumw2'] += np.sum(evtweights**2)
        
              
        
        # ---- event selection and object selection ---- #
        

        
        # ht cut #
        selection.add('htCut',
            ak.sum(Jets.pt, axis=1) > self.htCut
        )

        # met filters #
        if isData:
            selection.add('metfilter', getMETFilter(self.iov, events))
                
        # jet id #
        selection.add('jetid', ak.any((FatJets.jetId > 0), axis=1))
        FatJets = FatJets[FatJets.jetId > 0]
                
        # jet kinematics # 
#         jetkincut = (FatJets.pt > self.ak8PtMin) & (np.abs(calcRapidity(FatJets.p4)) < 2.4)
        
        FatJets_rapidity = .5*np.log( (FatJets.p4.energy + FatJets.p4.pz)/(FatJets.p4.energy - FatJets.p4.pz) )
        jetkincut = (FatJets.pt > self.ak8PtMin) & (np.abs(FatJets_rapidity) < 2.4)
        
        selection.add('jetkincut', ak.any(jetkincut, axis=1))
        FatJets = FatJets[jetkincut]
        del jetkincut
        
        
        # at least 2 ak8 jets #
        selection.add('twoFatJets', (ak.num(FatJets) >= 2))
        
#         # ak8 jet has 2 subjets # **NEW**
#         hasSubjets = ((FatJets.subJetIdx1 > -1) & (FatJets.subJetIdx2 > -1))
#         FatJets = FatJets[hasSubjets]
#         selection.add('hasSubjets', ak.any(hasSubjets, axis=1))

#         print('jetkincut', len(FatJets))
        
        

        # cutflow #
        cuts = []
        for cut in selection.names:
            cuts.append(cut)
            output['cutflow'][cut] += len(FatJets[selection.all(*cuts)])
        del cuts


        # event cuts #
        eventCut = selection.all(*selection.names)
        FatJets = FatJets[eventCut]
        SubJets = SubJets[eventCut]
        Jets    = Jets[eventCut]
        evtweights = evtweights[eventCut]
        if not isData: GenJets = GenJets[eventCut]
            
            

        # ---- ttbar candidates ---- #
        
        # index = [[0], [1], [0], ... [0], [1], [1]] type='{# events} * var * int64'
        index = ak.unflatten( np.random.RandomState(1234567890).randint(2, size=len(FatJets)), np.ones(len(FatJets), dtype='i'))
        
        jet0 = FatJets[index]
        jet1 = FatJets[1 - index]        
        ttbarcands = ak.cartesian([jet0, jet1])
        del index
        
        
        
        # ttbar event cuts  #
        
        # at least 1 ttbar candidate #
        oneTTbar = (ak.num(ttbarcands) >= 1)
        
        # ---- Apply Delta Phi Cut for Back to Back Topology ---- #
        dPhiCut = ak.flatten(np.abs(ttbarcands.slot0.p4.delta_phi(ttbarcands.slot1.p4)) > 2.1)  
        
        
        # ttbar candidates have 2 subjets #
        hasSubjets0 = ((ttbarcands.slot0.subJetIdx1 > -1) & (ttbarcands.slot0.subJetIdx2 > -1))
        hasSubjets1 = ((ttbarcands.slot1.subJetIdx1 > -1) & (ttbarcands.slot1.subJetIdx2 > -1))
        GoodSubjets = ak.flatten(((hasSubjets0) & (hasSubjets1)))
        
        # apply ttbar event cuts #
        output['cutflow']['oneTTbar'] += len(FatJets[oneTTbar])
        output['cutflow']['dPhiCut'] += len(FatJets[(oneTTbar & dPhiCut)])
        output['cutflow']['Good Subjets'] += len(FatJets[(oneTTbar & dPhiCut & GoodSubjets)])

        ttbarcandCuts = (oneTTbar & dPhiCut & GoodSubjets)
        ttbarcands = ttbarcands[ttbarcandCuts]
        FatJets = FatJets[ttbarcandCuts]
        Jets = Jets[ttbarcandCuts]
        SubJets = SubJets[ttbarcandCuts]
        events = events[ttbarcandCuts]
        evtweights = evtweights[ttbarcandCuts]
        if not isData: GenJets = GenJets[ttbarcandCuts]
        del oneTTbar, dPhiCut, ttbarcandCuts, hasSubjets0, hasSubjets1, GoodSubjets
        
        
        
        
        # ttbarmass
        
        ttbarmass = (ttbarcands.slot0.p4 + ttbarcands.slot1.p4).mass
        
        # subjets
           
        SubJet00 = SubJets[ttbarcands.slot0.subJetIdx1]
        SubJet01 = SubJets[ttbarcands.slot0.subJetIdx2]
        SubJet10 = SubJets[ttbarcands.slot1.subJetIdx1]
        SubJet11 = SubJets[ttbarcands.slot1.subJetIdx2]

        # top tagger #
        
        
        
        # ----------- DeepAK8 Tagger (Discriminator Cut) ----------- #
        if self.useDeepAK8:
            ttag_s0 = ttbarcands.slot0.deepTagMD_TvsQCD > self.deepAK8Cut
            ttag_s1 = ttbarcands.slot1.deepTagMD_TvsQCD > self.deepAK8Cut
            antitag = ttbarcands.slot0.deepTagMD_TvsQCD < self.deepAK8Cut 

            
        # ----------- CMS Top Tagger Version 2 (SD and Tau32 Cuts) ----------- #
        else:
            tau32_s0 = np.where(ttbarcands.slot0.tau2>0,ttbarcands.slot0.tau3/ttbarcands.slot0.tau2, 0 )
            tau32_s1 = np.where(ttbarcands.slot1.tau2>0,ttbarcands.slot1.tau3/ttbarcands.slot1.tau2, 0 )

            taucut_s0 = tau32_s0 < self.tau32Cut
            taucut_s1 = tau32_s1 < self.tau32Cut

            mcut_s0 = (self.minMSD < ttbarcands.slot0.msoftdrop) & (ttbarcands.slot0.msoftdrop < self.maxMSD) 
            mcut_s1 = (self.minMSD < ttbarcands.slot1.msoftdrop) & (ttbarcands.slot1.msoftdrop < self.maxMSD) 

            ttag_s0 = (taucut_s0) & (mcut_s0)
            ttag_s1 = (taucut_s1) & (mcut_s1)
            antitag = (~taucut_s0) & (mcut_s0) # The Probe jet will always be ttbarcands.slot1 (at)


        
        # ---- Define "Top Tag" Regions ---- #
        antitag_probe = np.logical_and(antitag, ttag_s1) # Found an antitag and ttagged probe pair for mistag rate (AT&Pt)
        pretag =  ttag_s0 # Only jet0 (pret)
        ttag0 =   (~ttag_s0) & (~ttag_s1) # No tops tagged (0t)
        ttag1 =   ttag_s0 ^ ttag_s1 # Exclusively one top tagged (1t)
        ttagI =   ttag_s0 | ttag_s1 # At least one top tagged ('I' for 'inclusive' tagger; >=1t; 1t+2t)
        ttag2 =   ttag_s0 & ttag_s1 # Both jets top tagged (2t)
        Alltags = ttag0 | ttagI #Either no tag or at least one tag (0t+1t+2t)
        
        
        
        # b tagger #
        
        btag_s0 = ( np.maximum(SubJet00.btagDeepB , SubJet01.btagDeepB) > self.bdisc )
        btag_s1 = ( np.maximum(SubJet10.btagDeepB , SubJet11.btagDeepB) > self.bdisc )
        
        # --- Define "B Tag" Regions ---- #
        btag0 = (~btag_s0) & (~btag_s1) #(0b)
        btag1 = btag_s0 ^ btag_s1 #(1b)
        btag2 = btag_s0 & btag_s1 #(2b)
        
        
        
#         if (self.bkgEst):
#             btag0, btag1, btag2 = btagCorrections([btag0, btag1, btag2], 
#                                                   [SubJet00, SubJet01, SubJet10, SubJet11], 
#                                                   isData, 
#                                                   self.bdisc,
#                                                   sysType='central')
        
        
        
        # rapidity #
        cen = np.abs(calcRapidity(ttbarcands.slot0.p4) - calcRapidity(ttbarcands.slot1.p4)) < 1.0
        fwd = (~cen)
        
        
        
        # analysis category mask #
        
        regs = [cen,fwd]
        btags = [btag0,btag1,btag2]
        ttags = [antitag_probe,antitag,pretag,ttag0,ttag1,ttagI,ttag2,Alltags]
        
        cats = [ (t&b&y) for t,b,y in itertools.product(ttags, btags, regs) ]
        labels_and_categories = dict(zip( self.anacats, cats ))
        
        
        jetmass = ttbarcands.slot1.p4.mass
        jetp = ttbarcands.slot1.p4.p
                        
        # if running background estimation
        if (self.bkgEst):
            
            
            
            # for mistag rate weights
            mistag_rate_df = pd.read_csv(f'mistag/mistag_rate_{self.iov}.csv')
            pbins = mistag_rate_df['jetp bins'].values
            mistag_weights = np.ones(len(evtweights), dtype=float)
            
            
            # for mass modification
            qcdfile = util.load(f'outputs/QCD_{self.iov}.coffea')

            
            for ilabel,icat in labels_and_categories.items():
                
                
                icat = ak.flatten(icat)

                # ilabel[-5:] = bcat + ycat (0bcen for example) 

                # get mistag rate for antitag region
                mistag_rate = mistag_rate_df['at' + ilabel[-5:]].values

                # get p bin for probe jet p
                mistag_pbin = np.digitize(ak.flatten(jetp[icat]), pbins) - 1


                # store mistag weights for events in this category
                mistag_weights[icat] = mistag_rate[mistag_pbin]


                # mass modification procedure

                # get distribution of jet mass in QCD signal ('2t') region
                qcd_jetmass_counts = qcdfile['jetmass'][{'dataset':sum, 'anacat':self.label_to_int_dict['2t' + ilabel[-5:]]}].values()
                qcd_jetmass_bins = qcdfile['jetmass'].axes['jetmass'].centers - qcdfile['jetmass'].axes['jetmass'].widths/2


                # randomly select jet mass from distribution
                ModMass_hist_dist = ss.rv_histogram([qcd_jetmass_counts[:-1], qcd_jetmass_bins])
#                 jet1_modp4 = copy.copy(ttbarcands.slot1.p4)
#                 jet1_modp4["fMass"] = ModMass_hist_dist.rvs(size=len(jet1_modp4))
                ttbarcands.slot1.p4[icat]["fMass"] = ModMass_hist_dist.rvs(size=len(ttbarcands.slot1.p4[icat]))

    
        jetpt = ttbarcands.slot1.p4.pt
        jeteta = ttbarcands.slot1.p4.eta
        jetphi = ttbarcands.slot1.p4.phi
        jetmass = ttbarcands.slot1.p4.mass
        jetp = ttbarcands.slot1.p4.p
           
        
        
        
        # values for mistag rate calculation #
        
        numerator = np.where(antitag_probe, ttbarcands.slot1.p4.p, -1)
        denominator = np.where(antitag, ttbarcands.slot1.p4.p, -1)
        
        
       
        
        # event weights #
        
        weights = evtweights
        
        if self.bkgEst: weights = weights * mistag_weights
        
        # pt reweighting #
        if ('TTbar' in dataset):
            ttbar_wgt = pTReweighting(ttbarcands.slot0.pt, ttbarcands.slot1.pt)
            weights = weights * ttbar_wgt
            
            

        
        
  
        for i, [ilabel,icat] in enumerate(labels_and_categories.items()):
        
            icat = ak.flatten(icat)
        
            output['numerator'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     jetp = ak.flatten(numerator[icat]),
                                     weight = weights[icat],
                                    )
            output['denominator'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     jetp = ak.flatten(denominator[icat]),
                                     weight = weights[icat],
                                    )
            
            output['ttbarmass'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     ttbarmass = ak.flatten(ttbarmass[icat]),
                                     weight = weights[icat],
                                    )
            
            output['jetmass'].fill(dataset = dataset,
                                   anacat = self.label_to_int_dict[ilabel],
                                   jetmass = ak.flatten(jetmass[icat]),
                                   weight = weights[icat],
                                  )
            output['jetpt'].fill(dataset = dataset,
                                   anacat = self.label_to_int_dict[ilabel],
                                   jetpt = ak.flatten(jetpt[icat]),
                                   weight = weights[icat],
                                  )
            output['jeteta'].fill(dataset = dataset,
                                   anacat = self.label_to_int_dict[ilabel],
                                   jeteta = ak.flatten(jeteta[icat]),
                                   weight = weights[icat],
                                  )
            output['jetphi'].fill(dataset = dataset,
                                   anacat = self.label_to_int_dict[ilabel],
                                   jetphi = ak.flatten(jetphi[icat]),
                                   weight = weights[icat],
                                  )
        
        
        


        return output

    def postprocess(self, accumulator):
        return accumulator
        
        
        