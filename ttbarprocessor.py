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
import random
import correctionlib
import hist
import json

import awkward as ak

# for dask, `from python.corrections import` does not work
sys.path.append(os.getcwd()+'/python/')

from corrections import (
    GetFlavorEfficiency,
    HEMCleaning,
    GetL1PreFiringWeight,
    GetJECUncertainties,
    GetPDFWeights,
    GetPUSF,
    GetQ2weights,
    getLumiMaskRun2,
    getMETFilter,
    pTReweighting,
)
from btagCorrections import btagCorrections
from functions import getRapidity



#ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)


# --- Define 'Manual bins' to use for mistag plots for aesthetic purposes--- #
manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]


def update(events, collections):
    # https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


"""Package to perform the data-driven mistag-rate-based ttbar hadronic analysis. """
class TTbarResProcessor(processor.ProcessorABC):
    def __init__(self,
                 htCut=1400.,
                 ak8PtMin=400.,
                 minMSD=105.,
                 maxMSD=210.,
                 tau32Cut=0.65,
                 bdisc=0.5847,
                 deepAK8Cut=0.94,
                 useDeepAK8=True,
                 iov='2016APV',
                 bkgEst=False,
                 syst=False,
                 systematics = ['nominal', 'pileup'],
                 anacats = ['2t0bcen'],
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
        self.syst = syst
        self.systematics = systematics
        
        
        self.transfer_function = np.load('plots/save.npy')

        
        
        
        # analysis categories #
        self.anacats = anacats
        self.label_dict = {i: label for i, label in enumerate(self.anacats)}
        self.label_to_int_dict = {label: i for i, label in enumerate(self.anacats)}

        
        # systematics
        syst_category_strings = ['nominal'] 
        for s in self.systematics:
            if not 'nominal' in s:
                syst_category_strings.append(s+'Down')
                syst_category_strings.append(s+'Up')
        
        
        # axes
        dataset_axis     = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary Dataset")
        syst_axis        = hist.axis.StrCategory(syst_category_strings, name="systematic")
        ttbarmass_axis   = hist.axis.Regular(50, 800, 8000, name="ttbarmass", label=r"$m_{t\bar{t}}$ [GeV]")
        jetmass_axis     = hist.axis.Regular(50, 0, 500, name="jetmass", label=r"Jet $m$ [GeV]")
        ttbarmass2D_axis = hist.axis.Regular(20, 800, 8000, name="ttbarmass", label=r"$m_{t\bar{t}}$ [GeV]")
        jetmass2D_axis   = hist.axis.Regular(20, 0, 500, name="jetmass", label=r"Jet $m_{SD}$ [GeV]")
        jetpt_axis       = hist.axis.Regular(50, 400, 2000, name="jetpt", label=r"Jet $p_{T}$ [GeV]")
        jetp_axis        = hist.axis.Regular(100, 400, 3600, name="jetp", label=r"Jet $p$ [GeV]")
        jeteta_axis      = hist.axis.Regular(50, -2.4, 2.4, name="jeteta", label=r"Jet $\eta$")
        jetphi_axis      = hist.axis.Regular(50, -np.pi, np.pi, name="jetphi", label=r"Jet $\phi$")
        cats_axis        = hist.axis.IntCategory(range(len(self.anacats)), name="anacat", label="Analysis Category")
        manual_axis      = hist.axis.Variable(manual_bins, name="jetp", label=r"Jet Momentum [GeV]")
        btag_axis        = hist.axis.Regular(10, 0, 1, name="bdisc", label=r"DeepCSV")
        ttag_axis        = hist.axis.Regular(10, 0, 1, name="tdisc", label=r"DeepAK8")
        nsub_axis        = hist.axis.Regular(10, 0, 1, name="nsub", label=r"$\tau_{3} / \tau_{2}$")

        
        # output
        self.histo_dict = {

            
            # histograms
            'ttbarmass'  : hist.Hist(syst_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'numerator'  : hist.Hist(cats_axis, manual_axis, storage="weight", name="Counts"),
            'denominator': hist.Hist(cats_axis, manual_axis, storage="weight", name="Counts"),
            'jetmass' : hist.Hist(cats_axis, jetmass_axis, storage="weight", name="Counts"),
            'jetpt'  : hist.Hist(cats_axis, jetpt_axis, storage="weight", name="Counts"),
            'jeteta'  : hist.Hist(cats_axis, jeteta_axis, storage="weight", name="Counts"),
            'jetphi'  : hist.Hist(cats_axis, jetphi_axis, storage="weight", name="Counts"),
            'jetp'  : hist.Hist(cats_axis, jetp_axis, storage="weight", name="Counts"),
            'discriminators'  : hist.Hist(cats_axis,
                                          jetp_axis,
                                          btag_axis,
                                          ttag_axis,
                                          nsub_axis,
                                          storage="weight", name="Counts"),
            'deepak8'  : hist.Hist(cats_axis,
                                          jetp_axis,
                                          ttbarmass_axis,
                                          ttag_axis,
                                          storage="weight", name="Counts"),
            
            
            'mtt_vs_mt' : hist.Hist(syst_axis, cats_axis, jetmass2D_axis, ttbarmass2D_axis, storage="weight", name="Counts"),

            
            'deepak8_over_jetp': hist.Hist(cats_axis, ttag_axis, jetp_axis, storage="weight", name="Counts"),
            'tau32_over_jetp': hist.Hist(cats_axis, nsub_axis, jetp_axis, storage="weight", name="Counts"),
            'bdisc_over_jetpt': hist.Hist(cats_axis, btag_axis, jetp_axis, storage="weight", name="Counts"),


                        
            # accumulators
            'cutflow': processor.defaultdict_accumulator(int),
            
        }
        
      

        
    @property
    def accumulator(self):
        return self._accumulator
    
    
    
    def process(self, events):
        
        # reference for return processor.accumulate
        # https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py
        
        
        # Remove events with large weights
        if "QCD" in events.metadata['dataset']: 
            events = events[ events.Generator.binvar > 400 ] 
        
            if events.metadata['dataset'] not in self.means_stddevs : 
                average = np.average( events.genWeight )
                stddev = np.std( events.genWeight )
                self.means_stddevs[events.metadata['dataset']] = (average, stddev)            
            average,stddev = self.means_stddevs[events.metadata['dataset']]
            vals = (events.genWeight - average ) / stddev
            events = events[(np.abs(vals) < 2)]

        
        isData = ('JetHT' in events.metadata['dataset']) or ('SingleMu' in events.metadata['dataset'])
        
        noCorrections = (not 'jes' in self.systematics and not 'jer' in self.systematics)

        if isData or noCorrections:
            return self.process_analysis(events, 'nominal')
        
        
        FatJets = events.FatJet
        GenJets = events.GenJet
        Jets = events.Jet
        
        
        FatJets["p4"] = ak.with_name(FatJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        GenJets["p4"] = ak.with_name(GenJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        Jets["p4"]    = ak.with_name(Jets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")

        
        FatJets["p4"] = ak.with_name(FatJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        GenJets["p4"] = ak.with_name(GenJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        Jets["p4"]    = ak.with_name(Jets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")

        FatJets["matched_gen_0p2"] = FatJets.p4.nearest(GenJets.p4, threshold=0.2)
        FatJets["pt_gen"] = ak.values_astype(ak.fill_none(FatJets.matched_gen_0p2.pt, 0), np.float32)

        Jets["matched_gen_0p2"] = Jets.p4.nearest(GenJets.p4, threshold=0.2)
        Jets["pt_gen"] = ak.values_astype(ak.fill_none(Jets.matched_gen_0p2.pt, 0), np.float32)


        corrected_fatjets = GetJECUncertainties(FatJets, events, self.iov, R='AK8')
        corrected_jets = GetJECUncertainties(Jets, events, self.iov, R='AK4')
        
        
        del FatJets, GenJets, Jets
        
        if 'jes' in self.systematics:
            corrections = [
                ({"Jet": corrected_jets, "FatJet": corrected_fatjets}, 'nominal'),
                ({"Jet": corrected_jets.JES_jes.up, "FatJet": corrected_fatjets.JES_jes.up}, "jesUp"),
                ({"Jet": corrected_jets.JES_jes.down, "FatJet": corrected_fatjets.JES_jes.down}, "jesDown"),
            ]
        if 'jer' in self.systematics:
            corrections.extend([
                ({"Jet": corrected_jets.JER.up, "FatJet": corrected_fatjets.JER.up}, "jerUp"),
                ({"Jet": corrected_jets.JER.down, "FatJet": corrected_fatjets.JER.down}, "jerDown"),
            ])
            
        return processor.accumulate(self.process_analysis(update(events, collections), name) for collections, name in corrections)


    def process_analysis(self, events, correction):
        
                
        output = self.histo_dict        
        
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']
        
        isData = ('JetHT' in dataset) or ('SingleMu' in dataset)
            
        # Remove events with large weights
        if "QCD" in dataset: 
            events = events[ events.Generator.binvar > 400 ] 
        
        # lumi mask #
        if (isData):
            
            lumi_mask = np.array(getLumiMaskRun2(self.iov)(events.run, events.luminosityBlock), dtype=bool)
            events = events[lumi_mask]
            del lumi_mask

        elif 'QCD' in dataset: 
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
         
        # if blinding results in 0 events
        if (len(events) < 1): return output
        
        # event selection #
        selection = PackedSelection()

        # trigger cut #
        if isData:
            
            triggernames = { 
            
            "2016APV": ["PFHT900"],
            "2016" : ["PFHT900"],
            "2017" : ["PFHT1050"],
            "2018" : ["PFHT1050"],
        
            }
                        
            selection.add('trigger', events.HLT[triggernames[self.iov][0]])
            

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
            evtweights = np.ones(len(events))
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
        jetkincut = (FatJets.pt > self.ak8PtMin) & (np.abs(getRapidity(FatJets.p4)) < 2.4)
        
        selection.add('jetkincut', ak.any(jetkincut, axis=1))
        FatJets = FatJets[jetkincut]
        del jetkincut
        
        
        # at least 2 ak8 jets #
        selection.add('twoFatJets', (ak.num(FatJets) >= 2))
        

        # event cuts #
        
        # save cutflow
        cuts = []
        for cut in selection.names:
            cuts.append(cut)
            output['cutflow'][cut] += len(FatJets[selection.all(*cuts)])
        del cuts
        
        eventCut = selection.all(*selection.names)
        FatJets = FatJets[eventCut]
        SubJets = SubJets[eventCut]
        Jets    = Jets[eventCut]
        evtweights = evtweights[eventCut]
        events = events[eventCut]

        if not isData: GenJets = GenJets[eventCut]
            

        # ---- ttbar candidates ---- #
        
        # index = [[0], [1], [0], ... [0], [1], [1]] type='{# events} * var * int64'
        index = ak.unflatten( np.random.RandomState(random.seed()).randint(2, size=len(FatJets)), np.ones(len(FatJets), dtype='i'))
        
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
        
        
        
        # ----------- DeepAK8 Tagger (Discriminator Cut) ----------- #
        if self.useDeepAK8:
            ttag_s0_disc = ttbarcands.slot0.deepTagMD_TvsQCD > self.deepAK8Cut
            ttag_s1_disc = ttbarcands.slot1.deepTagMD_TvsQCD > self.deepAK8Cut
            antitag_disc = (ttbarcands.slot0.deepTagMD_TvsQCD < self.deepAK8Cut) & (ttbarcands.slot0.deepTagMD_TvsQCD > 0.2)
            
            mcut_s0 = (self.minMSD < ttbarcands.slot0.msoftdrop) & (ttbarcands.slot0.msoftdrop < self.maxMSD) 
            mcut_s1 = (self.minMSD < ttbarcands.slot1.msoftdrop) & (ttbarcands.slot1.msoftdrop < self.maxMSD) 

            ttag_s0 = (ttag_s0_disc) #& (mcut_s0)
            ttag_s1 = (ttag_s1_disc) #& (mcut_s1)
            antitag = (antitag_disc) #& (mcut_s0) # The Probe jet will always be ttbarcands.slot1 (at)

            
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
        
        
        # tau32 cuts for plotting
        tau32_s0 = np.where(ttbarcands.slot0.tau2>0,ttbarcands.slot0.tau3/ttbarcands.slot0.tau2, 0 )
        tau32_s1 = np.where(ttbarcands.slot1.tau2>0,ttbarcands.slot1.tau3/ttbarcands.slot1.tau2, 0 )

        taucut_s0 = tau32_s0 < self.tau32Cut
        taucut_s1 = tau32_s1 < self.tau32Cut
        
        
        
        # ---- Define "Top Tag" Regions ---- #
        antitag_probe = np.logical_and(antitag, ttag_s1) # Found an antitag and ttagged probe pair for mistag rate (AT&Pt)
        pretag =  ttag_s0 # Only jet0 (pret)
        ttag0 =   (~ttag_s0) & (~ttag_s1) # No tops tagged (0t)
        ttag1 =   ttag_s0 ^ ttag_s1 # Exclusively one top tagged (1t)
        ttagI =   ttag_s0 | ttag_s1 # At least one top tagged ('I' for 'inclusive' tagger; >=1t; 1t+2t)
        ttag2 =   ttag_s0 & ttag_s1 # Both jets top tagged (2t)
        Alltags = ttag0 | ttagI #Either no tag or at least one tag (0t+1t+2t)
                         
        
        
        # b tagger #
        
        bdisc_s0 = np.maximum(SubJet00.btagDeepB , SubJet01.btagDeepB)
        bdisc_s1 = np.maximum(SubJet10.btagDeepB , SubJet11.btagDeepB)
        tdisc_s0 = ttbarcands.slot0.deepTagMD_TvsQCD
        tdisc_s1 = ttbarcands.slot1.deepTagMD_TvsQCD

        
        btag_s0 = ( np.maximum(SubJet00.btagDeepB , SubJet01.btagDeepB) > self.bdisc )
        btag_s1 = ( np.maximum(SubJet10.btagDeepB , SubJet11.btagDeepB) > self.bdisc )
        
        # --- Define "B Tag" Regions ---- #
        btag0 = (~btag_s0) & (~btag_s1) #(0b)
        btag1 = btag_s0 ^ btag_s1 #(1b)
        btag2 = btag_s0 & btag_s1 #(2b)
        
        # rapidity #
        cen = np.abs(getRapidity(ttbarcands.slot0.p4) - getRapidity(ttbarcands.slot1.p4)) < 1.0
        fwd = (~cen)
    
    
        # rapidity, btag and top tag categories
        regs = {'cen': cen, 'fwd': fwd}
        btags = {'0b': btag0, '1b':btag1, '2b':btag2}
        ttags = {"AT&Pt": antitag_probe, 
                 "at":antitag, 
                 "pret":pretag, 
                 "0t":ttag0, 
                 "1t":ttag1, 
                 ">=1t":ttagI, 
                 "2t":ttag2,
                 ">=0t":Alltags
                }
        
        
        # get all analysis category masks
        categories = { t[0]+b[0]+y[0] : (t[1]&b[1]&y[1])  for t,b,y in itertools.product( ttags.items(), 
                                                                        btags.items(), 
                                                                        regs.items())
            }
        
        # use subset of analysis category masks from ttbaranalysis.py
        labels_and_categories = {label:categories[label] for label in self.anacats}
    
        
        jetmass = ttbarcands.slot1.p4.mass
        jetp = ttbarcands.slot1.p4.p
        jetmsd = ttbarcands.slot0.msoftdrop

        
        
        # event weights #
        
        weights = Weights(len(evtweights))
        weights.add('genWeight', evtweights)
                        
        # if running background estimation
        if (self.bkgEst):

            # for mistag rate weights
            mistag_rate_df = pd.read_csv(f'data/corrections/backgroundEstimate/mistag_rate_{self.iov}.csv')
            pbins = mistag_rate_df['jetp bins'].values
            mistag_weights = np.ones(len(FatJets), dtype=float)
            
            
            # for mass modification

#             qcdfile = util.load(f'data/corrections/backgroundEstimate/QCD_{self.iov}.coffea')
            qcd_jetmass_dict = json.load(open(f'data/corrections/backgroundEstimate/QCD_jetmass_{self.iov}.json'))
            qcd_jetmass_bins = qcd_jetmass_dict['bins']
        
        
            # for transfer function
            
            bins_mt  = np.arange(0,500,10)
            bins_mtt = np.arange(800,8000,144)
                     
    
            for ilabel,icat in labels_and_categories.items():
            
                icat = ak.flatten(icat)

                # get antitag region and signal region labels
                # ilabel[-5:] = bcat + ycat (0bcen for example)
                label_at = 'at'+ilabel[-5:]
                label_2t = '2t'+ilabel[-5:]

                
                # get mistag rate for antitag region
                mistag_rate = mistag_rate_df[label_at].values

                # get p bin for probe jet p
                mistag_pbin = np.digitize(ak.flatten(jetp[icat]), pbins) - 1

                # store mistag weights for events in this category
                mistag_weights[icat] = mistag_rate[mistag_pbin]



                # qcd mass modification #

                # get distribution of jet mass in QCD signal ('2t') region
                qcd_jetmass_counts = qcd_jetmass_dict[label_2t]

                # randomly select jet mass from distribution
                ModMass_hist_dist = ss.rv_histogram([qcd_jetmass_counts[:-1], qcd_jetmass_bins])
                ttbarcands.slot1.p4[icat]["fMass"] = ModMass_hist_dist.rvs(size=len(ttbarcands.slot1.p4[icat]))
                
                
            weights.add('mistag', mistag_weights)
    
        del jetmass, jetp, jetmsd
        
        jetpt = ttbarcands.slot1.p4.pt
        jeteta = ttbarcands.slot1.p4.eta
        jetphi = ttbarcands.slot1.p4.phi
        jetmass = ttbarcands.slot1.p4.mass
        jetp = ttbarcands.slot1.p4.p
        
        # plot same jetmass as pre-tagged, anti-tagged jet
        jetmsd = ttbarcands.slot0.msoftdrop
           
        
        # values for mistag rate calculation #
        numerator = np.where(antitag_probe, ttbarcands.slot1.p4.p, -1)
        denominator = np.where(antitag, ttbarcands.slot1.p4.p, -1)
        
        # pt reweighting #
#         if ('TTbar' in dataset):
#             ttbar_wgt = pTReweighting(ttbarcands.slot0.pt, ttbarcands.slot1.pt)
#             weights.add('ptReweighting', ak.flatten(ttbar_wgt))
                 
        if self.syst and not isData:
                    
            if 'pileup' in self.systematics:
                
                puNom, puUp, puDown = GetPUSF(events, self.iov)
                weights.add("pileup", 
                    weight=puNom, 
                    weightUp=puUp, 
                    weightDown=puDown,
                           )

            if ('prefiring' in self.systematics) and ("L1PreFiringWeight" in events.fields):
                if ('2016' in self.iov) or ('2017' in self.iov):
                
                    prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events)
                    weights.add("prefiring", 
                        weight=prefiringNom, 
                        weightUp=prefiringUp, 
                        weightDown=prefiringDown,
                               )
                    
            if 'pdf' in self.systematics:
                
                pdfUp, pdfDown, pdfNom = GetPDFWeights(events)
                weights.add("pdf", 
                    weight=pdfNom, 
                    weightUp=pdfUp, 
                    weightDown=pdfDown,
                           )    
            
            if 'q2' in self.systematics:
                
                q2Nom, q2Up, q2Down = GetQ2weights(events)
                
                weights.add("q2", 
                    weight=q2Nom, 
                    weightUp=q2Up, 
                    weightDown=q2Down,
                           )   
                
            if 'btag' in self.systematics:
                
                btag_wgts_nom = np.ones(len(events))
                btag_wgts_up  = np.ones(len(events))
                btag_wgts_down = np.ones(len(events))
                
                btag_wgts_nom_bcats = btagCorrections([btag0, btag1, btag2], 
                                                      [SubJet00, SubJet01, SubJet10, SubJet11], 
                                                      isData, 
                                                      self.bdisc,
                                                      sysType='central')
                
                btag_wgts_up_bcats = btagCorrections([btag0, btag1, btag2], 
                                                      [SubJet00, SubJet01, SubJet10, SubJet11], 
                                                      isData, 
                                                      self.bdisc,
                                                      sysType='up')
                
                btag_wgts_down_bcats = btagCorrections([btag0, btag1, btag2], 
                                                      [SubJet00, SubJet01, SubJet10, SubJet11], 
                                                      isData, 
                                                      self.bdisc,
                                                      sysType='down')
                
                
                btag_wgts_nom[ak.flatten(btag0)]  = btag_wgts_nom_bcats['0b'][ak.flatten(btag0)]
                btag_wgts_up[ak.flatten(btag0)]   = btag_wgts_nom_bcats['0b'][ak.flatten(btag0)]
                btag_wgts_down[ak.flatten(btag0)] = btag_wgts_nom_bcats['0b'][ak.flatten(btag0)]
                btag_wgts_nom[ak.flatten(btag1)]  = btag_wgts_nom_bcats['1b'][ak.flatten(btag1)]
                btag_wgts_up[ak.flatten(btag1)]   = btag_wgts_nom_bcats['1b'][ak.flatten(btag1)]
                btag_wgts_down[ak.flatten(btag1)] = btag_wgts_nom_bcats['1b'][ak.flatten(btag1)]
                btag_wgts_nom[ak.flatten(btag2)]  = btag_wgts_nom_bcats['2b'][ak.flatten(btag2)]
                btag_wgts_up[ak.flatten(btag2)]   = btag_wgts_nom_bcats['2b'][ak.flatten(btag2)]
                btag_wgts_down[ak.flatten(btag2)] = btag_wgts_nom_bcats['2b'][ak.flatten(btag2)]
                
                weights.add("btag", 
                    weight=btag_wgts_nom, 
                    weightUp=btag_wgts_up, 
                    weightDown=btag_wgts_down,
                           )
                
                del btag_wgts_nom, btag_wgts_up, btag_wgts_down
                del btag_wgts_nom_bcats, btag_wgts_up_bcats, btag_wgts_down_bcats



        for i, [ilabel,icat] in enumerate(labels_and_categories.items()):
        
            icat = ak.flatten(icat)
                
                
                                
            output['numerator'].fill(anacat = i,
                                     jetp = ak.flatten(numerator[icat]),
                                     weight = weights.weight()[icat],
                                    )
            
            output['denominator'].fill(anacat = i,
                                       jetp = ak.flatten(denominator[icat]),
                                       weight = weights.weight()[icat],
                                    )
            
            output['ttbarmass'].fill(systematic=correction,
                                     anacat = i,
                                     ttbarmass = ak.flatten(ttbarmass[icat]),
                                     weight = weights.weight()[icat],
                                    )
            
            output['jetmass'].fill(anacat = i,
                                   jetmass = ak.flatten(jetmass[icat]),
                                   weight = weights.weight()[icat],
                                  )
            output['jetpt'].fill(anacat = i,
                                 jetpt = ak.flatten(jetpt[icat]),
                                 weight = weights.weight()[icat],
                                  )
            
            output['jeteta'].fill(anacat = i,
                                  jeteta = ak.flatten(jeteta[icat]),
                                  weight = weights.weight()[icat],
                                  )
            output['jetphi'].fill(anacat = i,
                                  jetphi = ak.flatten(jetphi[icat]),
                                  weight = weights.weight()[icat],
                                  )
            
            output['mtt_vs_mt'].fill(systematic=correction,
                                     anacat = i,
                                     jetmass = ak.flatten(jetmsd[icat]),
                                     ttbarmass = ak.flatten(ttbarmass[icat]),
                                     weight = weights.weight()[icat],
                                    )
            
            output['discriminators'].fill(anacat = i,
                                          jetp = ak.flatten(jetp[icat]),
                                          bdisc = ak.flatten(bdisc_s1[icat]),
                                          tdisc = ak.flatten(tdisc_s1[icat]),
                                          nsub = ak.flatten(tau32_s1)[icat],
                                          weight = weights.weight()[icat],
                                         )

            output['deepak8'].fill(anacat = i,
                                   jetp = ak.flatten(jetp[icat]),
                                   ttbarmass = ak.flatten(ttbarmass[icat]),
                                   tdisc = ak.flatten(tdisc_s1[icat]),
                                   weight = weights.weight()[icat],
                                  )
            
            


                
            if not 'jes' in correction and not 'jer' in correction:    
            
                for syst in weights.variations:

                    output['ttbarmass'].fill(systematic=syst,
                                         anacat = i,
                                         ttbarmass = ak.flatten(ttbarmass[icat]),
                                         weight = weights.weight(syst)[icat],
                                        )

                    output['mtt_vs_mt'].fill(systematic=syst,
                                         anacat = i,
                                         ttbarmass = ak.flatten(ttbarmass[icat]),
                                         jetmass = ak.flatten(jetmsd[icat]),
                                         weight = weights.weight(syst)[icat],
                                        )

        
        


        return output

    def postprocess(self, accumulator):
        return accumulator
        
        
        
