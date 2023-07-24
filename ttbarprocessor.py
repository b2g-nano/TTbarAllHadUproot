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
import logging
import psutil
import time

import awkward as ak

# for dask, `from python.corrections import` does not work
sys.path.append(os.getcwd()+'/python/')

from corrections import (
    GetFlavorEfficiency,
    HEMCleaning,
    HEMVeto,
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




# logging
logfile = 'coffea_' + str(int(time.time())) + '.log'
print(logfile)
logging.basicConfig(filename=logfile, level=logging.DEBUG)
logger = logging.getLogger('__main__')
logger.setLevel(logging.DEBUG)


#ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)


# --- Define 'Manual bins' to use for mistag plots for aesthetic purposes--- #
manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]




def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_bytes = memory_info.rss
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)

    return memory_usage_mb



def update(events, collections):
    # https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
#     logger.debug('update:%s:%s', time.time(), collections)
    
    for name, value in collections.items():
        out = ak.with_field(out, value, name)

    return out


"""Package to perform the data-driven mistag-rate-based ttbar hadronic analysis. """
class TTbarResProcessor(processor.ProcessorABC):
    def __init__(self,
                 htCut=950.,
                 ak8PtMin=400.,
                 minMSD=105.,
                 maxMSD=210.,
                 tau32Cut=0.65,
                 bdisc=0.5847,
                 deepAK8Cut='tight',
                 useDeepAK8=True,
                 useDeepCSV=True,
                 iov='2016',
                 bkgEst=False,
                 noSyst=False,
                 systematics = ['nominal', 'pileup'],
                 anacats = ['2t0bcen'],
                 #rpf_params = {'params':[1.0], 'errors':[0.0]},
                ):
                 
        self.iov = iov
        self.htCut = htCut
        self.minMSD = minMSD
        self.maxMSD = maxMSD
        self.tau32Cut = tau32Cut
        self.ak8PtMin = ak8PtMin
        self.bdisc = bdisc
        self.useDeepAK8 = useDeepAK8
        self.useDeepCSV = useDeepCSV
        self.means_stddevs = defaultdict()
        self.bkgEst = bkgEst
        self.noSyst = noSyst
        self.systematics = systematics
        #self.rpf_params = rpf_params        
        
#         self.transfer_function = np.load('plots/save.npy')

        deepak8cuts = {
            'loose':{ # 1%
                '2016APV': 0.486, 
                '2016': 0.475,
                '2017': 0.487,
                '2018': 0.477,
            },
            'medium':{ # 0.5%
                '2016APV': 0.677, 
                '2016': 0.666,
                '2017': 0.673,
                '2018': 0.669,
            },
            'tight': { # 0.1%
                '2016APV': 0.902, 
                '2016': 0.897,
                '2017': 0.898,
                '2018': 0.900,
            } 
        }
    
        btagcuts = {
            'loose':{
                '2016APV': 0.2027, 
                '2016':    0.1918,
                '2017':    0.1355,
                '2018':    0.1208,
            },
            'medium':{
                '2016APV': 0.6001, 
                '2016':    0.5847,
                '2017':    0.4506,
                '2018':    0.4506,
            } 
        }
        
        
        self.weights = {}
    
        
        
        self.deepAK8Cut = deepak8cuts['medium'][self.iov]
        
        
        if self.useDeepCSV:
            self.bdisc = btagcuts['medium'][self.iov]
        else:
            self.bdisc = 0.8484
        
        
        
        
        # analysis categories #
        self.anacats = anacats
        self.label_dict = {i: label for i, label in enumerate(self.anacats)}
        self.label_to_int_dict = {label: i for i, label in enumerate(self.anacats)}

        
        # systematics
        syst_category_strings = ['nominal']
        if not self.noSyst:
            for s in self.systematics:
                if (s != 'nominal'):
                    
                    if ('hem' in s):
                        syst_category_strings.append(s)
                    else:
                        syst_category_strings.append(s+'Down')
                        syst_category_strings.append(s+'Up')
        
#         syst_category_strings = ['nominal', 'test1', 'test2', 'test3', 'test4']
        
        # axes
        dataset_axis     = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary Dataset")
        syst_axis        = hist.axis.StrCategory(syst_category_strings, name="systematic")
        ttbarmass_axis   = hist.axis.Regular(50, 800, 8000, name="ttbarmass", label=r"$m_{t\bar{t}}$ [GeV]")
        jetmass_axis     = hist.axis.Regular(50, 0, 500, name="jetmass", label=r"Jet $m$ [GeV]")
        jetmsd_axis      = hist.axis.Regular(20, 0, 500, name="jetmass", label=r"Jet $m_{SD}$ [GeV]")
        ttbarmass2D_axis = hist.axis.Regular(20, 800, 6800, name="ttbarmass", label=r"$m_{t\bar{t}}$ [GeV]")
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
            'ttbarmass'  : hist.Hist(syst_axis, cats_axis, ttbarmass2D_axis, storage="weight", name="Counts"),
            'numerator'  : hist.Hist(cats_axis, manual_axis, storage="weight", name="Counts"),
            'denominator': hist.Hist(cats_axis, manual_axis, storage="weight", name="Counts"),
            'jetmass' : hist.Hist(syst_axis, cats_axis, jetmass2D_axis, storage="weight", name="Counts"),
            'jetmsd' : hist.Hist(syst_axis, cats_axis, jetmsd_axis, storage="weight", name="Counts"),
            'jetpt'  : hist.Hist(syst_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            'jeteta'  : hist.Hist(syst_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            'jetphi'  : hist.Hist(syst_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            'jetp'  : hist.Hist(syst_axis, cats_axis, jetp_axis, storage="weight", name="Counts"),
#             'discriminators'  : hist.Hist(cats_axis,
#                                           jetp_axis,
#                                           btag_axis,
#                                           ttag_axis,
#                                           nsub_axis,
#                                           storage="weight", name="Counts"),
#             'deepak8'  : hist.Hist(cats_axis,
#                                           jetp_axis,
#                                           ttbarmass_axis,
#                                           ttag_axis,
#                                           storage="weight", name="Counts"),
            
            
            'mtt_vs_mt' : hist.Hist(syst_axis, cats_axis, jetmass2D_axis, ttbarmass2D_axis, storage="weight", name="Counts"),

            
            'deepak8_over_jetp': hist.Hist(cats_axis, ttag_axis, jetp_axis, storage="weight", name="Counts"),
            'tau32_over_jetp': hist.Hist(cats_axis, nsub_axis, jetp_axis, storage="weight", name="Counts"),
            'bdisc_over_jetpt': hist.Hist(cats_axis, btag_axis, jetp_axis, storage="weight", name="Counts"),


                        
            # accumulators
            'cutflow': processor.defaultdict_accumulator(int),
            'weights': processor.defaultdict_accumulator(float),
            'systematics': processor.defaultdict_accumulator(float),
        }
        
      

        
    @property
    def accumulator(self):
        return self._accumulator
    
    
    
    def process(self, events):
        
        
        logger.debug('memory:%s:start %s preprocessor: %s', time.time(), events.metadata['dataset'], get_memory_usage())

                
        # reference for return processor.accumulate
        # https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py
        
        nEvents = len(events.event)


        
        # Remove events with large weights
        if "QCD" in events.metadata['dataset']: # and ('2017' not in self.iov): 
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

        if noCorrections or self.noSyst:
            return self.process_analysis(events, 'nominal', nEvents)
        
        
        if isData:
            
            return processor.accumulate([
                self.process_analysis(events, 'nominal', nEvents),
                self.process_analysis(events, 'hemVeto', nEvents)
            ]) 
        
        
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


        corrected_fatjets = GetJECUncertainties(FatJets, events, self.iov, R='AK8', isData=isData)
        corrected_jets = GetJECUncertainties(Jets, events, self.iov, R='AK4', isData=isData)
        
        
        if (len(corrected_jets.pt[0]) > 1) and  (len(corrected_fatjets.pt[0]) > 1) :
        
            logger.debug('JEC:%s:JES up, nom, down:%s:%s:%s', time.time(), 
                         corrected_fatjets.JES_jes.up.pt[0][0],
                         corrected_fatjets.pt[0][0],
                         corrected_fatjets.JES_jes.down.pt[0][0])

            logger.debug('JEC:%s:JER up, nom, down:%s:%s:%s', time.time(), 
                         corrected_fatjets.JER.up.pt[0][0],
                         corrected_fatjets.pt[0][0],
                         corrected_fatjets.JER.down.pt[0][0])

            logger.debug('JEC:%s:JES up, nom, down AK4:%s:%s:%s', time.time(), 
                         corrected_jets.JES_jes.up.pt[0][0],
                         corrected_jets.pt[0][0],
                         corrected_jets.JES_jes.down.pt[0][0])

            logger.debug('JEC:%s:JER up, nom, down AK4:%s:%s:%s', time.time(), 
                         corrected_jets.JER.up.pt[0][0],
                         corrected_jets.pt[0][0],
                         corrected_jets.JER.down.pt[0][0])



        
        
        if 'jes' in self.systematics:
            corrections = [
                ({"Jet": Jets, "FatJet": FatJets}, 'nominal'),
                ({"Jet": corrected_jets.JES_jes.up, "FatJet": corrected_fatjets.JES_jes.up}, "jesUp"),
                ({"Jet": corrected_jets.JES_jes.down, "FatJet": corrected_fatjets.JES_jes.down}, "jesDown"),
            ]
        if 'jer' in self.systematics:
            corrections.extend([
                ({"Jet": corrected_jets.JER.up, "FatJet": corrected_fatjets.JER.up}, "jerUp"),
                ({"Jet": corrected_jets.JER.down, "FatJet": corrected_fatjets.JER.down}, "jerDown"),
            ])
            
            
        
            
#         if ('hem' in self.systematics) and ('2018' in self.iov):
            
#             if 'hemVeto' in self.systematics:
                
            
#                 corrections.extend([
#                     ({"Jet": corrected_jets, "FatJet": corrected_fatjets}, "hemVeto"),
#                     ])

            
#             corrected_jets_hem    = HEMCleaning(corrected_jets)
#             corrected_fatjets_hem = HEMCleaning(corrected_fatjets)

#             corrections.extend([
#             ({"Jet": corrected_jets_hem, "FatJet": corrected_fatjets_hem}, "hem"),
#             ])
            
#             corrections.extend([
#             ({"Jet": Jets, "FatJet": FatJets}, "hem"),
#             ])

                

        
        
#         # get nominal output
#         output_total = self.process_analysis(update(events, corrections[0][0]), 'nominal', nEvents)
        
# #         logger.debug('output:%s:nominal:%s:%s', time.time(), output_total['cutflow'], output_total['systematics'])
# #         logger.debug('output:%s:nominal:%s', time.time(), output_total['weights'])
        
#         # loop through corrections
#         outputs = {}
#         for collections, name in corrections[1:]:
#             process_output = self.process_analysis(update(events, collections), name, nEvents)
#             outputs[name] = process_output
            
# #             logger.debug('output:%s:%s:%s', time.time(), name, process_output['weights'])


#         # combine outputs
#         for name, output_correction in outputs.items():
#             for key in output_total.keys():

#                 if 'hist' in str(type(output_total[key])):
#                     if 'systematic' in list(output_total[key].axes.name):
#                         output_total[key] += output_correction[key]

#                 elif 'accumulator' in str(type(output_total[key])):
#                     if key != 'cutflow':
#                         output_total[key][name] = process_output[key][name]


        
        # loop through corrections
        outputs = []
        for collections, name in corrections:
            outputs.append(self.process_analysis(update(events, collections), name, nEvents))
           
        output_total = processor.accumulate(outputs)                       

                        
        return output_total

     


    def process_analysis(self, events, correction, nEvents):
        
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']
        
        logger.debug('memory:%s: start processor %s:%s', time.time(), correction, get_memory_usage())

                
        isNominal = (correction=='nominal')
        isData = ('JetHT' in dataset) or ('SingleMu' in dataset)

        
        
        if (self.iov == '2018'):
            
            if isData:
                
                # keep events below 
                    
                    
                events = events[HEMVeto(events.Jet, events.FatJet, events.run)]


            else:
                events = events[HEMVeto(events.Jet, events.FatJet, events.run)]
                

        
                
        output = self.histo_dict 
        
        if isNominal:
            output['cutflow']['all events 1'] += nEvents
        
        
        # lumi mask #
        if (isData):
            
            lumi_mask = np.array(getLumiMaskRun2(self.iov)(events.run, events.luminosityBlock), dtype=bool)
            events = events[lumi_mask]
            del lumi_mask

        
        
#         # blinding #
#         if isData: #and (('2017' in self.iov) or ('2018' in self.iov)):
#             events = events[::10]
            
        
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
                    
        
        logger.debug('memory:%s: get nanoAOD objects %s:%s', time.time(), correction, get_memory_usage())

        
        
        
        # ---- Get event weights from dataset ---- #

        # if blinding + trigger results in few events
        if (len(events) < 10): return output
        
        
        
        if isData:
            evtweights = np.ones(len(events))
        else:
            if "LHEWeight_originalXWGTUP" not in events.fields: 
                evtweights = events.genWeight
            else: 
                evtweights = events.LHEWeight_originalXWGTUP
        if isNominal:
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
        if isNominal:
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

    
#         index = ak.unflatten( 
#             np.ones(len(FatJets), 
#                     dtype='i'
#                    ), 
#             np.ones(len(FatJets), 
#                     dtype='i'
#                    )
#         )
        
        
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
        if isNominal:
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
        
        
        logger.debug('memory:%s: apply event cuts %s:%s', time.time(), correction, get_memory_usage())

        
        # ttbarmass
        ttbarmass = (ttbarcands.slot0.p4 + ttbarcands.slot1.p4).mass
        
        # subjets
        SubJet00 = SubJets[ttbarcands.slot0.subJetIdx1]
        SubJet01 = SubJets[ttbarcands.slot0.subJetIdx2]
        SubJet10 = SubJets[ttbarcands.slot1.subJetIdx1]
        SubJet11 = SubJets[ttbarcands.slot1.subJetIdx2]
        
        
        
        # ----------- DeepAK8 Tagger (Discriminator Cut) ----------- #
        if self.useDeepAK8:
            ttag_s0_disc = (ttbarcands.slot0.deepTagMD_TvsQCD > self.deepAK8Cut)
            ttag_s1_disc = (ttbarcands.slot1.deepTagMD_TvsQCD > self.deepAK8Cut)
            antitag_disc = ((ttbarcands.slot0.deepTagMD_TvsQCD < self.deepAK8Cut) & (ttbarcands.slot0.deepTagMD_TvsQCD > 0.2))
            
            mcut_s0 = (self.minMSD < ttbarcands.slot0.msoftdrop) & (ttbarcands.slot0.msoftdrop < self.maxMSD) 
            mcut_s1 = (self.minMSD < ttbarcands.slot1.msoftdrop) & (ttbarcands.slot1.msoftdrop < self.maxMSD) 

            ttag_s0 = (ttag_s0_disc) #& (mcut_s0)
            ttag_s1 = (ttag_s1_disc) #& (mcut_s1)
            antitag = (antitag_disc) #& (mcut_s0) # The Probe jet will always be ttbarcands.slot1 (at)

            
        # ----------- CMS Top Tagger Version 2 (SD and Tau32 Cuts) ----------- #
        else:
            tau32_s0 = np.where(ttbarcands.slot0.tau2>0,ttbarcands.slot0.tau3/ttbarcands.slot0.tau2, 0 )
            tau32_s1 = np.where(ttbarcands.slot1.tau2>0,ttbarcands.slot1.tau3/ttbarcands.slot1.tau2, 0 )

            taucut_s0 = (tau32_s0 < self.tau32Cut)
            taucut_s1 = (tau32_s1 < self.tau32Cut)

            mcut_s0 = ((self.minMSD < ttbarcands.slot0.msoftdrop) & (ttbarcands.slot0.msoftdrop < self.maxMSD) )
            mcut_s1 = ((self.minMSD < ttbarcands.slot1.msoftdrop) & (ttbarcands.slot1.msoftdrop < self.maxMSD) )

            ttag_s0 = ((taucut_s0) & (mcut_s0))
            ttag_s1 = ((taucut_s1) & (mcut_s1))
            antitag = ((~taucut_s0) & (mcut_s0)) # The Probe jet will always be ttbarcands.slot1 (at)
        
        
        # discriminators for plotting
        tau32_s0 = np.where(ttbarcands.slot0.tau2>0,ttbarcands.slot0.tau3/ttbarcands.slot0.tau2, 0 )
        tau32_s1 = np.where(ttbarcands.slot1.tau2>0,ttbarcands.slot1.tau3/ttbarcands.slot1.tau2, 0 )

        taucut_s0 = (tau32_s0 < self.tau32Cut)
        taucut_s1 = (tau32_s1 < self.tau32Cut)
        
        bdisc_s0 = np.maximum(SubJet00.btagDeepB , SubJet01.btagDeepB)
        bdisc_s1 = np.maximum(SubJet10.btagDeepB , SubJet11.btagDeepB)
        
        tdisc_s0 = ttbarcands.slot0.deepTagMD_TvsQCD
        tdisc_s1 = ttbarcands.slot1.deepTagMD_TvsQCD
        
        
        
        # ---- Define "Top Tag" Regions ---- #
        antitag_probe = np.logical_and(antitag, ttag_s1) # Found an antitag and ttagged probe pair for mistag rate (AT&Pt)
        pretag =  (ttag_s0)                    # Only jet0 (pret)
        ttag0 =   ((~ttag_s0) & (~ttag_s1))    # No tops tagged (0t)
        ttag1 =   (ttag_s0 ^ ttag_s1)          # Exclusively one top tagged (1t)
        ttagI =   (ttag_s0 | ttag_s1)          # At least one top tagged ('I' for 'inclusive' tagger; >=1t; 1t+2t)
        ttag2 =   (ttag_s0 & ttag_s1)          # Both jets top tagged (2t)
        Alltags = (ttag0 | ttagI)              # Either no tag or at least one tag (0t+1t+2t)
                         
        
        
        # b tagger #
        
        
        if self.useDeepCSV:
            btag_s0 = ( np.maximum(SubJet00.btagDeepB , SubJet01.btagDeepB) > self.bdisc )
            btag_s1 = ( np.maximum(SubJet10.btagDeepB , SubJet11.btagDeepB) > self.bdisc )

        else:
            btag_s0 = ( np.maximum(SubJet00.btagCSVV2 , SubJet01.btagCSVV2) > 0.8484 )
            btag_s1 = ( np.maximum(SubJet10.btagCSVV2 , SubJet11.btagCSVV2) > 0.8484 )


        # --- Define "B Tag" Regions ---- #
        btag0 = ((~btag_s0) & (~btag_s1)) #(0b)
        btag1 = (btag_s0 ^ btag_s1) #(1b)
        btag2 = (btag_s0 & btag_s1) #(2b)
        
        # rapidity #
        cen = (np.abs(getRapidity(ttbarcands.slot0.p4) - getRapidity(ttbarcands.slot1.p4)) < 1.0)
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
    
    
    
        logger.debug('memory:%s: get analysis categories %s:%s', time.time(), correction, get_memory_usage())

        
        jetmass = ttbarcands.slot1.p4.mass
        jetp = ttbarcands.slot1.p4.p
        jetmsd = ttbarcands.slot0.msoftdrop

        
        
        # event weights #
        
        # if few events
        if (len(evtweights) < 10): return output
        
        
        self.weights[correction] = Weights(len(evtweights))
        
        self.weights[correction].add('genWeight', evtweights)
                        
        # if running background estimation
        if (self.bkgEst) and isNominal:
            
            if self.bkgEst == '2dalphabet':
                # for transfer function

                # transfer functions multiplies by bin count
                # parameters need to be divided by bin size
                xbinsize = 25
                ybinsize = 360
                scale_tf = 1 / (1000 * xbinsize*ybinsize)

                # get bins of mt and mtt and x and y values
                bins_mt  = np.arange(0,500,xbinsize) # 20 bins in mt
                bins_mtt = np.arange(800,8000,ybinsize) # 20 bins in mtt
                x = (1/xbinsize) * bins_mt[(np.digitize(ak.flatten(jetmass), bins_mt) - 1)]
                y = (1/ybinsize) * bins_mtt[(np.digitize(ak.flatten(ttbarmass), bins_mtt) - 1)]

                # get parameters of transfer function with uncertainties
                p = self.rpf_params['param']
                pUp = [p + err for p, err in zip(self.rpf_params['param'], self.rpf_params['error'])]
                pDn = [p - err for p, err in zip(self.rpf_params['param'], self.rpf_params['error'])]



                if '0x1' in self.rpf_params['function']:

                     # @0 + @1*y

                    rpfNom  = p[0] + p[1] * y
                    rpfUp   = pUp[0] + pUp[1] * y
                    rpfDown = pDn[0] + pDn[1] * y

                elif '1x0' in self.rpf_params['function']:

                    # @0 + @1*x

                    rpfNom  = (p[1] * x + p[0])
                    rpfUp   = (pUp[1] * x + pUp[0])
                    rpfDown = (pDn[1] * x + pDn[0])

                elif '3x1' in self.rpf_params['function']:

                    # (@0+@1*x+@2*x*x+@3*x*x*x)*(1+@4*y)

                    rpfNom   = scale_tf * ( p[0] + p[1]*x + p[2]*x*x + p[3]*x*x*x ) * ( 1 + p[4]*y )
                    rpfUp    = scale_tf * ( pUp[0] + pUp[1]*x + pUp[2]*x*x + pUp[3]*x*x*x ) * ( 1 + pUp[4]*y )
                    rpfDown  = scale_tf * ( pDn[0] + pDn[1]*x + pDn[2]*x*x + pDn[3]*x*x*x ) * ( 1 + pDn[4]*y )

                else:

                    rpfNom = np.ones(len(events))   
                    rpfUp = np.ones(len(events))   
                    rpfDown = np.ones(len(events))  


                self.weights[correction].add("transferFunction",
                        weight=rpfNom, 
                        weightUp=rpfUp, 
                        weightDown=rpfDown,
                               ) 
                
            elif self.bkgEst == 'mistag':

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
                bins_mtt = np.arange(800,8000,360)


                for ilabel,icat in labels_and_categories.items():

                    icat = ak.flatten(icat)
                    
                    if 'pret' in ilabel:

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


                self.weights[correction].add('mistag', mistag_weights)
    
        del jetmass, jetp, jetmsd
        
        jetpt = ttbarcands.slot0.p4.pt
        jeteta = ttbarcands.slot0.p4.eta
        jetphi = ttbarcands.slot0.p4.phi
        jetmass = ttbarcands.slot0.p4.mass
        jetp = ttbarcands.slot0.p4.p
        
        # plot same jetmass as pre-tagged, anti-tagged jet
        jetmsd = ttbarcands.slot0.msoftdrop
           
        
        # values for mistag rate calculation #
        numerator = np.where(antitag_probe, ttbarcands.slot1.p4.p, -1)
        denominator = np.where(antitag, ttbarcands.slot1.p4.p, -1)
        
        # pt reweighting #
#         if ('TTbar' in dataset):
#             ttbar_wgt = pTReweighting(ttbarcands.slot0.pt, ttbarcands.slot1.pt)
#             weights[correction].add('ptReweighting', ak.flatten(ttbar_wgt))
                 
        if (not self.noSyst) and (not isData) and isNominal:
                    
            if 'pileup' in self.systematics:
                
                puNom, puUp, puDown = GetPUSF(events, self.iov)
                self.weights[correction].add("pileup", 
                    weight=puNom, 
                    weightUp=puUp, 
                    weightDown=puDown,
                           )
                
            logger.debug('memory:%s: pileup systematics %s:%s', time.time(), correction, get_memory_usage())

            if ('prefiring' in self.systematics) and ("L1PreFiringWeight" in events.fields):
                if ('2016' in self.iov) or ('2017' in self.iov):
                
                    prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events)
                    self.weights[correction].add("prefiring", 
                        weight=prefiringNom, 
                        weightUp=prefiringUp, 
                        weightDown=prefiringDown,
                               )
                    
            logger.debug('memory:%s: prefiring systematics %s:%s', time.time(), correction, get_memory_usage())
                    
            if 'pdf' in self.systematics:
                
                pdfUp, pdfDown, pdfNom = GetPDFWeights(events)
                self.weights[correction].add("pdf", 
                    weight=pdfNom, 
                    weightUp=pdfUp, 
                    weightDown=pdfDown,
                           )    
                
            logger.debug('memory:%s: pdf systematics %s:%s', time.time(), correction, get_memory_usage())
            
            if 'q2' in self.systematics:
                
                q2Nom, q2Up, q2Down = GetQ2weights(events)
                
                self.weights[correction].add("q2", 
                    weight=q2Nom, 
                    weightUp=q2Up, 
                    weightDown=q2Down,
                           )   
                
                
            logger.debug('memory:%s: q2 systematics %s:%s', time.time(), correction, get_memory_usage())

                
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
                btag_wgts_up[ak.flatten(btag0)]   = btag_wgts_up_bcats['0b'][ak.flatten(btag0)]
                btag_wgts_down[ak.flatten(btag0)] = btag_wgts_down_bcats['0b'][ak.flatten(btag0)]
                btag_wgts_nom[ak.flatten(btag1)]  = btag_wgts_nom_bcats['1b'][ak.flatten(btag1)]
                btag_wgts_up[ak.flatten(btag1)]   = btag_wgts_up_bcats['1b'][ak.flatten(btag1)]
                btag_wgts_down[ak.flatten(btag1)] = btag_wgts_down_bcats['1b'][ak.flatten(btag1)]
                btag_wgts_nom[ak.flatten(btag2)]  = btag_wgts_nom_bcats['2b'][ak.flatten(btag2)]
                btag_wgts_up[ak.flatten(btag2)]   = btag_wgts_up_bcats['2b'][ak.flatten(btag2)]
                btag_wgts_down[ak.flatten(btag2)] = btag_wgts_down_bcats['2b'][ak.flatten(btag2)]
                
                self.weights[correction].add("btag", 
                    weight=btag_wgts_nom, 
                    weightUp=btag_wgts_up, 
                    weightDown=btag_wgts_down,
                           )
                
                del btag_wgts_nom, btag_wgts_up, btag_wgts_down
                del btag_wgts_nom_bcats, btag_wgts_up_bcats, btag_wgts_down_bcats
                
                
                
                logger.debug('memory:%s: btag systematics %s:%s', time.time(), correction, get_memory_usage())

                
                
            
                            
                



        for i, [ilabel,icat] in enumerate(labels_and_categories.items()):
            
            icat = ak.flatten(icat)
            
            # final cutflow per analysis category
            output['cutflow'][ilabel] += len(events.event[icat])
                                                           

            
            output['jetmass'].fill(
                                   systematic=correction,
                                   anacat = i,
                                   jetmass = ak.flatten(jetmass[icat]),
                                   weight = self.weights[correction].weight()[icat],
                                  )
            output['jetmsd'].fill(
                                   systematic=correction,
                                   anacat = i,
                                   jetmass = ak.flatten(jetmsd[icat]),
                                   weight = self.weights[correction].weight()[icat],
                                  )
            
            output['jetpt'].fill(
                                 systematic=correction,
                                 anacat = i,
                                 jetpt = ak.flatten(jetpt[icat]),
                                 weight = self.weights[correction].weight()[icat],
                                  )
            
            output['jeteta'].fill(
                                  systematic=correction,
                                  anacat = i,
                                  jeteta = ak.flatten(jeteta[icat]),
                                  weight = self.weights[correction].weight()[icat],
                                  )
            output['jetphi'].fill(
                                  systematic=correction,
                                  anacat = i,
                                  jetphi = ak.flatten(jetphi[icat]),
                                  weight = self.weights[correction].weight()[icat],
                                  )
            
            output['mtt_vs_mt'].fill(
                                     systematic=correction,
                                     anacat = i,
                                     jetmass = ak.flatten(jetmsd[icat]),
                                     ttbarmass = ak.flatten(ttbarmass[icat]),
                                     weight = self.weights[correction].weight()[icat],
                                    )
            
            output['ttbarmass'].fill(systematic=correction,
                                         anacat = i,
                                         ttbarmass = ak.flatten(ttbarmass[icat]),
                                         weight = self.weights[correction].weight()[icat],
                                        )
            
            
            
            
            
            
            # save weights
            
            output['weights'][correction] += np.sum(self.weights[correction].weight())
            output['systematics'][correction] += len(events.event[icat])


                
            if isNominal:    
                
#                 output['discriminators'].fill(anacat = i,
#                                           jetp = ak.flatten(jetp[icat]),
#                                           bdisc = ak.flatten(bdisc_s1[icat]),
#                                           tdisc = ak.flatten(tdisc_s1[icat]),
#                                           nsub = ak.flatten(tau32_s1)[icat],
#                                           weight = weights[correction].weight()[icat],
#                                          )

#                 output['deepak8'].fill(anacat = i,
#                                        jetp = ak.flatten(jetp[icat]),
#                                        ttbarmass = ak.flatten(ttbarmass[icat]),
#                                        tdisc = ak.flatten(tdisc_s1[icat]),
#                                        weight = weights[correction].weight()[icat],
#                                       )


                for syst in self.weights[correction].variations:
                    
                    
                    output['weights'][syst] += np.sum(self.weights[correction].weight(syst))
                    output['systematics'][syst] += len(events.event[icat])
                    
                    output['jetmass'].fill(
                                   systematic=syst,
                                   anacat = i,
                                   jetmass = ak.flatten(jetmass[icat]),
                                   weight = self.weights[correction].weight()[icat],
                                  )
                    output['jetmsd'].fill(
                                           systematic=syst,
                                           anacat = i,
                                           jetmass = ak.flatten(jetmsd[icat]),
                                           weight = self.weights[correction].weight()[icat],
                                          )
            
                    output['jetpt'].fill(
                                         systematic=syst,
                                         anacat = i,
                                         jetpt = ak.flatten(jetpt[icat]),
                                         weight = self.weights[correction].weight()[icat],
                                          )

                    output['jeteta'].fill(
                                          systematic=syst,
                                          anacat = i,
                                          jeteta = ak.flatten(jeteta[icat]),
                                          weight = self.weights[correction].weight()[icat],
                                          )
                    output['jetphi'].fill(
                                          systematic=syst,
                                          anacat = i,
                                          jetphi = ak.flatten(jetphi[icat]),
                                          weight = self.weights[correction].weight()[icat],
                                          )

                    output['ttbarmass'].fill(systematic=syst,
                                         anacat = i,
                                         ttbarmass = ak.flatten(ttbarmass[icat]),
                                         weight = self.weights[correction].weight(syst)[icat],
                                        )

                    output['mtt_vs_mt'].fill(systematic=syst,
                                         anacat = i,
                                         ttbarmass = ak.flatten(ttbarmass[icat]),
                                         jetmass = ak.flatten(jetmsd[icat]),
                                         weight = self.weights[correction].weight(syst)[icat],
                                        )
                    
                    
        logger.debug('memory:%s: fill histograms %s:%s', time.time(), correction, get_memory_usage())
                    
                    
                    
                    


        

        return output

    def postprocess(self, accumulator):
        logger.debug('memory:%s: finish processor:%s', time.time(), get_memory_usage())
        return accumulator
        
        
        
